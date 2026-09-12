import numpy as np
import torch


def otsu_threshold(data: torch.Tensor, nbins: int = 256) -> torch.Tensor:
    """Batched Otsu threshold, one per sample, on the input's own device.

    ``data`` is (N, ...); the remaining dimensions are flattened per sample.
    Matches skimage.filters.threshold_otsu to within one histogram bin, but
    keeps the work on device, so training never synchronises to the CPU.

    The threshold comes from an argmax over histogram bins and is therefore not
    differentiable; the result is detached. Constant samples return their own
    value, leaving nothing below them, where skimage rejects the histogram.
    """
    if nbins < 2:
        raise ValueError("nbins must be at least 2")
    flat = data.detach().flatten(1)
    if flat.numel() == 0:
        raise ValueError("data must be non-empty")
    if not flat.is_floating_point():
        flat = flat.float()
    low = flat.min(1, keepdim=True).values
    width = flat.max(1, keepdim=True).values - low
    span = width.clamp_min(torch.finfo(flat.dtype).tiny)
    index = ((flat - low) / span * nbins).long().clamp_(0, nbins - 1)
    counts = torch.zeros(flat.shape[0], nbins, dtype=flat.dtype, device=flat.device)
    counts.scatter_add_(1, index, torch.ones_like(flat))
    edges = torch.arange(nbins + 1, dtype=flat.dtype, device=flat.device)
    centers = low + (edges[:-1] + edges[1:]) / 2 / nbins * span

    # Otsu maximises between-class variance across the split between bins.
    below = counts.cumsum(1)
    above = counts.flip(1).cumsum(1).flip(1)
    weighted = counts * centers
    mean_below = weighted.cumsum(1) / below.clamp_min(1)
    mean_above = weighted.flip(1).cumsum(1).flip(1) / above.clamp_min(1)
    variance = (
        below[:, :-1] * above[:, 1:] * (mean_below[:, :-1] - mean_above[:, 1:]) ** 2
    )
    threshold = centers.gather(1, variance.argmax(1, keepdim=True))
    # A constant sample has no split to find; its own value leaves nothing below it.
    return torch.where(width > 0, threshold, low).squeeze(1)


def calc_porosity(data: np.ndarray | torch.Tensor) -> float:
    """Void fraction of one image or volume, thresholded by its own Otsu value."""
    tensor = torch.as_tensor(data)
    if not tensor.is_floating_point():
        tensor = tensor.float()
    threshold = otsu_threshold(tensor.reshape(1, -1))
    return (tensor < threshold).to(torch.float64).mean().item()


class PorosityLoss(torch.nn.Module):
    """Compare the Otsu porosity of a prediction with that of its target.

    Inputs have shape (N, 1, H, W) or (N, 1, D, H, W) and matching intensity
    scales. Each image is thresholded by its own Otsu value, the same way
    :func:`calc_porosity` measures a saved prediction, so the loss optimises the
    quantity that gets reported. A uniform intensity offset therefore leaves the
    porosity unchanged; the pixel loss is what constrains intensity fidelity.

    The default mask is sigmoid((threshold - image) / temperature). Temperature
    is in image intensity units; 0.05 assumes normalized [-1, 1] images and
    should be tuned on validation data. With ``hard_mask=True``, forward
    porosities use binary masks, while backward uses sigmoid derivatives (a
    straight-through estimator, not the derivative of hard thresholding). Otsu
    thresholds and the target never receive gradients.

    Smooth L1 compares predicted_porosity / target_porosity against 1.
    Below ``eps`` target porosity, use the stabilized relative error
    (predicted_porosity - target_porosity) / eps instead. This gives zero
    loss for matching zero porosities. ``beta`` is a relative-error fraction,
    so the default 0.01 enters the linear regime at one percent error.
    ``reduction='none'`` returns one loss per sample.

    Thresholds are computed on the inputs' own device, so this module never
    synchronises to the CPU and needs no precomputed references.
    """

    def __init__(
        self,
        temperature: float = 0.05,
        beta: float = 0.01,
        eps: float = 1e-3,
        reduction: str = "mean",
        hard_mask: bool = False,
    ):
        super().__init__()
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and non-negative")
        if not np.isfinite(eps) or not 0 < eps <= 1:
            raise ValueError("eps must be a finite porosity fraction in (0, 1]")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.temperature = temperature
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        self.hard_mask = hard_mask

    def porosity(self, image: torch.Tensor) -> torch.Tensor:
        """Soft Otsu porosity per sample, differentiable through the mask."""
        threshold = otsu_threshold(image).reshape((-1,) + (1,) * (image.ndim - 1))
        mask = torch.sigmoid((threshold - image) / self.temperature)
        if self.hard_mask:
            binary = (image < threshold).to(dtype=image.dtype)
            mask = binary + (mask - mask.detach())
        return mask.flatten(1).mean(1)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compare the predicted porosity with the target's own Otsu porosity."""
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have matching shapes")
        if prediction.ndim not in (4, 5) or prediction.shape[1] != 1:
            raise ValueError("Expected batched single-channel 2D images or 3D volumes")
        if prediction.numel() == 0:
            raise ValueError("prediction must be non-empty")
        if not prediction.is_floating_point():
            raise TypeError("prediction must be a floating-point tensor")
        if not target.is_floating_point():
            raise TypeError("target must be a floating-point tensor")
        if prediction.device != target.device:
            raise ValueError("prediction and target must be on the same device")

        dtype = torch.promote_types(prediction.dtype, target.dtype)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        prediction = prediction.to(dtype=dtype)
        target = target.to(dtype=dtype)
        if not (torch.isfinite(prediction).all() and torch.isfinite(target).all()):
            raise ValueError("prediction and target must contain only finite values")

        predicted_porosity = self.porosity(prediction)
        with torch.no_grad():
            target_porosity = self.porosity(target.detach())
        relative_error = (predicted_porosity - target_porosity) / target_porosity.clamp_min(
            self.eps
        )
        return torch.nn.functional.smooth_l1_loss(
            relative_error,
            torch.zeros_like(relative_error),
            beta=self.beta,
            reduction=self.reduction,
        )
