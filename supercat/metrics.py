from skimage import filters
import numpy as np
import torch


def calc_porosity(data: np.ndarray | torch.Tensor) -> float:
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    threshold = filters.threshold_otsu(data)

    binary_mask = data < threshold
    total_pixels = data.size
    void_pixels = np.sum(binary_mask)
    return void_pixels / total_pixels


@torch.no_grad()
def porosity_reference(
    target: torch.Tensor,
    temperature: float = 0.05,
    hard_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute scalar Otsu threshold and porosity for one normalized HR sample."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not target.is_floating_point():
        raise TypeError("target must be a floating-point tensor")
    if target.numel() == 0 or not torch.isfinite(target).all():
        raise ValueError("target must be non-empty and contain only finite values")
    target = target.detach().cpu()
    if target.dtype in (torch.float16, torch.bfloat16):
        target = target.float()
    threshold = target.new_tensor(filters.threshold_otsu(target.numpy().reshape(-1)))
    mask = (
        (target < threshold).to(target.dtype)
        if hard_mask
        else torch.sigmoid((threshold - target) / temperature)
    )
    return threshold, mask.mean()


class PorosityLoss(torch.nn.Module):
    """Compare per-sample Otsu porosity ratios using a differentiable mask surrogate.

    Inputs have shape (N, 1, H, W) or (N, 1, D, H, W) and matching intensity
    scales. Each target supplies a detached Otsu threshold shared with its
    prediction. The default mask is sigmoid((threshold - image) / temperature).
    Temperature is in image intensity units; 0.05 assumes normalized [-1, 1]
    images and should be tuned on validation data.

    With ``hard_mask=True``, forward porosities use binary masks, while backward
    uses sigmoid derivatives (a straight-through estimator, not the derivative
    of hard thresholding). Otsu and the target never receive gradients.

    Smooth L1 compares predicted_porosity / target_porosity against 1.
    Below ``eps`` target porosity, use the stabilized relative error
    (predicted_porosity - target_porosity) / eps instead. This gives zero
    loss for matching zero porosities. ``beta`` is a relative-error fraction,
    so the default 0.01 enters the linear regime at one percent error.
    ``reduction='none'`` returns one loss per sample.

    Preferred call: loss(prediction, hr_threshold, hr_porosity), with one threshold
    and porosity per sample. Precompute these with porosity_reference using the
    same temperature and hard_mask settings as this loss. This path never runs
    Otsu or reads HR pixels; metadata is detached and moved to the prediction's
    device. The two-argument loss(prediction, target) remains supported, but runs
    Otsu on CPU. This module does not register itself with the training logger.
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

    def forward(
        self,
        prediction: torch.Tensor,
        hr_threshold: torch.Tensor,
        hr_porosity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compare predicted porosity with precomputed HR references or an HR image."""
        target = hr_threshold if hr_porosity is None else None
        if target is not None and prediction.shape != target.shape:
            raise ValueError("prediction and target must have matching shapes")
        if prediction.ndim not in (4, 5) or prediction.shape[1] != 1:
            raise ValueError("Expected batched single-channel 2D images or 3D volumes")
        if prediction.numel() == 0:
            raise ValueError("prediction must be non-empty")
        if not prediction.is_floating_point():
            raise TypeError("prediction must be a floating-point tensor")
        if target is not None:
            if not target.is_floating_point():
                raise TypeError("target must be a floating-point tensor")
            if prediction.device != target.device:
                raise ValueError("prediction and target must be on the same device")
            references = [
                porosity_reference(sample, self.temperature, self.hard_mask)
                for sample in target
            ]
            hr_threshold = torch.stack([reference[0] for reference in references])
            hr_porosity = torch.stack([reference[1] for reference in references])

        dtype = prediction.dtype
        for reference in (hr_threshold, hr_porosity):
            if (
                not isinstance(reference, torch.Tensor)
                or not reference.is_floating_point()
            ):
                raise TypeError("HR references must be floating-point tensors")
            if reference.numel() != prediction.shape[0]:
                raise ValueError("Expected one HR threshold and porosity per sample")
            dtype = torch.promote_types(dtype, reference.dtype)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        prediction = prediction.to(dtype=dtype)
        threshold = (
            hr_threshold.detach()
            .to(device=prediction.device, dtype=dtype)
            .reshape((-1,) + (1,) * (prediction.ndim - 1))
        )
        target_porosity = (
            hr_porosity.detach().to(device=prediction.device, dtype=dtype).reshape(-1)
        )
        if not (
            torch.isfinite(prediction).all()
            and torch.isfinite(threshold).all()
            and torch.isfinite(target_porosity).all()
        ):
            raise ValueError(
                "prediction and HR references must contain only finite values"
            )
        if ((target_porosity < 0) | (target_porosity > 1)).any():
            raise ValueError("HR porosity must be a fraction in [0, 1]")
        predicted_mask = torch.sigmoid((threshold - prediction) / self.temperature)
        if self.hard_mask:
            binary = (prediction < threshold).to(dtype=dtype)
            predicted_mask = binary + (predicted_mask - predicted_mask.detach())
        predicted_porosity = predicted_mask.flatten(1).mean(1)
        relative_error = (predicted_porosity - target_porosity) / target_porosity.clamp_min(
            self.eps
        )
        return torch.nn.functional.smooth_l1_loss(
            relative_error,
            torch.zeros_like(relative_error),
            beta=self.beta,
            reduction=self.reduction,
        )
