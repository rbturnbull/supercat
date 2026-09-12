"""Adapters that combine existing image losses for WiDiTApp's training hooks."""

import math

import torch

from .metrics import PorosityLoss


def snr_weight(alpha_bar: torch.Tensor) -> torch.Tensor:
    """Return min(1, sqrt(alpha_bar / (1 - alpha_bar))) for each sample.

    WiDiTApp evaluates auxiliary image losses on the predicted clean image. With
    the default EPSILON parameterisation that estimate is recovered from the
    model output by dividing by sqrt(alpha_bar), so gradients reaching the model
    are amplified by sqrt((1 - alpha_bar) / alpha_bar): about 0.01 at the least
    noisy timestep of the default schedule and 157 at the noisiest. Scaling by
    this weight cancels the amplification wherever it exceeds one, which keeps
    the effective weight of the auxiliary loss bounded instead of letting the
    timesteps whose clean-image estimate is mostly noise dominate training.
    """
    if not torch.isfinite(alpha_bar).all():
        raise ValueError("alpha_bar must contain only finite values")
    alpha_bar = alpha_bar.detach().clamp(0.0, 1.0)
    tiny = torch.finfo(alpha_bar.dtype).tiny
    return (alpha_bar / (1.0 - alpha_bar).clamp_min(tiny)).sqrt().clamp(max=1.0)


class AuxiliaryLoss(torch.nn.Module):
    """Add a weighted auxiliary batch mean to an optional parent criterion.

    ``alpha_bar`` holds one cumulative alpha per sample and is supplied by
    WiDiTApp's diffusion training loop; the regression path omits it. When it is
    present each sample's auxiliary loss is scaled by :func:`snr_weight` before
    the batch mean, so declaring the argument here is what opts this module into
    signal-to-noise weighting. The parent criterion is never weighted: it acts on
    the image directly rather than through the clean-image estimate.
    """

    def __init__(self, parent, auxiliary, weight):
        super().__init__()
        self.parent = parent
        self.auxiliary = auxiliary
        self.weight = weight

    def forward(self, prediction, target, alpha_bar=None):
        auxiliary = self.auxiliary(prediction, target)
        if alpha_bar is not None:
            if not isinstance(alpha_bar, torch.Tensor):
                raise TypeError("alpha_bar must be a tensor")
            weights = snr_weight(
                alpha_bar.to(device=auxiliary.device, dtype=auxiliary.dtype)
            )
            if auxiliary.ndim == 0:
                weights = weights.mean()
            elif weights.reshape(-1).shape != auxiliary.shape:
                raise ValueError("Expected one alpha_bar per auxiliary loss")
            auxiliary = auxiliary * weights.reshape(auxiliary.shape)
        loss = self.weight * auxiliary.mean()
        if self.parent is not None:
            loss = loss + self.parent(prediction, target).mean()
        return loss


def add_porosity_loss(parent, porosity_loss_weight, temperature):
    """Keep the parent unchanged at zero weight, otherwise attach PorosityLoss."""
    if not math.isfinite(porosity_loss_weight) or porosity_loss_weight < 0:
        raise ValueError("porosity_loss_weight must be finite and non-negative")
    if porosity_loss_weight == 0:
        return parent
    return AuxiliaryLoss(
        parent,
        PorosityLoss(temperature=temperature, hard_mask=False, reduction="none"),
        porosity_loss_weight,
    )
