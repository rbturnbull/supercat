"""Adapters that combine existing image losses for WiDiTApp's training hooks."""

import math

import torch

from .metrics import PorosityLoss


class AuxiliaryLoss(torch.nn.Module):
    """Add a weighted auxiliary batch mean to an optional parent criterion."""

    def __init__(self, parent, auxiliary, weight):
        super().__init__()
        self.parent = parent
        self.auxiliary = auxiliary
        self.weight = weight

    def forward(self, prediction, target):
        loss = self.weight * self.auxiliary(prediction, target).mean()
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
        PorosityLoss(temperature=temperature, hard_mask=False),
        porosity_loss_weight,
    )
