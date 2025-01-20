import math
import numpy as np
import torch.nn.functional as F
from torch import nn
from skimage import filters
import torch

def psnr(residual_prediction, high_res, residual, max=2.0):
    """
    A metric to calculate the peak signal-to-noise ratio.

    It is given in eq. 3 of 
    PSNR = 10 \log_{10}(\frac{I^2}{L_{2_{Loss}}})
    where I = 2 because the HR and SR pixel values are between [-1,1].
    """
    L2 = F.mse_loss(residual_prediction, residual)
    return 10 * math.log10(max**2/L2)


def smooth_l1_loss(residual_prediction, high_res, residual):
    return F.smooth_l1_loss(residual_prediction, residual)


def calc_porosity(data:np.ndarray) -> float:
    threshold = filters.threshold_otsu(data)

    binary_mask = data < threshold
    total_pixels = data.size
    void_pixels = np.sum(binary_mask)
    return void_pixels / total_pixels


def otsu_threshold(images):
    """
    Computes Otsu's threshold for a batch of 2D/3D images using PyTorch (GPU-compatible).
    
    Args:
        images (torch.Tensor): Batch tensor of shape (B, C, H, W) for 2D or (B, C, D, H, W) for 3D.
    
    Returns:
        thresholds (torch.Tensor): Otsu thresholds for each sample in the batch (B, C).
    """
    B, C = images.shape[:2]  # Batch size, channels
    hist_bins = 256  # Number of histogram bins

    # Flatten images to (B*C, -1)
    images_flat = images.reshape(B * C, -1)

    # Scale to [0, hist_bins - 1] and ensure it's within valid range
    images_scaled = (images_flat * (hist_bins - 1)).long()
    images_scaled = torch.clamp(images_scaled, 0, hist_bins - 1)  # Ensure values are valid

    # Compute histogram using bincount
    hist = torch.zeros((B * C, hist_bins), device=images.device, dtype=torch.float32)
    for i in range(B * C):
        hist[i] = torch.bincount(images_scaled[i], minlength=hist_bins).float()

    # Reshape histogram to (B, C, hist_bins)
    hist = hist.view(B, C, hist_bins)

    # Normalize histogram
    total_pixels = images_flat.shape[1]
    hist = hist / total_pixels  # Normalize to probability distribution

    # Compute cumulative sums
    cumulative_sum = torch.cumsum(hist, dim=-1)  # (B, C, bins)
    bin_values = torch.linspace(0, 1, hist_bins, device=images.device).view(1, 1, -1)
    cumulative_mean = torch.cumsum(hist * bin_values, dim=-1)

    # Compute between-class variance
    mean_total = cumulative_mean[:, :, -1]  # (B, C)
    numerator = (mean_total.unsqueeze(-1) * cumulative_sum - cumulative_mean) ** 2
    denominator = cumulative_sum * (1 - cumulative_sum)
    inter_class_variance = numerator / (denominator + 1e-6)  # Avoid division by zero

    # Find the optimal threshold for each sample in batch
    optimal_threshold_idx = torch.argmax(inter_class_variance, dim=-1)
    optimal_threshold = optimal_threshold_idx / (hist_bins - 1)  # Normalize bin index to intensity

    return optimal_threshold  # (B, C)


class OtsuLoss(nn.Module):
    def __init__(self, base_weight:float = 1.0, threshold_addition_weight:float=1.0, scale:float=1.0):
        super().__init__()
        self.base_weight = base_weight
        self.threshold_addition_weight = threshold_addition_weight
        self.scale = scale

    def forward(self, residual_prediction, high_res, residual):
        thresholds = otsu_threshold(high_res)
        weighting = torch.exp( - (high_res - thresholds.unsqueeze(-1).unsqueeze(-1))**2/self.scale ) + self.base_weight

        base_loss = F.smooth_l1_loss(residual_prediction, residual, reduction="none")

        return (weighting * base_loss).mean()