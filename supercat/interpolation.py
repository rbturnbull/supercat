import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator

def bicubic_align_corners(data: Tensor, scale_factor: float) -> Tensor:
    if data.ndim != 2:
        raise ValueError("Input for bicubic interpolation must be 2D (height, width).")

    # Add batch and channel dimensions for PyTorch interpolation
    data = data.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    result = F.interpolate(data, scale_factor=scale_factor, mode="bicubic", align_corners=True)
    return result.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

def tricubic_align_corners(data: Tensor, scale_factor: float) -> Tensor:
    if data.ndim != 3:
        raise ValueError("Input for tricubic interpolation must be 3D (depth, height, width).")

    # Convert tensor to NumPy for Scipy interpolation
    data_np = data.cpu().numpy()
    d, h, w = data_np.shape

    # Original grid with corner alignment
    x = np.linspace(0, d - 1, d)
    y = np.linspace(0, h - 1, h)
    z = np.linspace(0, w - 1, w)

    # New grid with corner alignment
    x_new = np.linspace(0, d - 1, int(d * scale_factor))
    y_new = np.linspace(0, h - 1, int(h * scale_factor))
    z_new = np.linspace(0, w - 1, int(w * scale_factor))
    xv, yv, zv = np.meshgrid(x_new, y_new, z_new, indexing="ij")

    # Interpolation function
    interp_func = RegularGridInterpolator((x, y, z), data_np, method="cubic")

    # Interpolated data
    points = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=-1)
    result_np = interp_func(points).reshape(xv.shape)

    # Convert back to PyTorch tensor with the original dtype and device
    return torch.tensor(result_np, dtype=data.dtype, device=data.device)

def interpolate_cubic(data: Tensor, scale_factor: float) -> Tensor:
    if data.ndim == 2:  # Bicubic
        return bicubic_align_corners(data, scale_factor)
    elif data.ndim == 3:  # Tricubic
        return tricubic_align_corners(data, scale_factor)
    else:
        raise ValueError("Data must be 2D or 3D for bicubic or tricubic interpolation.")
