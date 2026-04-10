from skimage import filters
import numpy as np
import torch


def calc_porosity(data:np.ndarray|torch.Tensor) -> float:
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    threshold = filters.threshold_otsu(data)

    binary_mask = data < threshold
    total_pixels = data.size
    void_pixels = np.sum(binary_mask)
    return void_pixels / total_pixels
