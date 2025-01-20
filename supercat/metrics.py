import math
import numpy as np
import torch.nn.functional as F
from skimage import filters

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
