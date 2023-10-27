import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from scipy import ndimage as ndi
from skimage.restoration import denoise_nl_means
from skimage.filters import threshold_otsu


def generate_distance_map(original_image, signed=False):
    """
    precondition: original image
    postcondition: distance map computed based on binary segmentation with
        Otsu's thresholding on the non-local mean smoothed original image
    """

    smoothed_image = denoise_nl_means(original_image, fast_mode=True, patch_size=5, patch_distance=6)
    threshold = threshold_otsu(smoothed_image)
    
    foreground = (smoothed_image > threshold) * 1
    background = (foreground - 1) * (-1)
    
    foreground_sdf = ndi.distance_transform_edt(foreground)
    background_sdf = ndi.distance_transform_edt(background)
    distance_map = foreground_sdf + background_sdf

    if signed:
        distance_map = (-1) * foreground_sdf + background_sdf

    return distance_map


def generate_re_weighting(distance_map, percentile=50, alpha=0.5) -> np.ndarray:
    """
    precondition: unsigned distance map, the % lowest value, alpha coefficient 
        for the re weighting map
    postcondition: re weighting map, calculated based on distnce map and 
        parameters of percentile and minimum values
    """
    
    # extract the value at desired percentileS
    flat_vector = distance_map.flatten()
    clip_value = np.sort(flat_vector)[int(len(flat_vector) * percentile / 100)]
    
    # cap the distance map
    clipped_distance_map = np.clip(distance_map, 0, clip_value)
    
    # compute the weights
    distance_coef_map = (clip_value - clipped_distance_map + 1) / clip_value
    weights_map = distance_coef_map * (1 - alpha) + alpha
    
    return weights_map


# example usage




# plan for initial experiments
# 1. investigate the effect of choice of alpha, [0.3, 0.5, 0.7], while keeping percentile=50
# 2. investigate the effect of choice of percentile, [30, 50, 70], while keeping alpha=0.5



class EdgeLoss(nn.Module):
    def __init__(self,
        percentile:float,
        alpha:float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.percentile = percentile
        self.alpha = alpha

    def forward(self, prediction, target):
        # Calculate the edge weights
        weights_map = torch.zeros_like(target)
        for sample_index, sample in enumerate(target):
            distance_map = generate_distance_map(sample)
            w = generate_re_weighting(distance_map, percentile=self.percentile, alpha=self.alpha)
            weights_map[sample_index] = torch.as_tensor(w)

        # Calculate the raw loss
        loss = weights_map * F.smooth_l1_loss(prediction, target, reduction="none")

        return loss.mean()