import numpy as np
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

def bicubic_align_corners(data: np.ndarray, scale_factor: float) -> np.ndarray:
    # Input dimensions
    h, w = data.shape

    # Original grid with corner alignment
    x = np.linspace(0, h - 1, h)
    y = np.linspace(0, w - 1, w)

    # New grid with corner alignment
    x_new = np.linspace(0, h - 1, int(h * scale_factor))
    y_new = np.linspace(0, w - 1, int(w * scale_factor))

    # Interpolate
    spline = RectBivariateSpline(x, y, data)
    return spline(x_new, y_new)

def tricubic_align_corners(data: np.ndarray, scale_factor: float) -> np.ndarray:
    # Input dimensions
    d, h, w = data.shape

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
    interp_func = RegularGridInterpolator((x, y, z), data, method="cubic")  # "cubic" for tricubic

    # Interpolated data
    points = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=-1)
    return interp_func(points).reshape(xv.shape)

def interpolate_cubic(data: np.ndarray, scale_factor: float) -> np.ndarray:
    if data.ndim == 2:  # Bicubic
        return bicubic_align_corners(data, scale_factor)
    elif data.ndim == 3:  # Tricubic
        return tricubic_align_corners(data, scale_factor)
    else:
        raise ValueError("Data must be 2D or 3D for bicubic or tricubic interpolation")
