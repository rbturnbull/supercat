import pytest
import numpy as np
from supercat.interpolation import bicubic_align_corners, tricubic_align_corners, interpolate_cubic

@pytest.fixture
def sample_2d_data():
    """Fixture to provide a 2D array with sufficient points for cubic interpolation."""
    return np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ])


@pytest.fixture
def sample_3d_data():
    """Fixture to provide a 3D array with sufficient points for cubic interpolation."""
    return np.array([
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        [[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]],
        [[33, 34, 35, 36], [37, 38, 39, 40], [41, 42, 43, 44], [45, 46, 47, 48]],
        [[49, 50, 51, 52], [53, 54, 55, 56], [57, 58, 59, 60], [61, 62, 63, 64]],
    ])


def test_bicubic_align_corners_correctness(sample_2d_data):
    """Test that bicubic interpolation aligns corners correctly."""
    scale_factor = 2
    result = bicubic_align_corners(sample_2d_data, scale_factor)

    # Check corners
    assert np.isclose(result[0, 0], sample_2d_data[0, 0]), "Top-left corner mismatch"
    assert np.isclose(result[-1, 0], sample_2d_data[-1, 0]), "Bottom-left corner mismatch"
    assert np.isclose(result[0, -1], sample_2d_data[0, -1]), "Top-right corner mismatch"
    assert np.isclose(result[-1, -1], sample_2d_data[-1, -1]), "Bottom-right corner mismatch"


def test_tricubic_align_corners_correctness(sample_3d_data):
    """Test that tricubic interpolation aligns corners correctly."""
    scale_factor = 2
    result = tricubic_align_corners(sample_3d_data, scale_factor)

    # Check corners
    assert np.isclose(result[0, 0, 0], sample_3d_data[0, 0, 0]), "Top-left-front corner mismatch"
    assert np.isclose(result[-1, 0, 0], sample_3d_data[-1, 0, 0]), "Bottom-left-front corner mismatch"
    assert np.isclose(result[0, -1, -1], sample_3d_data[0, -1, -1]), "Top-right-back corner mismatch"
    assert np.isclose(result[-1, -1, -1], sample_3d_data[-1, -1, -1]), "Bottom-right-back corner mismatch"


def test_bicubic_interpolation_values(sample_2d_data):
    """Test that bicubic interpolation gives expected cubic interpolated values."""
    scale_factor = 2
    result = bicubic_align_corners(sample_2d_data, scale_factor)

    # Check an interpolated value (e.g., center point)
    interpolated_center = result[result.shape[0] // 2, result.shape[1] // 2]
    assert interpolated_center > sample_2d_data.min() and interpolated_center < sample_2d_data.max()


def test_tricubic_interpolation_values(sample_3d_data):
    """Test that tricubic interpolation gives expected cubic interpolated values."""
    scale_factor = 2
    result = tricubic_align_corners(sample_3d_data, scale_factor)

    # Check an interpolated value (e.g., center point)
    interpolated_center = result[
        result.shape[0] // 2, result.shape[1] // 2, result.shape[2] // 2
    ]
    assert interpolated_center > sample_3d_data.min() and interpolated_center < sample_3d_data.max()
