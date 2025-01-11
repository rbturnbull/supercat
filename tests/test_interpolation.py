import pytest
import torch
from supercat.interpolation import bicubic_align_corners, tricubic_align_corners, interpolate_cubic

@pytest.fixture
def sample_2d_tensor():
    """Fixture to provide a simple 2D tensor."""
    return torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ])

@pytest.fixture
def sample_3d_tensor():
    """Fixture to provide a simple 3D tensor."""
    return torch.tensor([
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        [
            [17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0, 28.0],
            [29.0, 30.0, 31.0, 32.0],
        ],
        [
            [33.0, 34.0, 35.0, 36.0],
            [37.0, 38.0, 39.0, 40.0],
            [41.0, 42.0, 43.0, 44.0],
            [45.0, 46.0, 47.0, 48.0],
        ],
        [
            [49.0, 50.0, 51.0, 52.0],
            [53.0, 54.0, 55.0, 56.0],
            [57.0, 58.0, 59.0, 60.0],
            [61.0, 62.0, 63.0, 64.0],
        ],
    ])

def test_bicubic_align_corners_shape(sample_2d_tensor):
    """Test the shape of the interpolated 2D tensor."""
    scale_factor = 2
    result = bicubic_align_corners(sample_2d_tensor, scale_factor)
    expected_shape = (int(sample_2d_tensor.shape[0] * scale_factor),
                      int(sample_2d_tensor.shape[1] * scale_factor))
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

def test_bicubic_align_corners_values(sample_2d_tensor):
    """Test that bicubic interpolation aligns corners correctly."""
    scale_factor = 2
    result = bicubic_align_corners(sample_2d_tensor, scale_factor)

    # Check corners
    assert torch.isclose(result[0, 0], sample_2d_tensor[0, 0]), "Top-left corner mismatch"
    assert torch.isclose(result[-1, 0], sample_2d_tensor[-1, 0]), "Bottom-left corner mismatch"
    assert torch.isclose(result[0, -1], sample_2d_tensor[0, -1]), "Top-right corner mismatch"
    assert torch.isclose(result[-1, -1], sample_2d_tensor[-1, -1]), "Bottom-right corner mismatch"

def test_tricubic_align_corners_shape(sample_3d_tensor):
    """Test the shape of the interpolated 3D tensor."""
    scale_factor = 2
    result = tricubic_align_corners(sample_3d_tensor, scale_factor)
    expected_shape = (int(sample_3d_tensor.shape[0] * scale_factor),
                      int(sample_3d_tensor.shape[1] * scale_factor),
                      int(sample_3d_tensor.shape[2] * scale_factor))
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

def test_tricubic_align_corners_values(sample_3d_tensor):
    """Test that tricubic interpolation aligns corners correctly."""
    scale_factor = 2
    result = tricubic_align_corners(sample_3d_tensor, scale_factor)

    # Check corners
    assert torch.isclose(result[0, 0, 0], sample_3d_tensor[0, 0, 0]), "Top-left-front corner mismatch"
    assert torch.isclose(result[-1, 0, 0], sample_3d_tensor[-1, 0, 0]), "Bottom-left-front corner mismatch"
    assert torch.isclose(result[0, -1, -1], sample_3d_tensor[0, -1, -1]), "Top-right-back corner mismatch"
    assert torch.isclose(result[-1, -1, -1], sample_3d_tensor[-1, -1, -1]), "Bottom-right-back corner mismatch"

def test_interpolate_cubic_bicubic(sample_2d_tensor):
    """Test `interpolate_cubic` for bicubic interpolation."""
    scale_factor = 2
    result = interpolate_cubic(sample_2d_tensor, scale_factor)
    expected_shape = (int(sample_2d_tensor.shape[0] * scale_factor),
                      int(sample_2d_tensor.shape[1] * scale_factor))
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

def test_interpolate_cubic_tricubic(sample_3d_tensor):
    """Test `interpolate_cubic` for tricubic interpolation."""
    scale_factor = 2
    result = interpolate_cubic(sample_3d_tensor, scale_factor)
    expected_shape = (int(sample_3d_tensor.shape[0] * scale_factor),
                      int(sample_3d_tensor.shape[1] * scale_factor),
                      int(sample_3d_tensor.shape[2] * scale_factor))
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

def test_interpolate_cubic_invalid():
    """Test `interpolate_cubic` with invalid input dimensions."""
    invalid_tensor = torch.ones(4, 4, 4, 4)  # Invalid 4D tensor
    scale_factor = 2
    with pytest.raises(ValueError, match="Data must be 2D or 3D for bicubic or tricubic interpolation"):
        interpolate_cubic(invalid_tensor, scale_factor)
