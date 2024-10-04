import pytest
import torch
from supercat.augmentation import flip_and_rotate

@pytest.fixture
def setup_2d_tensors():
    """Fixture to set up 2D tensors for input, target, and residual."""
    tensor_2d_input = torch.arange(16).reshape(1, 1, 4, 4)  # Shape: (batch_size, channels, height, width)
    tensor_2d_target = tensor_2d_input * 10
    tensor_2d_residual = tensor_2d_target - tensor_2d_input
    return tensor_2d_input, tensor_2d_target, tensor_2d_residual

@pytest.fixture
def setup_3d_tensors():
    """Fixture to set up 3D tensors for input, target, and residual."""
    tensor_3d_input = torch.arange(64).reshape(1, 1, 4, 4, 4)  # Shape: (batch_size, channels, depth, height, width)
    tensor_3d_target = tensor_3d_input * 10
    tensor_3d_residual = tensor_3d_target - tensor_3d_input
    return tensor_3d_input, tensor_3d_target, tensor_3d_residual


@pytest.mark.parametrize("sym_id", range(8))
def test_2d_transformations(setup_2d_tensors, sym_id):
    """Test different transformations for 2D tensors using sym_id."""
    tensor_2d_input, tensor_2d_target, tensor_2d_residual = setup_2d_tensors
    
    # Apply the transformation for the given sym_id
    input, target, residual = flip_and_rotate((tensor_2d_input, tensor_2d_target, tensor_2d_residual), sym_id=sym_id)
    
    # Check that the shapes remain the same
    assert input.shape == tensor_2d_input.shape
    assert target.shape == tensor_2d_target.shape
    assert residual.shape == tensor_2d_residual.shape

@pytest.mark.parametrize("sym_id", range(24))
def test_3d_transformations(setup_3d_tensors, sym_id):
    """Test different transformations for 3D tensors using sym_id."""
    tensor_3d_input, tensor_3d_target, tensor_3d_residual = setup_3d_tensors
    
    # Apply the transformation for the given sym_id
    input, target, residual = flip_and_rotate((tensor_3d_input, tensor_3d_target, tensor_3d_residual), sym_id=sym_id)
    
    # Check that the shapes remain the same
    assert input.shape == tensor_3d_input.shape
    assert target.shape == tensor_3d_target.shape
    assert residual.shape == tensor_3d_residual.shape

@pytest.mark.parametrize("sym_id", [0, 1, 2, 3])  # Specific to rotation cases
def test_2d_rotation_preserves_structure(setup_2d_tensors, sym_id):
    """Test that specific rotation transformations preserve structure for 2D tensors."""
    tensor_2d_input, tensor_2d_target, tensor_2d_residual = setup_2d_tensors
    
    # Apply the transformation for the given sym_id
    input, target, residual = flip_and_rotate((tensor_2d_input, tensor_2d_target, tensor_2d_residual), sym_id=sym_id)
    
    # Check that rotations change the content but preserve shape
    assert input.shape == tensor_2d_input.shape
    assert target.shape == tensor_2d_target.shape
    assert residual.shape == tensor_2d_residual.shape
    if sym_id == 0:
        assert torch.equal(input, tensor_2d_input)
    else:
        assert not torch.equal(input, tensor_2d_input)  # Ensure it's not just the original tensor

@pytest.mark.parametrize("sym_id", [4, 5, 6, 7])  # Specific to flipping cases
def test_2d_flip_preserves_structure(setup_2d_tensors, sym_id):
    """Test that specific flip transformations preserve structure for 2D tensors."""
    tensor_2d_input, tensor_2d_target, tensor_2d_residual = setup_2d_tensors
    
    # Apply the transformation for the given sym_id
    input, target, residual = flip_and_rotate((tensor_2d_input, tensor_2d_target, tensor_2d_residual), sym_id=sym_id)
    
    # Check that flips change the content but preserve shape
    assert input.shape == tensor_2d_input.shape
    assert target.shape == tensor_2d_target.shape
    assert residual.shape == tensor_2d_residual.shape
    assert not torch.equal(input, tensor_2d_input)  # Ensure it's not just the original tensor

@pytest.mark.parametrize("sym_id", [0, 1, 2, 3, 4, 5])  # Covering a few transformations for 3D
def test_3d_transformations_structure(setup_3d_tensors, sym_id):
    """Test that 3D transformations preserve structure and shape."""
    tensor_3d_input, tensor_3d_target, tensor_3d_residual = setup_3d_tensors
    
    # Apply the transformation for the given sym_id
    input, target, residual = flip_and_rotate((tensor_3d_input, tensor_3d_target, tensor_3d_residual), sym_id=sym_id)
    
    # Check that shapes remain the same
    assert input.shape == tensor_3d_input.shape
    assert target.shape == tensor_3d_target.shape
    assert residual.shape == tensor_3d_residual.shape
    if sym_id == 0:
        assert torch.equal(input, tensor_3d_input)
    else:
        assert not torch.equal(input, tensor_3d_input)  # Ensure content has changed

@pytest.mark.parametrize("sym_id", [16, 17, 18, 19])  # Flips along depth for 3D
def test_3d_flip_depth(setup_3d_tensors, sym_id):
    """Test that flipping along depth (axis -3) works for 3D tensors."""
    tensor_3d_input, tensor_3d_target, tensor_3d_residual = setup_3d_tensors
    
    # Apply the transformation for the given sym_id
    input, target, residual = flip_and_rotate((tensor_3d_input, tensor_3d_target, tensor_3d_residual), sym_id=sym_id)
    
    # Check that shapes remain the same
    assert input.shape == tensor_3d_input.shape
    assert target.shape == tensor_3d_target.shape
    assert residual.shape == tensor_3d_residual.shape
    assert not torch.equal(input, tensor_3d_input)  # Ensure content has changed


def test_unique_2d(setup_2d_tensors):
    """Test that all 2D transformations are unique."""
    tensor_2d_input, tensor_2d_target, tensor_2d_residual = setup_2d_tensors
    transformed_tensors = []
    for sym_id in range(8):
        input, target, residual = flip_and_rotate((tensor_2d_input, tensor_2d_target, tensor_2d_residual), sym_id=sym_id)
        transformed_tensors.append(input.flatten())
    stacked_tensors = torch.stack(transformed_tensors)
    unique_tensors = torch.unique(stacked_tensors, dim=0)
    assert len(unique_tensors) == len(stacked_tensors) == 8


def test_unique_3d(setup_3d_tensors):
    """Test that all 3D transformations are unique."""
    tensor_3d_input, tensor_3d_target, tensor_3d_residual = setup_3d_tensors
    transformed_tensors = []
    for sym_id in range(24):
        input, target, residual = flip_and_rotate((tensor_3d_input, tensor_3d_target, tensor_3d_residual), sym_id=sym_id)
        transformed_tensors.append(input.flatten())
    stacked_tensors = torch.stack(transformed_tensors)
    unique_tensors = torch.unique(stacked_tensors, dim=0)
    assert len(unique_tensors) == len(stacked_tensors) == 24


def test_unique_all_2d(setup_2d_tensors):
    """Test that all 2D transformations are unique."""
    tensor_2d_input, tensor_2d_target, tensor_2d_residual = setup_2d_tensors
    transformed_tensors = []
    for sym_id in range(8):
        input, target, residual = flip_and_rotate((tensor_2d_input, tensor_2d_target, tensor_2d_residual), sym_id=sym_id)
        transformed_tensors.append(input.flatten())
        transformed_tensors.append(target.flatten())
        transformed_tensors.append(residual.flatten())
    stacked_tensors = torch.stack(transformed_tensors)
    unique_tensors = torch.unique(stacked_tensors, dim=0)
    assert len(unique_tensors) == len(stacked_tensors) == 8*3


def test_unique_all_3d(setup_3d_tensors):
    """Test that all 3D transformations are unique."""
    tensor_3d_input, tensor_3d_target, tensor_3d_residual = setup_3d_tensors
    transformed_tensors = []
    for sym_id in range(24):
        input, target, residual = flip_and_rotate((tensor_3d_input, tensor_3d_target, tensor_3d_residual), sym_id=sym_id)
        transformed_tensors.append(input.flatten())
        transformed_tensors.append(target.flatten())
        transformed_tensors.append(residual.flatten())
    stacked_tensors = torch.stack(transformed_tensors)
    unique_tensors = torch.unique(stacked_tensors, dim=0)
    assert len(unique_tensors) == len(stacked_tensors) == 24*3
