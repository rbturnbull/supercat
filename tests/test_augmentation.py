import pytest
import torch
from flip_and_rotate import FlipAndRotate  # Assuming the class is in flip_and_rotate.py

@pytest.fixture
def setup_2d_tensors():
    """Fixture to set up 2D tensors for input, target, and residual."""
    tensor_2d_input = torch.arange(16).reshape(1, 1, 4, 4)  # Shape: (batch_size, channels, height, width)
    tensor_2d_target = torch.arange(16).reshape(1, 1, 4, 4)  # Same shape for the target
    tensor_2d_residual = torch.arange(16).reshape(1, 1, 4, 4)  # Same shape for residual
    return tensor_2d_input, tensor_2d_target, tensor_2d_residual

@pytest.fixture
def setup_3d_tensors():
    """Fixture to set up 3D tensors for input, target, and residual."""
    tensor_3d_input = torch.arange(64).reshape(1, 1, 4, 4, 4)  # Shape: (batch_size, channels, depth, height, width)
    tensor_3d_target = torch.arange(64).reshape(1, 1, 4, 4, 4)  # Same shape for the target
    tensor_3d_residual = torch.arange(64).reshape(1, 1, 4, 4, 4)  # Same shape for residual
    return tensor_3d_input, tensor_3d_target, tensor_3d_residual

@pytest.fixture
def flip_and_rotate():
    """Fixture to initialize FlipAndRotate object."""
    return FlipAndRotate()

@pytest.mark.parametrize("sym_id", range(8))
def test_2d_transformations(setup_2d_tensors, flip_and_rotate, sym_id):
    """Test different transformations for 2D tensors using sym_id."""
    tensor_2d_input, tensor_2d_target, tensor_2d_residual = setup_2d_tensors
    
    # Apply the transformation for the given sym_id
    input, target, residual = flip_and_rotate((tensor_2d_input, tensor_2d_target, tensor_2d_residual), sym_id=sym_id)
    
    # Check that the shapes remain the same
    assert input.shape == tensor_2d_input.shape
    assert target.shape == tensor_2d_target.shape
    assert residual.shape == tensor_2d_residual.shape

@pytest.mark.parametrize("sym_id", range(24))
def test_3d_transformations(setup_3d_tensors, flip_and_rotate, sym_id):
    """Test different transformations for 3D tensors using sym_id."""
    tensor_3d_input, tensor_3d_target, tensor_3d_residual = setup_3d_tensors
    
    # Apply the transformation for the given sym_id
    input, target, residual = flip_and_rotate((tensor_3d_input, tensor_3d_target, tensor_3d_residual), sym_id=sym_id)
    
    # Check that the shapes remain the same
    assert input.shape == tensor_3d_input.shape
    assert target.shape == tensor_3d_target.shape
    assert residual.shape == tensor_3d_residual.shape

@pytest.mark.parametrize("sym_id", [0, 1, 2, 3])  # Specific to rotation cases
def test_2d_rotation_preserves_structure(setup_2d_tensors, flip_and_rotate, sym_id):
    """Test that specific rotation transformations preserve structure for 2D tensors."""
    tensor_2d_input, tensor_2d_target, tensor_2d_residual = setup_2d_tensors
    
    # Apply the transformation for the given sym_id
    input, target, residual = flip_and_rotate((tensor_2d_input, tensor_2d_target, tensor_2d_residual), sym_id=sym_id)
    
    # Check that rotations change the content but preserve shape
    assert input.shape == tensor_2d_input.shape
    assert target.shape == tensor_2d_target.shape
    assert residual.shape == tensor_2d_residual.shape
    assert not torch.equal(input, tensor_2d_input)  # Ensure it's not just the original tensor

@pytest.mark.parametrize("sym_id", [4, 5, 6, 7])  # Specific to flipping cases
def test_2d_flip_preserves_structure(setup_2d_tensors, flip_and_rotate, sym_id):
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
def test_3d_transformations_structure(setup_3d_tensors, flip_and_rotate, sym_id):
    """Test that 3D transformations preserve structure and shape."""
    tensor_3d_input, tensor_3d_target, tensor_3d_residual = setup_3d_tensors
    
    # Apply the transformation for the given sym_id
    input, target, residual = flip_and_rotate((tensor_3d_input, tensor_3d_target, tensor_3d_residual), sym_id=sym_id)
    
    # Check that shapes remain the same
    assert input.shape == tensor_3d_input.shape
    assert target.shape == tensor_3d_target.shape
    assert residual.shape == tensor_3d_residual.shape
    assert not torch.equal(input, tensor_3d_input)  # Ensure content has changed

@pytest.mark.parametrize("sym_id", [16, 17, 18, 19])  # Flips along depth for 3D
def test_3d_flip_depth(setup_3d_tensors, flip_and_rotate, sym_id):
    """Test that flipping along depth (axis -3) works for 3D tensors."""
    tensor_3d_input, tensor_3d_target, tensor_3d_residual = setup_3d_tensors
    
    # Apply the transformation for the given sym_id
    input, target, residual = flip_and_rotate((tensor_3d_input, tensor_3d_target, tensor_3d_residual), sym_id=sym_id)
    
    # Check that shapes remain the same
    assert input.shape == tensor_3d_input.shape
    assert target.shape == tensor_3d_target.shape
    assert residual.shape == tensor_3d_residual.shape
    assert not torch.equal(input, tensor_3d_input)  # Ensure content has changed
