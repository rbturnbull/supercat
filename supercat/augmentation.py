import random
import torch


TRANSFORMATIONS = [
    lambda tensor : tensor,  # Identity
    lambda tensor : tensor.flip(-2).transpose(-1, -2),  # 90° rotation
    lambda tensor : tensor.flip(-2).flip(-1),           # 180° rotation
    lambda tensor : tensor.flip(-2).transpose(-1, -2).flip(-1),  # TR-BL diagonal reflection
    lambda tensor : tensor.flip(-1),                    # Vertical reflection
    lambda tensor : tensor.flip(-2),                    # Horizontal reflection
    lambda tensor : tensor.flip(-1).transpose(-1, -2),  # TL-BR diagonal reflection
    lambda tensor : tensor.flip(-1).flip(-2).transpose(-1, -2),  # TR-BL diagonal reflection with both flips
    lambda tensor : tensor.flip(-3),
    lambda tensor : tensor.flip(-1).flip(-3),
    lambda tensor : tensor.flip(-2).flip(-3),
    lambda tensor : tensor.flip(-1).flip(-2).flip(-3),
    lambda tensor : tensor.transpose(-1, -2).flip(-3),
    lambda tensor : tensor.transpose(-1, -2).flip(-1).flip(-3),
    lambda tensor : tensor.transpose(-1, -2).flip(-2).flip(-3),
    lambda tensor : tensor.transpose(-1, -2).flip(-1).flip(-2).flip(-3),
    lambda tensor : tensor.transpose(-1, -3),
    lambda tensor : tensor.transpose(-1, -3).flip(-1),
    lambda tensor : tensor.transpose(-1, -3).flip(-2),
    lambda tensor : tensor.transpose(-1, -3).flip(-3),
    lambda tensor : tensor.transpose(-1, -3).flip(-1).flip(-2),
    lambda tensor : tensor.transpose(-1, -3).flip(-1).flip(-3),
    lambda tensor : tensor.transpose(-1, -3).flip(-2).flip(-3),
    lambda tensor : tensor.transpose(-1, -3).flip(-1).flip(-2).flip(-3),
    lambda tensor : tensor.transpose(-2, -3),
    lambda tensor : tensor.transpose(-2, -3).flip(-1),
    lambda tensor : tensor.transpose(-2, -3).flip(-2),
    lambda tensor : tensor.transpose(-2, -3).flip(-3),
    lambda tensor : tensor.transpose(-2, -3).flip(-1).flip(-2),
    lambda tensor : tensor.transpose(-2, -3).flip(-1).flip(-3),
    lambda tensor : tensor.transpose(-2, -3).flip(-2).flip(-3),
    lambda tensor : tensor.transpose(-2, -3).flip(-1).flip(-2).flip(-3),
    lambda tensor : tensor.transpose(-2, -3).transpose(-1, -2),
    lambda tensor : tensor.transpose(-2, -3).transpose(-1, -2).flip(-1),
    lambda tensor : tensor.transpose(-2, -3).transpose(-1, -2).flip(-2),
    lambda tensor : tensor.transpose(-2, -3).transpose(-1, -2).flip(-3),
    lambda tensor : tensor.transpose(-2, -3).transpose(-1, -2).flip(-1).flip(-2),
    lambda tensor : tensor.transpose(-2, -3).transpose(-1, -2).flip(-1).flip(-3),
    lambda tensor : tensor.transpose(-2, -3).transpose(-1, -2).flip(-2).flip(-3),
    lambda tensor : tensor.transpose(-2, -3).transpose(-1, -2).flip(-1).flip(-2).flip(-3),
    lambda tensor : tensor.transpose(-1, -2).transpose(-2, -3),
    lambda tensor : tensor.transpose(-1, -2).transpose(-2, -3).flip(-1),
    lambda tensor : tensor.transpose(-1, -2).transpose(-2, -3).flip(-2),
    lambda tensor : tensor.transpose(-1, -2).transpose(-2, -3).flip(-3),
    lambda tensor : tensor.transpose(-1, -2).transpose(-2, -3).flip(-1).flip(-2),
    lambda tensor : tensor.transpose(-1, -2).transpose(-2, -3).flip(-1).flip(-3),
    lambda tensor : tensor.transpose(-1, -2).transpose(-2, -3).flip(-2).flip(-3),
    lambda tensor : tensor.transpose(-1, -2).transpose(-2, -3).flip(-1).flip(-2).flip(-3),
]


def flip_and_rotate(
    batch:tuple[torch.Tensor,torch.Tensor,torch.Tensor], 
    sym_id:int|None=None
) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    """
    Applies a random or specified flip and/or rotation transformation to a batch of 2D or 3D tensors. 
    This function transforms the input, target, and residual tensors by flipping, rotating, and 
    transposing them in a consistent manner.

    Args:
        batch (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing the input, target, 
            and residual tensors to be transformed.
        sym_id (int or None, optional): The transformation identifier. If None, a random transformation 
            is chosen. For 2D tensors, sym_id ranges from 0 to 7; for 3D tensors, it ranges from 0 to 47.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the transformed input, 
            target, and residual tensors after applying the flip and/or rotation.

    Raises:
        AssertionError: If the sym_id is out of the valid range for the tensor dimensions.
    """
    input, target, residual = batch

    # Determine if 2D (4D) or 3D (5D)
    is_2d = (input.ndim == 4)
    number_of_transforms = 8 if is_2d else len(TRANSFORMATIONS)
    
    if sym_id is None:
        # Randomly choose one of the transformations
        sym_id = random.randint(0, number_of_transforms - 1)
    
    # Shortcut for no transformation
    if sym_id == 0:
        return batch

    assert 0 <= sym_id < number_of_transforms    
    transformation = TRANSFORMATIONS[sym_id]
    
    input = transformation(input)
    target = transformation(target)
    residual = transformation(residual)

    return input, target, residual

