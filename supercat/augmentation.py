import random
import torch


def apply_transformation(tensor, sym_id):
    match(sym_id):
        case 0:
            return tensor  # Identity
        case 1:
            return tensor.flip(-2).transpose(-1, -2)  # 90° rotation
        case 2:
            return tensor.flip(-2).flip(-1)           # 180° rotation
        case 3:
            return tensor.flip(-2).transpose(-1, -2).flip(-1)  # TR-BL diagonal reflection
        case 4:
            return tensor.flip(-1)                    # Vertical reflection
        case 5:
            return tensor.flip(-2)                    # Horizontal reflection
        case 6:
            return tensor.flip(-1).transpose(-1, -2)  # TL-BR diagonal reflection
        case 7:
            return tensor.flip(-1).flip(-2).transpose(-1, -2)  # TR-BL diagonal reflection with both flips
        case 8:
            return tensor.transpose(-3, -2).flip(-2)  # Swap depth/height, flip height
        case 9:
            return tensor.transpose(-3, -2)           # Swap depth/height (no flip)
        case 10:
            return tensor.transpose(-3, -2).flip(-1)  # Swap depth/height, flip width
        case 11:
            return tensor.transpose(-3, -1).flip(-1)  # Swap depth/width, flip width
        case 12:
            return tensor.transpose(-3, -1)           # Swap depth/width (no flip)
        case 13:
            return tensor.transpose(-3, -1).flip(-2)  # Swap depth/width, flip height
        case 14:
            return tensor.flip(-3)                    # Flip depth only
        case 15:
            return tensor.flip(-3).flip(-1)           # Flip depth and width
        case 16:
            return tensor.flip(-3).flip(-2)           # Flip depth and height
        case 17:
            return tensor.flip(-3).flip(-2).flip(-1)  # Flip depth, height, and width
        case 18:
            return tensor.transpose(-2, -1).flip(-3)  # Transpose height/width, then flip depth
        case 19:
            return tensor.transpose(-2, -1).flip(-1).flip(-3)  # Transpose height/width, flip width and depth
        case 20:
            return tensor.transpose(-2, -1).flip(-2).flip(-3)  # Transpose height/width, flip height and depth
        case 21:
            return tensor.transpose(-2, -1).flip(-1).flip(-2).flip(-3)  # Transpose height/width, flip all axes
        case 22:
            return tensor.transpose(-3, -2).flip(-1).flip(-3)  # Swap depth/height, flip width and depth
        case 23:
            return tensor.transpose(-3, -2).flip(-2).flip(-3)  # Swap depth/height, flip height and depth
        case 24:
            return tensor.transpose(-3, -1).flip(-2).flip(-3)  # Swap depth/width, flip height and depth
        case 25:
            return tensor.transpose(-1, -3).flip(-1).flip(-2)  # Swap width/depth, flip width and height
        case 26:
            return tensor.transpose(-1, -3).flip(-3)  # Swap width/depth, flip depth
        case 27:
            return tensor.flip(-1).transpose(-3, -2).flip(-2)  # Flip width, swap depth/height, and flip height
        case 28:
            return tensor.transpose(-3, -1).flip(-1).flip(-3)  # Swap depth/width, flip width and depth
        case 29:
            return tensor.transpose(-1, -2).transpose(-2, -3)  # Cyclic permutation of axes (depth -> width -> height -> depth)
        case 30:
            return tensor.transpose(-2, -3).flip(-2)           # Rotate height/depth diagonals
        case 31:
            return tensor.transpose(-2, -3).flip(-3)           # Rotate depth diagonals, flip depth
        case 32:
            return tensor.transpose(-3, -2).flip(-1).flip(-2)  # Swap depth/height, flip width and height
        case 33:
            return tensor.transpose(-2, -1).flip(-3).flip(-1)  # Transpose height/width, flip depth and width
        case 34:
            return tensor.transpose(-3, -2).flip(-1).flip(-1)  # Swap depth/height, flip width twice (no effect on width)
        case 35:
            return tensor.transpose(-2, -3).flip(-1).flip(-2)  # Rotate depth diagonals, flip height and width
        case 36:
            return tensor.transpose(-1, -3).flip(-1).flip(-2)  # Rotate depth diagonals, flip width and height
        case 37:
            return tensor.transpose(-3, -2).flip(-1).flip(-3)  # Swap depth/height, flip depth and width
        case 38:
            return tensor.transpose(-1, -2).flip(-3)           # Rotate height/width, flip depth
        case 39:
            return tensor.transpose(-1, -3).flip(-1).flip(-1)  # Rotate depth diagonals, double flip width (no effect on width)
        case 40:
            return tensor.flip(-2).transpose(-3, -1).flip(-3)  # Rotate depth diagonals, flip depth and height
        case 41:
            return tensor.flip(-1).transpose(-2, -3).flip(-2)  # Rotate height/depth, flip height and depth
        case 42:
            return tensor.transpose(-2, -1).flip(-1).flip(-3)  # Transpose height/width, flip width and depth
        case 43:
            return tensor.transpose(-3, -2).flip(-3).flip(-1)  # Swap depth/height, flip depth and width
        case 44:
            return tensor.transpose(-3, -1).flip(-3).flip(-2)  # Swap depth/width, flip depth and height
        case 45:
            return tensor.flip(-2).transpose(-3, -1).flip(-1)  # Rotate height/depth diagonals, flip width
        case 46:
            return tensor.flip(-1).flip(-2).transpose(-1, -3)  # Rotate width/depth diagonals, flip height
        case 47:
            return tensor.transpose(-3, -2).flip(-2).flip(-1)  # Rotate depth diagonals, flip width and height        
        case _:
            raise ValueError(f"Invalid sym_id: {sym_id}")


def flip_and_rotate(batch:tuple[torch.Tensor,torch.Tensor,torch.Tensor], sym_id:int|None=None) -> tuple[torch.Tensor,torch.Tensor]:
    # Randomly choose one of the transformations
    input, target, residual = batch
    ndim = input.ndim  # Determine if 2D (4D) or 3D (5D)
        
    is_2d = (ndim == 4)

    number_of_transforms = 8 if is_2d else 48
    if sym_id is None:
        sym_id = random.randint(0, number_of_transforms - 1)
    
    assert 0 <= sym_id < number_of_transforms

    # No transformation
    if sym_id == 0:
        return batch
    
    input = apply_transformation(input, sym_id)
    target = apply_transformation(target, sym_id)
    residual = apply_transformation(residual, sym_id)

    return input, target, residual

