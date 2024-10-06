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
            return tensor.flip(-3)
        case 9:
            return tensor.flip(-1).flip(-3)
        case 10:
            return tensor.flip(-2).flip(-3)
        case 11:
            return tensor.flip(-1).flip(-2).flip(-3)
        case 12:
            return tensor.transpose(-1, -2).flip(-3)
        case 13:
            return tensor.transpose(-1, -2).flip(-1).flip(-3)
        case 14:
            return tensor.transpose(-1, -2).flip(-2).flip(-3)
        case 15:
            return tensor.transpose(-1, -2).flip(-1).flip(-2).flip(-3)
        case 16:
            return tensor.transpose(-1, -3)
        case 17:
            return tensor.transpose(-1, -3).flip(-1)
        case 18:
            return tensor.transpose(-1, -3).flip(-2)
        case 19:
            return tensor.transpose(-1, -3).flip(-3)
        case 20:
            return tensor.transpose(-1, -3).flip(-1).flip(-2)
        case 21:
            return tensor.transpose(-1, -3).flip(-1).flip(-3)
        case 22:
            return tensor.transpose(-1, -3).flip(-2).flip(-3)
        case 23:
            return tensor.transpose(-1, -3).flip(-1).flip(-2).flip(-3)
        case 24:
            return tensor.transpose(-2, -3)
        case 25:
            return tensor.transpose(-2, -3).flip(-1)
        case 26:
            return tensor.transpose(-2, -3).flip(-2)
        case 27:
            return tensor.transpose(-2, -3).flip(-3)
        case 28:
            return tensor.transpose(-2, -3).flip(-1).flip(-2)
        case 29:
            return tensor.transpose(-2, -3).flip(-1).flip(-3)
        case 30:
            return tensor.transpose(-2, -3).flip(-2).flip(-3)
        case 31:
            return tensor.transpose(-2, -3).flip(-1).flip(-2).flip(-3)
        case 32:
            return tensor.transpose(-2, -3).transpose(-1, -2)
        case 33:
            return tensor.transpose(-2, -3).transpose(-1, -2).flip(-1)
        case 34:
            return tensor.transpose(-2, -3).transpose(-1, -2).flip(-2)
        case 35:
            return tensor.transpose(-2, -3).transpose(-1, -2).flip(-3)
        case 36:
            return tensor.transpose(-2, -3).transpose(-1, -2).flip(-1).flip(-2)
        case 37:
            return tensor.transpose(-2, -3).transpose(-1, -2).flip(-1).flip(-3)
        case 38:
            return tensor.transpose(-2, -3).transpose(-1, -2).flip(-2).flip(-3)
        case 39:
            return tensor.transpose(-2, -3).transpose(-1, -2).flip(-1).flip(-2).flip(-3)
        case 40:
            return tensor.transpose(-1, -2).transpose(-2, -3)
        case 41:
            return tensor.transpose(-1, -2).transpose(-2, -3).flip(-1)
        case 42:
            return tensor.transpose(-1, -2).transpose(-2, -3).flip(-2)
        case 43:
            return tensor.transpose(-1, -2).transpose(-2, -3).flip(-3)
        case 44:
            return tensor.transpose(-1, -2).transpose(-2, -3).flip(-1).flip(-2)
        case 45:
            return tensor.transpose(-1, -2).transpose(-2, -3).flip(-1).flip(-3)
        case 46:
            return tensor.transpose(-1, -2).transpose(-2, -3).flip(-2).flip(-3)
        case 47:
            return tensor.transpose(-1, -2).transpose(-2, -3).flip(-1).flip(-2).flip(-3)        
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

