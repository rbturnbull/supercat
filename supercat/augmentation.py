import random
import torch

def rotate_z(tensor):  # 90° rotation around z-axis (h -> -w, w -> h)
    return tensor.transpose(-1, -2).flip(-1)

def rotate_x(tensor):  # 90° rotation around x-axis (3D: d -> -h, h -> d)
    return tensor.transpose(-2, -3).flip(-2)

def rotate_y(tensor):  # 90° rotation around y-axis (3D: w -> -d, d -> w)
    return tensor.transpose(-1, -3).flip(-1)

def flip_along(tensor, axes):  # General flip along specified axes
    for axis in axes:
        tensor = tensor.flip(axis)
    return tensor


class FlipAndRotate():
    def __init__(self):    
        # Define transformations (first 8 for both 2D and 3D, next 16 only for 3D)
        self.transforms = (
            (None, []),                    # No transformation
            (rotate_z, []),              # 90° z-rotation
            (lambda t: rotate_z(rotate_z(t)), []),  # 180° z-rotation
            (lambda t: rotate_z(rotate_z(rotate_z(t))), []),  # 270° z-rotation
            (None, [-1]),                  # Flip along width
            (None, [-2]),                  # Flip along height
            (rotate_z, [-1]),            # 90° z-rotation + flip width
            (lambda t: rotate_z(rotate_z(t)), [-2]),  # 180° z-rotation + flip height
            (rotate_x, []),              # 90° x-rotation
            (lambda t: rotate_x(rotate_x(t)), []),  # 180° x-rotation
            (lambda t: rotate_x(rotate_x(rotate_x(t))), []),  # 270° x-rotation
            (rotate_y, []),              # 90° y-rotation
            (lambda t: rotate_y(rotate_y(t)), []),  # 180° y-rotation
            (lambda t: rotate_y(rotate_y(rotate_y(t))), []),  # 270° y-rotation
            (rotate_z, [-3]),            # 90° z-rotation + flip depth
            (rotate_z, [-2]),            # 90° z-rotation + flip height
            (None, [-3]),                  # Flip along depth
            (rotate_y, [-3]),            # 90° y-rotation + flip depth
            (lambda t: rotate_x(rotate_x(t)), [-1]),  # 180° x-rotation + flip width
            (rotate_x, [-3]),            # 90° x-rotation + flip depth
            (lambda t: rotate_y(rotate_y(rotate_y(t))), [-1]),  # 270° y-rotation + flip width
            (None, [-2, -3]),              # Flip along height and depth
            (None, [-1, -2]),              # Flip along width and height
            (rotate_z, [-1, -2]),        # 90° z-rotation + flip width and height
        )

    def __call__(self, batch:tuple[torch.Tensor,torch.Tensor]) -> tuple[torch.Tensor,torch.Tensor]:
        # Randomly choose one of the transformations
        xb, yb = batch
        ndim = xb.ndim  # Determine if 2D (4D) or 3D (5D)
        number_of_transforms = len(self.transforms) if ndim == 5 else 8
        sym_id = random.randint(0, number_of_transforms - 1)
        rotation, flip_axes = self.transforms[sym_id]

        # Apply transformations to both input and target batches
        xb = rotation(xb) if rotation else xb
        yb = rotation(yb) if rotation else yb

        xb = flip_along(xb, flip_axes)
        yb = flip_along(yb, flip_axes)

        return xb, yb

