import random
import torch

# def rotate_z(tensor):  # 90° rotation around z-axis (h -> -w, w -> h)
#     return tensor.transpose(-1, -2).flip(-1)

# def rotate_x(tensor):  # 90° rotation around x-axis (3D: d -> -h, h -> d)
#     return tensor.transpose(-2, -3).flip(-2)

# def rotate_y(tensor):  # 90° rotation around y-axis (3D: w -> -d, d -> w)
#     return tensor.transpose(-1, -3).flip(-1)

# def flip_along(tensor, axes):  # General flip along specified axes
#     for axis in axes:
#         tensor = tensor.flip(axis)
#     return tensor

# FLIP_ROTATE_TRANSFORMS = (
#     (None, []),                    # No transformation
#     (rotate_z, []),              # 90° z-rotation
#     (lambda t: rotate_z(rotate_z(t)), []),  # 180° z-rotation
#     (lambda t: rotate_z(rotate_z(rotate_z(t))), []),  # 270° z-rotation
#     (None, [-1]),                  # Flip along width
#     (None, [-2]),                  # Flip along height
#     (rotate_z, [-1]),            # 90° z-rotation + flip width
#     (lambda t: rotate_z(rotate_z(t)), [-1]),  # 180° z-rotation + flip along width (updated)
#     (rotate_x, []),              # 90° x-rotation
#     (lambda t: rotate_x(rotate_x(t)), []),  # 180° x-rotation
#     (lambda t: rotate_x(rotate_x(rotate_x(t))), []),  # 270° x-rotation
#     (rotate_y, []),              # 90° y-rotation
#     (lambda t: rotate_y(rotate_y(t)), []),  # 180° y-rotation
#     (lambda t: rotate_y(rotate_y(rotate_y(t))), []),  # 270° y-rotation
#     (rotate_z, [-3]),            # 90° z-rotation + flip depth
#     (rotate_z, [-2]),            # 90° z-rotation + flip height
#     (None, [-3]),                  # Flip along depth
#     (rotate_y, [-3]),            # 90° y-rotation + flip depth
#     (lambda t: rotate_x(rotate_x(t)), [-1]),  # 180° x-rotation + flip width
#     (rotate_x, [-3]),            # 90° x-rotation + flip depth
#     (lambda t: rotate_y(rotate_y(rotate_y(t))), [-1]),  # 270° y-rotation + flip width
#     (None, [-2, -3]),              # Flip along height and depth
#     (None, [-1, -2]),              # Flip along width and height
#     (rotate_z, [-1, -2]),        # 90° z-rotation + flip width and height
# )

# 2D transformation matrices (without using torch.eye)
TRANSFORMATIONS_2D = [
    torch.tensor([[1, 0], [0, 1]]),       # No transformation
    torch.tensor([[0, 1], [-1, 0]]),      # 90° rotation
    torch.tensor([[-1, 0], [0, -1]]),     # 180° rotation
    torch.tensor([[0, -1], [1, 0]]),      # 270° rotation
    torch.tensor([[1, 0], [0, -1]]),      # Flip along width (x-axis)
    torch.tensor([[-1, 0], [0, 1]]),      # Flip along height (y-axis)
    torch.tensor([[0, 1], [1, 0]]),       # 90° rotation + flip width
    torch.tensor([[0, -1], [-1, 0]]),     # 180° rotation + flip width
]

# 3D transformation matrices (without using torch.eye)
TRANSFORMATIONS_3D = [
    torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),   # No transformation
    torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),  # 90° rotation around x-axis
    torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), # 180° rotation around x-axis
    torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),  # 270° rotation around x-axis
    torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),  # 90° rotation around y-axis
    torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]), # 180° rotation around y-axis
    torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),  # 270° rotation around y-axis
    torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),  # 90° rotation around z-axis
    torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), # 180° rotation around z-axis
    torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),  # 270° rotation around z-axis
    torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # Flip along x-axis
    torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # Flip along y-axis
    torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # Flip along z-axis
    torch.tensor([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]), # 90° z-rotation + flip width
    torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),   # 270° z-rotation + flip width
    torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, -1]]), # 270° z-rotation + flip depth
    torch.tensor([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),  # 90° x-rotation + flip height
    torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), # 180° x-rotation + flip height
    torch.tensor([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]),# 270° y-rotation + flip width
    torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),# 180° rotation + flip all
    torch.tensor([[0, 0, 1], [-1, 0, 0], [0, 1, 0]]),  # 90° y-rotation + flip depth
    torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),  # 270° y-rotation + flip height
    torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, -1]]), # 90° z-rotation + flip depth
]

def apply_2d_transformation(tensor, index):
    # Get the transformation matrix
    transformation_matrix = TRANSFORMATIONS_2D[index]
    
    # Get the spatial dimensions of the tensor
    sample, channel, height, width = tensor.shape
    
    # Create a grid of coordinates (i.e., pixel locations)
    y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    
    # Stack the coordinates into a (2, height * width) tensor
    coords = torch.stack([x_coords.flatten(), y_coords.flatten()])  # Shape (2, height * width)
    
    # Apply the transformation matrix to the coordinates
    transformed_coords = torch.matmul(transformation_matrix, coords.float()).long()
    
    # Wrap coordinates to stay within bounds of the image (use modulo to avoid negative indices)
    transformed_coords[0] = transformed_coords[0] % width
    transformed_coords[1] = transformed_coords[1] % height
    
    # Use the transformed coordinates to map the values
    transformed_tensor = tensor[:, :, transformed_coords[1], transformed_coords[0]].view(sample, channel, height, width)
    
    return transformed_tensor

# Function to apply a 3D transformation based on an index
def apply_3d_transformation(tensor, index):
    transformation_matrix = TRANSFORMATIONS_3D[index]
    
    # Get the shape of the tensor
    sample, channel, spatial_dim1, spatial_dim2, spatial_dim3 = tensor.shape
    
    # Flatten the spatial dimensions into a (spatial, 3) shape for matrix multiplication
    spatial_tensor = tensor.view(sample * channel, spatial_dim1, spatial_dim2, spatial_dim3)
    flattened_tensor = spatial_tensor.view(-1, 3)
    
    # Apply the transformation matrix to the spatial dimensions
    transformed = torch.matmul(flattened_tensor, transformation_matrix.T)
    
    # Reshape back to the original (sample, channel, spatial, spatial, spatial)
    transformed = transformed.view(sample, channel, spatial_dim1, spatial_dim2, spatial_dim3)
    
    return transformed



def apply_2d_transformation(tensor, sym_id):
    match(sym_id):
        case 0:
            return tensor
        case 1:
            return tensor.flip(-2).transpose(-1, -2)
        case 2:
            return tensor.flip(-2).flip(-1)
        case 3:
            return tensor.flip(-2).transpose(-1, -2).flip(-1)
        case 4:
            return tensor.flip(-1)
        case 5:
            return tensor.flip(-2)
        case 6:
            return tensor.flip(-1).transpose(-1, -2)
        case 7:
            return tensor.flip(-1).flip(-2).transpose(-1, -2)
        case _:
            raise ValueError(f"Invalid sym_id: {sym_id}")
    

def flip_and_rotate(batch:tuple[torch.Tensor,torch.Tensor,torch.Tensor], sym_id:int|None=None) -> tuple[torch.Tensor,torch.Tensor]:
    # Randomly choose one of the transformations
    input, target, residual = batch
    ndim = input.ndim  # Determine if 2D (4D) or 3D (5D)
        
    is_2d = (ndim == 4)

    number_of_transforms = len(TRANSFORMATIONS_2D) if is_2d else len(TRANSFORMATIONS_3D)
    if sym_id is None:
        sym_id = random.randint(0, number_of_transforms - 1)
    
    assert 0 <= sym_id < number_of_transforms

    # No transformation
    if sym_id == 0:
        return batch
    
    apply_transformation = apply_2d_transformation if is_2d else apply_3d_transformation
    input = apply_transformation(input, sym_id)
    target = apply_transformation(target, sym_id)
    residual = apply_transformation(residual, sym_id)

    return input, target, residual

