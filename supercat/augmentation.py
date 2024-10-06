import random
import torch



# def generate_24_rotations():
#     rotations = []

#     # Helper function to generate a 90-degree rotation
#     def rotate_90(tensor, axis1, axis2):
#         return f"{tensor}.transpose({axis1}, {axis2})"

#     # Rotation groups: (face of the cube that is on top, then 4 possible rotations)
#     # 1. Keep original orientation (no rotation), rotate the original face
#     rotations.append(lambda t: t)  # Identity

#     # 2. Rotate around the z-axis (width-height plane)
#     rotations.append(lambda t: rotate_90(t, -1, -2))  # 90° around z-axis
#     rotations.append(lambda t: rotate_90(rotate_90(t, -1, -2), -1, -2))  # 180° around z-axis
#     rotations.append(lambda t: rotate_90(rotate_90(rotate_90(t, -1, -2), -1, -2), -1, -2))  # 270° around z-axis

#     # 3. Rotate around the y-axis (width-depth plane)
#     rotations.append(lambda t: rotate_90(t, -1, -3))  # 90° around y-axis
#     rotations.append(lambda t: rotate_90(rotate_90(t, -1, -3), -1, -3))  # 180° around y-axis
#     rotations.append(lambda t: rotate_90(rotate_90(rotate_90(t, -1, -3), -1, -3), -1, -3))  # 270° around y-axis

#     # 4. Rotate around the x-axis (height-depth plane)
#     rotations.append(lambda t: rotate_90(t, -2, -3))  # 90° around x-axis
#     rotations.append(lambda t: rotate_90(rotate_90(t, -2, -3), -2, -3))  # 180° around x-axis
#     rotations.append(lambda t: rotate_90(rotate_90(rotate_90(t, -2, -3), -2, -3), -2, -3))  # 270° around x-axis

#     # 5. Align another face on top (transposing axes) and rotate
#     rotations.append(lambda t: f"{t}.transpose(-2, -3)")  # Rotate depth to top
#     rotations.append(lambda t: rotate_90(f"{t}.transpose(-2, -3)", -1, -2))  # Rotate depth to top and 90° around new top
#     rotations.append(lambda t: rotate_90(rotate_90(f"{t}.transpose(-2, -3)", -1, -2), -1, -2))  # Rotate depth to top and 180°
#     rotations.append(lambda t: rotate_90(rotate_90(rotate_90(f"{t}.transpose(-2, -3)", -1, -2), -1, -2), -1, -2))  # Rotate depth to top and 270°

#     # 6. Rotate another face to top (swapping axes) and rotate
#     rotations.append(lambda t: f"{t}.transpose(-1, -3)")  # Rotate width to top
#     rotations.append(lambda t: rotate_90(f"{t}.transpose(-1, -3)", -2, -3))  # Rotate width to top and 90° around new top
#     rotations.append(lambda t: rotate_90(rotate_90(f"{t}.transpose(-1, -3)", -2, -3), -2, -3))  # Rotate width to top and 180°
#     rotations.append(lambda t: rotate_90(rotate_90(rotate_90(f"{t}.transpose(-1, -3)", -2, -3), -2, -3), -2, -3))  # Rotate width to top and 270°

#     # 7. Another face to top and rotate (different axes)
#     rotations.append(lambda t: f"{t}.transpose(-1, -2)")  # Rotate height to top
#     rotations.append(lambda t: rotate_90(f"{t}.transpose(-1, -2)", -2, -3))  # Rotate height to top and 90° around new top
#     rotations.append(lambda t: rotate_90(rotate_90(f"{t}.transpose(-1, -2)", -2, -3), -2, -3))  # Rotate height to top and 180°
#     rotations.append(lambda t: rotate_90(rotate_90(rotate_90(f"{t}.transpose(-1, -2)", -2, -3), -2, -3), -2, -3))  # Rotate height to top and 270°

#     return rotations


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

# def generate_48_transformations():
#     transformations = []

#     # Axes corresponding to width (-1), height (-2), and depth (-3)
#     axes = [-1, -2, -3]

#     # Helper function to generate rotations and flips
#     def add_rotation(tensor, transpose_axes):
#         return tensor.transpose(transpose_axes[0], transpose_axes[1])

#     def add_flip(tensor, flip_axes):
#         for axis in flip_axes:
#             tensor = tensor.flip(axis)
#         return tensor

#     # Step 1: Generate 24 unique rotations
#     # rotations = [
#     #     lambda tensor: tensor,  # Identity (0-degree rotation)
#     #     lambda tensor: add_rotation(tensor, (-1, -2)),  # 90-degree rotation around z-axis
#     #     lambda tensor: add_rotation(add_rotation(tensor, (-1, -2)), (-1, -2)),  # 180-degree rotation
#     #     lambda tensor: add_rotation(add_rotation(add_rotation(tensor, (-1, -2)), (-1, -2)), (-1, -2)),  # 270-degree rotation
#     #     lambda tensor: add_rotation(tensor, (-1, -3)),  # 90-degree rotation around y-axis
#     #     lambda tensor: add_rotation(add_rotation(tensor, (-1, -3)), (-1, -3)),  # 180-degree rotation around y-axis
#     #     lambda tensor: add_rotation(add_rotation(add_rotation(tensor, (-1, -3)), (-1, -3)), (-1, -3)),  # 270-degree rotation
#     #     lambda tensor: add_rotation(tensor, (-2, -3)),  # 90-degree rotation around x-axis
#     #     lambda tensor: add_rotation(add_rotation(tensor, (-2, -3)), (-2, -3)),  # 180-degree rotation around x-axis
#     #     lambda tensor: add_rotation(add_rotation(add_rotation(tensor, (-2, -3)), (-2, -3)), (-2, -3)),  # 270-degree rotation
#     #     # Rotations where axes are swapped
#     #     lambda tensor: tensor.transpose(-2, -3),  # Swap depth/height
#     #     lambda tensor: tensor.transpose(-1, -3),  # Swap depth/width
#     #     lambda tensor: tensor.transpose(-1, -2),  # Swap height/width
#     #     lambda tensor: tensor.transpose(-3, -2).transpose(-2, -1),  # Rotate axes cyclically
#     #     lambda tensor: tensor.transpose(-2, -1).transpose(-1, -3),  # Another cyclic permutation
#     #     # Continue generating cyclic rotations
#     # ]
#     rotations = generate_24_rotations()

#     # Step 2: Generate 24 unique reflections
#     reflections = [
#         lambda tensor: tensor,  # No Flip
#         lambda tensor: f"{tensor}.flip(-1)",  # Flip along width axis
#         lambda tensor: f"{tensor}.flip(-2)",  # Flip along height axis
#         lambda tensor: f"{tensor}.flip(-3)",  # Flip along depth axis
#         lambda tensor: f"{tensor}.flip(-1).flip(-2)",  # Flip along width and height
#         lambda tensor: f"{tensor}.flip(-1).flip(-3)",  # Flip along width and depth
#         lambda tensor: f"{tensor}.flip(-2).flip(-3)",  # Flip along height and depth
#         lambda tensor: f"{tensor}.flip(-1).flip(-2).flip(-3)",  # Flip along all three axes
#     ]

#     # Step 3: Combine rotations and reflections to get 48 unique transformations
#     for rotate in rotations:
#         for reflect in reflections:
#             transformations.append(lambda tensor, r=rotate, f=reflect: f(r(tensor)))

#     return transformations

# Apply the generated transformations
# def apply_transformation(tensor, sym_id):

#     # r = generate_24_rotations()
#     transformations = generate_48_transformations()

#     case = 8
#     duplicates = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175]
#     for i, transformation in enumerate(transformations):
#         if i in duplicates:
#             continue
#         print(f"case {case}:")
#         t_string = transformation("tensor")
#         print(f"    return {t_string}")
#         case += 1

#     breakpoint()

#     assert False
    
#     if sym_id < len(transformations):
#         return transformations[sym_id](tensor)
#     else:
#         raise ValueError(f"Invalid sym_id: {sym_id}")


def flip_and_rotate(batch:tuple[torch.Tensor,torch.Tensor,torch.Tensor], sym_id:int|None=None) -> tuple[torch.Tensor,torch.Tensor]:
    # Randomly choose one of the transformations
    input, target, residual = batch
    ndim = input.ndim  # Determine if 2D (4D) or 3D (5D)
        
    is_2d = (ndim == 4)

    number_of_transforms = 8 if is_2d else 48
    if sym_id is None:
        sym_id = random.randint(0, number_of_transforms - 1)
    
    # assert 0 <= sym_id < number_of_transforms

    # No transformation
    if sym_id == 0:
        return batch
    
    input = apply_transformation(input, sym_id)
    target = apply_transformation(target, sym_id)
    residual = apply_transformation(residual, sym_id)

    return input, target, residual

