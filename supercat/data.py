from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import rescale
import hdf5storage
from PIL import Image


TRANSFORMATIONS_3D = [
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
TRANSFORMATIONS_2D = TRANSFORMATIONS_3D[:8]

# def flip_and_rotate(
#     batch:tuple[torch.Tensor,torch.Tensor,torch.Tensor], 
#     sym_id:int|None=None
# ) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
#     """
#     Applies a random or specified flip and/or rotation transformation to a batch of 2D or 3D tensors. 
#     This function transforms the input, target, and residual tensors by flipping, rotating, and 
#     transposing them in a consistent manner.

#     Args:
#         batch (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing the input, target, 
#             and residual tensors to be transformed.
#         sym_id (int or None, optional): The transformation identifier. If None, a random transformation 
#             is chosen. For 2D tensors, sym_id ranges from 0 to 7; for 3D tensors, it ranges from 0 to 47.

#     Returns:
#         tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the transformed input, 
#             target, and residual tensors after applying the flip and/or rotation.

#     Raises:
#         AssertionError: If the sym_id is out of the valid range for the tensor dimensions.
#     """
#     input, target, residual = batch

#     # Determine if 2D (4D) or 3D (5D)
#     is_2d = (input.ndim == 4)
#     number_of_transforms = 8 if is_2d else len(TRANSFORMATIONS)
    
#     if sym_id is None:
#         # Randomly choose one of the transformations
#         sym_id = random.randint(0, number_of_transforms - 1)
    
#     # Shortcut for no transformation
#     if sym_id == 0:
#         return batch

#     assert 0 <= sym_id < number_of_transforms    
#     transformation = TRANSFORMATIONS[sym_id]
    
#     input = transformation(input)
#     target = transformation(target)
#     residual = transformation(residual)

#     return input, target, residual


def downscale_tricubic_rescale(vol: np.ndarray, factor: int = 4) -> np.ndarray:
    """
    Downscale a 3D volume by `factor` using tricubic interpolation + anti-aliasing.
    """
    low = rescale(
        vol,
        scale=1.0/factor,
        order=3,                # cubic
        mode='reflect',
        anti_aliasing=True,
        preserve_range=True,
        channel_axis=None       # ensures (D,H,W) or (C,D,H,W) handled correctly if needed
    )
    return low.astype(vol.dtype, copy=False)


def upscale_tricubic_rescale(vol: np.ndarray, factor: int = 4) -> np.ndarray:
    """
    Upscale a 3D volume by `factor` using tricubic interpolation.
    """
    up = rescale(
        vol,
        scale=factor,
        order=3,                # cubic
        mode='reflect',
        anti_aliasing=False,    # no AA on upscaling
        preserve_range=True,
        channel_axis=None
    )
    return up.astype(vol.dtype, copy=False)


def read_mat(path:Path):
    DEEPROCK_HDF5_KEY = "temp"
    try:
        data_dict = hdf5storage.loadmat(str(path))
    except Exception as err:
        raise IOError(f"Error reading 3D file '{path}':\n{err}")
    if DEEPROCK_HDF5_KEY not in data_dict:
        keys_found = ",".join(data_dict.keys())
        raise Exception(f"expected key {DEEPROCK_HDF5_KEY} not found in '{path}'.\nCheck the following keys: {keys_found}")

    return data_dict[DEEPROCK_HDF5_KEY]


class Deeprock3D(Dataset):
    def __init__(self, deeprock: Path, scale: int = 4, channel_first: bool = True, partition:str="train", augment:bool=False):
        self.deeprock = Path(deeprock)
        self.scale = int(scale)
        self.channel_first = channel_first
        categories = ["sandstone", "carbonate", "coal", "sand"]
        self.augment = augment

        hr_items: list[Path] = []
        for cat in categories:
            hr_dir = self.deeprock / f"{cat}3D" / f"{cat}3D_{partition}_HR"
            hr_items.extend(sorted(hr_dir.glob("*.mat")))
        self.hr_items = hr_items

        if len(self.hr_items) == 0:
            raise FileNotFoundError(f"No .mat files found under {self.deeprock}/*3D/*3D_{partition}_HR")

    def _hr_to_lr_path(self, hr_path: Path) -> Path:
        """
        Map .../<cat>3D/<cat>3D_train_HR/<file>.mat  -->
            .../<cat>3D/<cat>3D_train_TRI_unknown_X{scale}/<file>.mat
        """
        rel = hr_path.relative_to(self.deeprock)
        parts = list(rel.parts)
        # parts[-2] expected: "<cat>3D_train_HR"
        parent_name = parts[-2]
        if not parent_name.endswith("_HR"):
            raise ValueError(f"Unexpected parent '{parent_name}' for {hr_path}")
        parts[-2] = parent_name.replace("_HR", f"_LR_default_X{self.scale}").replace(".mat",f"x{self.scale}.mat")
        parts[-1] = parts[-1].replace(".mat",f"x{self.scale}.mat")
        return self.deeprock.joinpath(*parts)

    def __len__(self) -> int:
        return len(self.hr_items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hr_path = self.hr_items[idx]

        lr_path = self._hr_to_lr_path(hr_path)

        # if not lr_path.exists():
        #     raise FileNotFoundError(f"LR file missing for {hr_path.name}: {lr_path}")

        def transform_scale(data):
            return 2.0*data/255.0 - 1

        hr = transform_scale(read_mat(hr_path))
        lr = transform_scale(read_mat(lr_path))
        lr = upscale_tricubic_rescale(lr)

        # lr = transform_scale(read_mat(lr_path))

        # Optional: enforce channel-first tensors
        # If (D,H,W), add channel dim; if already (C,D,H,W), leave it.
        if hr.ndim == 3:
            hr = hr[None, ...]  # (1,D,H,W)
        if lr.ndim == 3:
            lr = lr[None, ...]
        if not self.channel_first and hr.ndim == 4:
            # Convert to (D,H,W,C)
            hr = np.moveaxis(hr, 0, -1)
            lr = np.moveaxis(lr, 0, -1)

        hr_t = torch.from_numpy(hr.copy())  # ensure contiguous
        lr_t = torch.from_numpy(lr.copy())

        if self.augment:
            transformation = TRANSFORMATIONS_3D[np.random.randint(0, len(TRANSFORMATIONS_3D))]
            hr_t = transformation(hr_t)
            lr_t = transformation(lr_t)

        assert lr_t.shape == hr_t.shape
        assert lr_t.max() < 1.01, f"Low resolution {lr_path} gives range {lr_t.min()}-{lr_t.max()}"
        assert lr_t.min() > -1.01, f"Low resolution {lr_path} gives range {lr_t.min()}-{lr_t.max()}"
        assert hr_t.max() < 1.01, f"High resolution {hr_path} gives range {hr_t.min()}-{hr_t.max()}"
        assert hr_t.min() > -1.01, f"High resolution {hr_path} gives range {hr_t.min()}-{hr_t.max()}"

        return hr_t, lr_t


class Deeprock2D(Dataset):
    def __init__(self, deeprock: Path, scale: int = 4, channel_first: bool = True, partition:str="train", augment:bool=False):
        self.deeprock = Path(deeprock)
        self.scale = int(scale)
        self.channel_first = channel_first
        categories = ["sandstone", "carbonate", "coal", "sand"]
        self.augment = augment

        hr_items: list[Path] = []
        for cat in categories:
            hr_dir = self.deeprock / f"{cat}2D" / f"{cat}2D_{partition}_HR"
            hr_items.extend(sorted(hr_dir.glob("*.png")))
        self.hr_items = hr_items

        if len(self.hr_items) == 0:
            raise FileNotFoundError(f"No .png files found under {self.deeprock}/*2D/*2D_{partition}_HR")

    def _hr_to_lr_path(self, hr_path: Path) -> Path:
        """
        Map .../<cat>2D/<cat>2D_train_HR/<file>.png  -->
            .../<cat>2D/<cat>2D_train_BI_unknown_X{scale}/<file>.png
        """
        rel = hr_path.relative_to(self.deeprock)
        parts = list(rel.parts)
        # parts[-2] expected: "<cat>2D_train_HR"
        parent_name = parts[-2]
        if not parent_name.endswith("_HR"):
            raise ValueError(f"Unexpected parent '{parent_name}' for {hr_path}")
        parts[-2] = parent_name.replace("_HR", f"_BI_unknown_X{self.scale}")
        return self.deeprock.joinpath(*parts)

    def __len__(self) -> int:
        return len(self.hr_items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hr_path = self.hr_items[idx]
        lr_path = self._hr_to_lr_path(hr_path)

        if not lr_path.exists():
            raise FileNotFoundError(f"LR file missing for {hr_path.name}: {lr_path}")

        def read_image(path):
            array = np.array(Image.open(path).convert("L"))
            return np.expand_dims(array, axis=0)

        def transform_scale(data):
            return 2.0*data/255.0 - 1

        hr = transform_scale(read_image(hr_path))
        lr = transform_scale(read_image(lr_path))

        # print(hr_path, lr_path)
        # print(hr.min(), hr.max(), lr.min(), lr.max())

        assert hr.shape[0] == 1

        hr_t = torch.from_numpy(hr.copy())  # ensure contiguous
        lr_t = torch.from_numpy(lr.copy())

        if self.augment:
            transformation = TRANSFORMATIONS_2D[np.random.randint(0, len(TRANSFORMATIONS_2D))]
            hr_t = transformation(hr_t)
            lr_t = transformation(lr_t)

        return hr_t, lr_t
    

def build_datasets3D(deeprock: Path, scale: int = 4, train_augment: bool = True):
    training_dataset = Deeprock3D(deeprock=deeprock, scale=scale, partition="train", augment=train_augment)
    validation_dataset = Deeprock3D(deeprock=deeprock, scale=scale, partition="valid")
    return training_dataset, validation_dataset


def build_datasets2D(deeprock: Path, scale: int = 4, train_augment: bool = True):
    training_dataset = Deeprock2D(deeprock=deeprock, scale=scale, partition="train", augment=train_augment)
    validation_dataset = Deeprock2D(deeprock=deeprock, scale=scale, partition="valid")
    return training_dataset, validation_dataset


