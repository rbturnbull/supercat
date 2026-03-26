from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from widitapp import WiDiTApp
from cluey import method

from .data import read_image_as_tensor, TRANSFORMATIONS_2D, TRANSFORMATIONS_3D


def _pad_to_size_with_reflect(image: torch.Tensor, target_size: int) -> torch.Tensor:
    """Pad trailing spatial dimensions to ``target_size`` using safe reflect steps."""
    for dim in (-1, -2):
        while image.shape[dim] < target_size:
            remaining = target_size - image.shape[dim]
            current = image.shape[dim]

            if current <= 1:
                # Reflection needs at least 2 pixels along the padded dimension.
                mode = "replicate"
                pad_amount = remaining
            else:
                mode = "reflect"
                pad_amount = min(remaining, current - 1)

            if dim == -1:
                image = torch.nn.functional.pad(image, (0, pad_amount), mode=mode)
            else:
                image = torch.nn.functional.pad(image, (0, 0, 0, pad_amount), mode=mode)

    return image

def find_files(base_path:Path, valid_extensions) -> list[Path]:
    """ Finds all images and movies in the given directory """
    if not base_path.is_dir():
        raise ValueError(f"{base_path} is not a valid directory")

    files = []
    for ext in valid_extensions:
        files.extend(base_path.rglob(f'*{ext.lower()}'))
        files.extend(base_path.rglob(f'*{ext.upper()}'))

    return files


def find_images(base_path:Path) -> list[Path]:
    return find_files(base_path, ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])


def find_movies(base_path:Path) -> list[Path]:
    return find_files(base_path, ['.mp4', '.avi', '.mov', '.mkv'])


class PretrainImagesDataset(Dataset):
    def __init__(self, path: Path, scale: int = 4, channel_first: bool = True, augment:bool=False, size:int=500):
        assert path is not None, "Path must be provided"
        self.path = Path(path)
        self.scale = int(scale)
        self.channel_first = channel_first
        self.augment = augment
        self.size = size

        self.items = find_images(path)

    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.items[idx]
        hr_t = read_image_as_tensor(path)

        # random crop if image is larger than size
        if hr_t.shape[-1] > self.size:
            start = np.random.randint(0, hr_t.shape[-1] - self.size)
            hr_t = hr_t[..., start:start+self.size]
        if hr_t.shape[-2] > self.size:
            start = np.random.randint(0, hr_t.shape[-2] - self.size)
            hr_t = hr_t[..., start:start+self.size,:]
        
        # Reflect-pad undersized images in safe steps; large single-shot
        # reflect pads are invalid when the requested padding exceeds size-1.
        hr_t = _pad_to_size_with_reflect(hr_t, self.size)

        # hack to ensure dimensions are divisible by 2 for downsampling
        if hr_t.shape[-1] % 2 != 0:
            hr_t = hr_t[..., :-1]
        if hr_t.shape[-2] % 2 != 0:
            hr_t = hr_t[..., :-1, :]
        
        # Downsample the image to create the low-resolution version
        lr_t = torch.nn.functional.interpolate(hr_t.unsqueeze(0), scale_factor=1/self.scale, mode='bicubic', align_corners=False).squeeze(0)

        # upscale back to original size for training
        lr_t = torch.nn.functional.interpolate(lr_t.unsqueeze(0), size=hr_t.shape[1:], mode='bicubic', align_corners=False).squeeze(0)

        if self.augment:
            transformation = TRANSFORMATIONS_2D[np.random.randint(0, len(TRANSFORMATIONS_2D))]
            hr_t = transformation(hr_t)
            lr_t = transformation(lr_t)

        return lr_t, hr_t


class SupercatPretrainImage(WiDiTApp):
    @method
    def datasets(
        self,
        training:Path=None,
        validation:Path=None,
        scale:int=4,
        augment:bool=True,
        size:int=224,
        **kwargs,
    ) -> tuple[Dataset, Dataset]:
        """ Returns training and validation datasets """
        assert training is not None, "Training path must be provided"
        assert validation is not None, "Validation path must be provided"
        training_dataset = PretrainImagesDataset(path=training, scale=scale, size=size, augment=augment)
        validation_dataset = PretrainImagesDataset(path=validation, scale=scale, size=size, augment=False)
        return training_dataset, validation_dataset

