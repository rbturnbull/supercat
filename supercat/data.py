import random
from dataclasses import dataclass, field
from pathlib import Path
import hdf5storage
import os
from pathlib import Path
from skimage import io, color
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import lightning as L
import numpy as np

np.float = float
np.int = int

from .augmentation import flip_and_rotate
from .diffusion import DDPM


class Pipeline(list):
    def __call__(self, batch):
        """ Apply each step (function) in the pipeline to the batch. """
        for step in self:
            batch = step(batch)
        return batch


def stack_not_none(tensors:list[torch.Tensor], **kwargs):
    tensors = [tensor for tensor in tensors if tensor is not None]
    return torch.stack(tensors, **kwargs)


def stack_collate(batch):
    upscaled, high_res, residuals = zip(*batch)
    return stack_not_none(upscaled, dim=0), stack_not_none(high_res, dim=0), stack_not_none(residuals, dim=0)


def video_shape(item:Path) -> tuple:
    from skvideo.io import ffprobe
    metadata = ffprobe(str(item))
    frame_count = int(metadata['video']['@nb_frames'])
    frame_width = int(metadata['video']['@width'])
    frame_height = int(metadata['video']['@height'])

    return (frame_count, frame_height, frame_width)


@dataclass
class TrainingItem():
    high_res: Path
    upsampled: Path|None=None


def crop_or_pad(tensor, crop_size, axis=0, random_crop:bool=False):
    # Get the size of the tensor along the specified axis
    size_along_axis = tensor.shape[axis]

    if crop_size > size_along_axis:
        # Calculate the amount of padding needed
        pad_size = crop_size - size_along_axis
        pad_before = pad_size // 2
        pad_after = pad_size - pad_before

        # Pad along the specified axis
        pad_dims = [(0, 0)] * tensor.ndim  # No padding for other dimensions
        pad_dims[axis] = (pad_before, pad_after)

        tensor = F.pad(tensor, pad=[v for dim in reversed(pad_dims) for v in dim])
    else:
        # Calculate start and end indices for the crop
        if random_crop:
            # Calculate a random start position
            start = torch.randint(0, size_along_axis - crop_size + 1, (1,)).item()
        else:
            # Calculate the center start position
            start = (size_along_axis - crop_size) // 2
        end = start + crop_size

        # Use slicing along the specified axis
        slices = [slice(None)] * tensor.ndim  # Create slices for all dimensions
        slices[axis] = slice(start, end)  # Set the slice for the specified axis

        tensor = tensor[tuple(slices)]
    return tensor

@dataclass(kw_only=True)
class SupercatDataset(Dataset):
    width:int|None=None
    height:int|None=None
    depth:int|None=None
    random_crop:bool=False

    def get_tensor(self, path:Path):
        DEEPROCK_HDF5_KEY = "temp"
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".mat":
            try:
                data_dict = hdf5storage.loadmat(str(path))
            except Exception as err:
                raise IOError(f"Error reading 3D file '{path}':\n{err}")
            if DEEPROCK_HDF5_KEY not in data_dict:
                keys_found = ",".join(data_dict.keys())
                raise Exception(f"expected key {DEEPROCK_HDF5_KEY} not found in '{path}'.\nCheck the following keys: {keys_found}")

            result =  data_dict[DEEPROCK_HDF5_KEY]/255.0
        elif suffix == ".mp4":
            from skvideo.io import vreader

            depth = self.depth
            height = self.height
            width = self.width
            assert depth
            assert height
            assert width

            try:
                frame_count, frame_height, frame_width = video_shape(path)
                frame_start = random.randint(0, max(frame_count - depth,0))
                frame_end = min(frame_start + depth, frame_count)

                y_start = random.randint(0, max(frame_height - height,0))
                y_end = min(y_start + height, frame_height)
                x_start = random.randint(0, max(frame_width - width,0))
                x_end = min(x_start + width, frame_width)
                reader = vreader(str(path), num_frames=depth, as_grey=True)
                image = np.zeros( (frame_end-frame_start, y_end-y_start, x_end-x_start), dtype=np.uint8 )

                for i, frame in enumerate(reader):
                    if i < frame_start:
                        continue
                    image[i-frame_start,:,:] = frame[0,y_start:y_end, x_start:x_end,0]
                    
                result = image/255.0 * 2 - 1.0
                result = torch.tensor(result, dtype=torch.float32)
            except Exception as err:
                print(f"Failed to read {path}: {err}")
                return None
        else:
            result = io.imread(path)
            if len(result.shape) == 3:
                # check if this is RGBA
                if result.shape[2] == 4:
                    result = result[:,:,:3]
                
                if result.shape[2] == 3:
                    # Convert to grayscale
                    result = color.rgb2gray(result)
                else:
                    raise ValueError(f"Unable to convert {path} to single channel.")
            
            if result.dtype == "uint8":
                result = result/255.0

        result = torch.as_tensor(result, dtype=float)
        

        if self.width:
            result = crop_or_pad(result, self.width, axis=-1, random_crop=self.random_crop)
        if self.height:
            result = crop_or_pad(result, self.height, axis=-2, random_crop=self.random_crop)
        if self.depth and len(result.shape) == 3:
            result = crop_or_pad(result, self.depth, axis=-3, random_crop=self.random_crop)

        # Rescale from -1 to 1
        result = result * 2 - 1.0

        # Add channel
        result = result.unsqueeze(0)

        return result


@dataclass(kw_only=True)
class SupercatPredictionDataset(SupercatDataset):
    items: list[Path]
    scale_factor: float = 2.0

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        low_res = self.get_tensor(item)
        print(low_res.min(), low_res.max())
        mode = 'trilinear' if len(low_res.shape) == 4 else 'bilinear'
        upsampled = F.interpolate(low_res.unsqueeze(0), scale_factor=self.scale_factor, mode=mode, align_corners=True).squeeze(0)
        return upsampled


@dataclass(kw_only=True)
class SupercatTrainingDataset(SupercatDataset):
    items: list[TrainingItem]
    scale_factor: float = 2.0

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        high_res = self.get_tensor(item.high_res)
        if high_res is None:
            return None, None, None
        
        if item.upsampled and item.upsampled.exists():
            upsampled = self.get_tensor(item.upsampled)
        else:
            # If no upsampled is provided, we'll just downsample the high_res image
            mode = 'trilinear' if len(high_res.shape) == 4 else 'bilinear'
            low_res = F.interpolate(high_res.unsqueeze(0), scale_factor=1/self.scale_factor, mode=mode, align_corners=True)
            upsampled = F.interpolate(low_res, scale_factor=self.scale_factor, mode=mode, align_corners=True).squeeze(0)
            assert upsampled.shape == high_res.shape, f"{item.high_res} shape {high_res.shape} != upsampled {upsampled.shape}"

        residual = high_res - upsampled

        return upsampled, high_res, residual


@dataclass
class SupercatDataModule(L.LightningDataModule):
    training_items:list[TrainingItem]
    validation_items:list[TrainingItem]
    batch_size:int = 1
    num_workers:int|None = None
    scale_factor:float = 2.0
    width:int|None=None
    height:int|None=None
    depth:int|None=None
    augment:bool = True
    random_crop_training:bool = True
    diffusion:bool = False

    def __post_init__(self):
        super().__init__()
        if self.diffusion:
            self.ddpm = DDPM()

    def setup(self, stage=None):
        if self.num_workers is None:
            self.num_workers = min(os.cpu_count(), 8)

        kwargs = dict(scale_factor=self.scale_factor, width=self.width, height=self.height, depth=self.depth)
        self.train_dataset = SupercatTrainingDataset(items=self.training_items, random_crop=self.random_crop_training, **kwargs)
        self.val_dataset = SupercatTrainingDataset(items=self.validation_items, random_crop=False, **kwargs)

    def train_dataloader(self, num_workers:int|None=None):
        num_workers = num_workers or self.num_workers

        collate_pipeline = Pipeline([stack_collate])
        if self.augment:
            collate_pipeline.append(flip_and_rotate)
        
        if self.diffusion:
            collate_pipeline.append(self.ddpm.modify_batch_training)

        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_pipeline)

    def val_dataloader(self):
        collate_pipeline = Pipeline([stack_collate])
        if self.diffusion:
            collate_pipeline.append(self.ddpm.modify_batch_not_training)

        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=collate_pipeline)
