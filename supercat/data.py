import re
from dataclasses import dataclass, field
from pathlib import Path
import hdf5storage
import os
from pathlib import Path
from skimage import io
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import lightning as L


# def is_validation_image(item:tuple):
#     "Returns True if this image should be part of the validation set i.e. if the parent directory doesn't have the string `_train_` in it."
#     return "_train_" not in item.parent.name


# def get_y(item, pattern=r"_BI_.*"):
#     dir_name = re.sub(pattern, "_HR", item.parent.name)            
#     return item.parent.parent/dir_name/item.name


# def get_items(self, directory):
#     if self.dim == 2:
#         return get_image_files(directory)
    
#     directory = Path(directory)
#     return list(directory.glob("*.mat"))            


@dataclass
class TrainingItem():
    high_res: Path
    upsampled: Path|None=None


@dataclass(kw_only=True)
class SupercatTrainingDataset(Dataset):
    items: list[TrainingItem]
    scale_factor: float = 2.0

    def __len__(self):
        return len(self.items)
    
    def get_tensor(self, path:Path):
        DEEPROCK_HDF5_KEY = "temp"
        path = Path(path)
        if path.suffix == ".mat":
            try:
                data_dict = hdf5storage.loadmat(str(path))
            except Exception as err:
                raise IOError(f"Error reading 3D file '{path}':\n{err}")
            if DEEPROCK_HDF5_KEY not in data_dict:
                keys_found = ",".join(data_dict.keys())
                raise Exception(f"expected key {DEEPROCK_HDF5_KEY} not found in '{path}'.\nCheck the following keys: {keys_found}")

            result =  data_dict[DEEPROCK_HDF5_KEY]/255.0
        else:
            result = io.imread(path)

        result = torch.float(result)

        return result

    def __getitem__(self, idx):
        item = self.items[idx]
        high_res = self.get_tensor(item.high_res)
        if item.upsampled and item.upsampled.exists():
            upsampled = self.get_tensor(item.upsampled)
        else:
            # If no upsampled is provided, we'll just downsample the high_res image
            breakpoint()
            low_res = F.interpolate(high_res, scale_factor=1/self.scale_factor, mode='bilinear', align_corners=True)
            upsampled = F.interpolate(low_res, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)

        residual = high_res - upsampled

        return upsampled, high_res, residual


@dataclass
class SupercatDataModule(L.LightningDataModule):
    training_items:list[TrainingItem]
    validation_items:list[TrainingItem]
    batch_size:int = 1
    num_workers:int|None = None

    def __post_init__(self):
        super().__init__()

    def setup(self, stage=None):
        if self.num_workers is None:
            self.num_workers = min(os.cpu_count(), 8)

        self.train_dataset = SupercatTrainingDataset(items=self.training_items)
        self.val_dataset = SupercatTrainingDataset(items=self.validation_items)

    def train_dataloader(self, num_workers:int|None=None):
        num_workers = num_workers or self.num_workers
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
