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


def stack_collate(batch):
    upscaled, high_res, residuals = zip(*batch)
    # # print(upscaled.shape)
    # print(len(upscaled), upscaled[0].shape)
    # print('collation', torch.stack(upscaled, dim=0).shape)
    return torch.stack(upscaled, dim=0), torch.stack(high_res, dim=0), torch.stack(residuals, dim=0)


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

            result = result/255.0 * 2 - 1.0

        result = torch.as_tensor(result, dtype=float)

        # Add channel
        result = result.unsqueeze(0)

        return result

    def __getitem__(self, idx):
        item = self.items[idx]
        high_res = self.get_tensor(item.high_res)
        if item.upsampled and item.upsampled.exists():
            upsampled = self.get_tensor(item.upsampled)
        else:
            # If no upsampled is provided, we'll just downsample the high_res image
            mode = 'trilinear' if len(high_res.shape) == 4 else 'bilinear'
            low_res = F.interpolate(high_res.unsqueeze(0), scale_factor=1/self.scale_factor, mode=mode, align_corners=True)
            upsampled = F.interpolate(low_res, scale_factor=self.scale_factor, mode=mode, align_corners=True).squeeze(0)

        residual = high_res - upsampled

        # print('upsampled.shape', upsampled.shape)
        # print('high_res.shape', high_res.shape)
        # print('residual.shape', residual.shape)

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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=num_workers, shuffle=True, collate_fn=stack_collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=stack_collate)
