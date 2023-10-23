from pathlib import Path

import torchapp as ta
from fastai.data.block import DataBlock
from fastai.data.core import DataLoaders
from fastai.data.transforms import RandomSplitter, get_image_files
from fastai.vision.augment import Resize
from fastai.vision.core import PILImageBW
from fastai.vision.data import ImageBlock

from supercat.noise.apps import NoiseSR


class ImageSR(NoiseSR):
    def dataloaders(
        self,
        base_dir: Path = ta.Param(default=None, help="The base directory for the dataset."),
        dim: int = ta.Param(default=2, help="The dimension of the dataset. 2 or 3."),
        depth: int = ta.Param(default=500, help="The depth of the noise image."),
        width: int = ta.Param(default=500, help="The width of the noise image."),
        height: int = ta.Param(default=500, help="The height of the noise image."),
        batch_size: int = ta.Param(default=16, help="The batch size for training."),
        valid_proportion: float = ta.Param(default=0.2, help="The proportion of the dataset to use for validation."),
        split_seed: int = 42,
    ):
        assert base_dir is not None, "You must specify a base directory for the dataset."
        base_dir = Path(base_dir)
        assert base_dir.exists(), f"Base directory {base_dir} does not exist."

        images = get_image_files(base_dir)

        self.dim = dim
        height = height or width
        depth = depth or width

        datablock = DataBlock(
            blocks=(ImageBlock(cls=PILImageBW)),
            splitter=RandomSplitter(valid_proportion, seed=split_seed),
            item_tfms=Resize((height, width)) if dim == 2 else Resize((depth, height, width)),
        )

        dataloaders = DataLoaders.from_dblock(
            datablock,
            source=images,
            bs=batch_size,
        )

        return dataloaders


if __name__ == "__main__":
    ImageSR.main()
