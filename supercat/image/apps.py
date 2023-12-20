from pathlib import Path
import torchapp as ta
from fastai.data.block import DataBlock, TransformBlock
from fastai.data.core import DataLoaders
from fastai.data.transforms import FuncSplitter, get_files, image_extensions

from supercat.noise.apps import NoiseSR
from supercat.image.transforms import ImageVideoReader, check_items
from rich.progress import track
import torchvision
import torch.nn.functional as F


def path_splitter_func(item:Path):
    item = Path(item)
    return "train" not in f"{item.parent.parent.name}/{item.parent.name}"


def get_image_video_files(directory: Path, recurse=True, folders=None):
    "Get video files in `path` recursively, only in `folders`, if specified."
    extensions = set(image_extensions)
    extensions.add(".mp4")
    return get_files(directory, extensions=extensions, recurse=recurse, folders=folders)


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
        check_readable:bool = True,
    ):  
        assert base_dir is not None, "You must specify a base directory for the dataset."
        base_dir = Path(base_dir)
        assert base_dir.exists(), f"Base directory {base_dir} does not exist."

        self.dim = dim
        height = height or width
        depth = depth or width

        shape = (height, width) if dim == 2 else (depth, height, width)
        reader = ImageVideoReader(shape=shape)

        datablock = DataBlock(
            blocks=(TransformBlock),
            splitter=FuncSplitter(path_splitter_func),
            item_tfms=[reader],
        )


        # items = get_image_video_files(base_dir/"train")[:2] + get_image_video_files(base_dir/"val")[:1]

        # Get items and loop through to find ones that are readable
        items = get_image_video_files(base_dir)
        torchvision.set_video_backend("pyav")

        if check_readable:
            items = check_items(items, shape)
            # readable_items = []
            # for item in track(items, description="Checking files are readable:"):
            #     try:
            #         if not item.suffix in image_extensions:
            #             min_size = min(video_shape(item))
            #             if min_size < max(shape):
            #                 continue
            #         readable_items.append(item)
            #     except Exception as err:
            #         print(f"Cannot read {item}: {err}")      
            # items = readable_items       

        print(f"Using {len(items)} items")
        
        dataloaders = DataLoaders.from_dblock(
            datablock,
            source=items,
            bs=batch_size,
        )
        
        return dataloaders
    
    def loss_func(self):
        """
        Returns the loss function to use with the model.
        """
        return F.smooth_l1_loss


if __name__ == "__main__":
    ImageSR.main()
