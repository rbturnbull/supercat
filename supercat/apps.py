import re
import torch
from enum import Enum
from pathlib import Path
from typing import List
from PIL import Image
from torch import nn
import torch.nn.functional as F
from fastcore.transform import Pipeline
from fastai.data.transforms import get_image_files, FuncSplitter, ToTensor
from fastai.data.core import DataLoaders
from fastai.data.block import DataBlock
from fastai.vision.data import ImageBlock
from fastai.vision.core import PILImageBW
from fastai.vision.augment import RandomCrop, Resize
from fastai.learner import Learner, load_learner
import torchapp as ta
from torchapp.util import call_func, add_kwargs, change_typer_to_defaults

from .metrics import psnr, mse
from .transforms import ImageBlock3D, read3D, write3D, InterpolateTransform
from .interpolation import interpolate3D, InterpolationMethod
from .models import ResidualUNet, VideoUnet3d

from rich.console import Console
console = Console()

class DownsampleScale(Enum):
    X2 = "X2"
    X4 = "X4"


class ClipUnitInterval(nn.Module):
    def forward(self, input):
        return F.hardtanh(input, 0.0, 1.0)


class DownsampleMethod(Enum):
    DEFAULT = "default"
    UNKNOWN = "unknown"


def get_y(item):
    dir_name = re.sub(r"_BI_.*", "_HR", item.parent.name)            
    return item.parent.parent/dir_name/item.name

def get_y_3d(item):
    dir_name = re.sub(r"_TRI_.*", "_HR", item.parent.name)            
    return item.parent.parent/dir_name/item.name

def is_validation_image(item:tuple):
    "Returns True if this image should be part of the validation set i.e. if the parent directory doesn't have the string `_train_` in it."
    return "_train_" not in item.parent.name


class Supercat(ta.TorchApp):
    """
    A deep learning model for CT scan superresolution.
    """
    def __init__(self):
        super().__init__()

        self.validate_individual = self.copy_method(self.validate_individual)
        add_kwargs(to_func=self.validate_individual, from_funcs=[self.pretrained_local_path, self.inference_dataloader])

        # Make copies of methods to use just for the CLI
        self.validate_individual_cli = self.copy_method(self.validate_individual)

        # Remove params from defaults in methods not used for the cli
        change_typer_to_defaults(self.validate_individual)
    
    def get_items(self, directory):
        return get_image_files(directory)

    def dataloaders(
        self,
        deeprock:Path = ta.Param(help="The path to the DeepRockSR-2D dataset."), 
        downsample_scale:DownsampleScale = ta.Param(DownsampleScale.X4.value, help="Should it use the 2x or 4x downsampled images.", case_sensitive=False),
        downsample_method:DownsampleMethod = ta.Param(DownsampleMethod.UNKNOWN.value, help="Should it use the default method to downsample (bicubic) or a random kernel (UNKNOWN)."),
        batch_size:int = ta.Param(default=10, help="The batch size."),
        force:bool = ta.Param(default=False, help="Whether or not to force the conversion of the bicubic upscaling."),
        max_samples:int = ta.Param(default=None, help="If set, then the number of input samples for training/validation is truncated at this number."),
        random_crop:int = ta.Param(default=None, help="If set, then randomly crop the images to this resolution during training."),
        include_sand:bool = ta.Param(default=False, help="Including DeepSand-SR dataset."),
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which Supercat uses in training and prediction.
        """
        assert deeprock is not None

        deeprock = Path(deeprock)
        bicubic = []
        highres = []
        
        # sources = ["shuffled2D"]
        sources = ["carbonate2D","coal2D","sandstone2D"]
        if include_sand:
            sources.append("sand2D")

        if isinstance(downsample_method, DownsampleMethod):
            downsample_method = downsample_method.value
        if isinstance(downsample_scale, DownsampleScale):
            downsample_scale = downsample_scale.value

        split_types = ["train","valid"] # There is also "test"
        # split_types = ["train","valid","test"] # hack
        
        for source in sources:
            for split_type in split_types:
                highres_dir = deeprock/source/f"{source}_{split_type}_HR"
                highres_split = self.get_items(highres_dir)
                highres.extend( highres_split )

                lowres_dir = deeprock/source/f"{source}_{split_type}_LR_{downsample_method}_{downsample_scale}"
                
                # We will save bicubic upscaled images
                bicubic_dir = deeprock/source/f"{source}_{split_type}_BI_{downsample_method}_{downsample_scale}" 
                bicubic_dir.mkdir(exist_ok=True)

                for index, highres_path in enumerate(highres_split):
                    bicubic_path = bicubic_dir/highres_path.name

                    if not bicubic_path.exists() or force:
                        components = highres_path.name.split(".")
                        lowres_name = f'{components[0]}{downsample_scale.lower()}.{components[1]}'
                        lowres_path = lowres_dir/lowres_name
                        print(split_type, highres_path, bicubic_path, lowres_path)
                        
                        # upscale with bicubic interpolation
                        print("Upscaling with bicubic")
                        highres_img = Image.open(highres_path)
                        lowres_img = Image.open(lowres_path)

                        # Convert to single channel
                        if lowres_img.mode == "RGB":
                            lowres_img = lowres_img.getchannel('R')
                            lowres_img.save(lowres_path)
                        if highres_img.mode == "RGB":
                            highres_img = highres_img.getchannel('R')
                            highres_img.save(highres_path)

                        bicubic_img = lowres_img.resize(highres_img.size,Image.BICUBIC)
                        if bicubic_img.mode == "RGB":
                            bicubic_img = bicubic_img.getchannel('R')

                        bicubic_img.save(bicubic_path)
                    bicubic.append(bicubic_path)

                    if max_samples and index > max_samples:
                        break

        item_transforms = [RandomCrop(random_crop)] if random_crop else []

        datablock = DataBlock(
            blocks=(ImageBlock(cls=PILImageBW), ImageBlock(cls=PILImageBW)),
            splitter=FuncSplitter(is_validation_image),
            get_y=get_y,
            item_tfms=item_transforms,
        )

        dataloaders = DataLoaders.from_dblock(
            datablock, 
            source=bicubic,
            bs=batch_size,
        )

        dataloaders.c = 1

        return dataloaders

    def inference_dataloader(
        self, 
        learner, 
        items:List[Path] = None, 
        item_dir: Path = ta.Param(None, help="A directory with images to upscale."), 
        width:int = ta.Param(500, help="The width of the final image."), 
        height:int = ta.Param(None, help="The height of the final image."), 
        **kwargs
    ):
        if not items:
            items = []
        if isinstance(items, (Path, str)):
            items = [items]
        if item_dir:
            items += self.get_items(item_dir)

        items = [Path(item) for item in items]
        self.items = items
        dataloader = learner.dls.test_dl(items, with_labels=True, **kwargs)

        height = height or width
        dataloader.after_item = Pipeline( [Resize(height, width), ToTensor] )

        return dataloader

    def output_results(self, results, return_images=False, **kwargs):
        list_to_return = []
        for item, result in zip(self.items, results[0]):
            extension = item.name[item.name.rfind(".")+1:].lower() 
            stub = item.name[:-len(extension)]
            new_name = f"{stub}upscaled.{extension}"
            new_path = item.parent/new_name
            pixels = torch.clip(result[0]*255, min=0, max=255)
            
            im = Image.fromarray( pixels.cpu().detach().numpy().astype('uint8') )
            
            console.print(f"Upscaled '{item}' ⮕ '{new_path}'")
            im.save(new_path)
            list_to_return.append(im if return_images else new_path)

        return list_to_return

    def model(
        self,
        initial_features:int = ta.Param(
            64,
            tune=True, 
            tune_min=16,
            tune_max=256,
            help="The number of features after the initial CNN layer."
        ),
        growth_factor:int = ta.Param(
            2.0,
            tune=True, 
            tune_min=1.0,
            tune_max=4.0,
            log=True,
            help="The factor to grow the number of convolutional filters each time the model downscales."
        ),
    ):
        return ResidualUNet(in_channels=1, out_channels=1, initial_features=initial_features, growth_factor=growth_factor, dim=2)

    def loss_func(self, mse:bool=True):
        """
        Returns the loss function to use with the model.

        By default the L1 loss or the Mean Absolute Error (MAE) is used.
        See 10.1029/2019WR026052
        """
        if mse:
            return nn.MSELoss()
        return nn.L1Loss()

    def metrics(self):
        metrics = super().metrics()
        metrics.extend([psnr])
        return metrics

    def validate_individual(self, csv, item_dir: Path = ta.Param(None, help="The dir with the images to upscale."), **kwargs):
        path = call_func(self.pretrained_local_path, **kwargs)
        learner = load_learner(path)
        with open(csv, 'w') as f:
            items = self.get_items(Path(item_dir).expanduser().resolve())
            print("image", "loss", "l2", "psnr", sep=",", file=f)
            for item in items:
                print(item)
                dataloader = learner.dls.test_dl([item], with_labels=True)
                values = learner.validate(dl=dataloader)
                print(item, *values, sep=",", file=f)


class Supercat3d(Supercat):
    def dataloaders(
        self,
        deeprock:Path = ta.Param(help="The path to the DeepRockSR-3D dataset."), 
        downsample_scale:DownsampleScale = ta.Param(DownsampleScale.X4.value, help="Should it use the 2x or 4x downsampled images.", case_sensitive=False),
        downsample_method:DownsampleMethod = ta.Param(DownsampleMethod.UNKNOWN.value, help="Should it use the default method to downsample (bicubic) or a random kernel (UNKNOWN)."),
        batch_size:int = ta.Param(default=8, help="The batch size."),
        force:bool = ta.Param(default=False, help="Whether or not to force the conversion of the tricubic upscaling."),
        max_samples:int = ta.Param(default=None, help="If set, then the number of input samples for training/validation is truncated at this number."),
        include_sand:bool = ta.Param(default=False, help="Including DeepSand-SR dataset."),
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which Supercat uses in training and prediction.

        Args:
            deeprock (Path): The path to the DeepRockSR-3D dataset.
            batch_size (int): The number of elements to use in a batch for training and prediction. Defaults to 32.
        """
        assert deeprock is not None

        deeprock = Path(deeprock)
        tricubic = []
        highres = []

        sources = ["carbonate3D","coal3D","sandstone3D"]

        if include_sand:
            sources += ["sand3D"]

        if isinstance(downsample_method, DownsampleMethod):
            downsample_method = downsample_method.value
        if isinstance(downsample_scale, DownsampleScale):
            downsample_scale = downsample_scale.value

        split_types = ["train","valid"] # There is also "test"
        # split_types = ["train","valid","test"] # hack
        
        for source in sources:
            for split_type in split_types:
                highres_dir = deeprock/source/f"{source}_{split_type}_HR"
                highres_split = self.get_items(highres_dir)
                highres.extend( highres_split )

                lowres_dir = deeprock/source/f"{source}_{split_type}_LR_{downsample_method}_{downsample_scale}"
                
                # We will save tricubic upscaled images
                tricubic_dir = deeprock/source/f"{source}_{split_type}_TRI_{downsample_method}_{downsample_scale}" 
                tricubic_dir.mkdir(exist_ok=True)

                for index, highres_path in enumerate(highres_split):
                    tricubic_path = tricubic_dir/highres_path.name

                    if not tricubic_path.exists() or force:
                        components = highres_path.name.split(".")
                        lowres_name = f'{components[0]}{downsample_scale.lower()}.{components[1]}'
                        lowres_path = lowres_dir/lowres_name
                        print(split_type, highres_path, tricubic_path, lowres_path)
                        
                        # upscale with tricubic interpolation
                        print("Upscaling with tricubic")
                        highres_img = read3D(highres_path)
                        lowres_img = read3D(lowres_path)

                        tricubic_img = interpolate3D(lowres_img, new_shape=highres_img.shape, method=InterpolationMethod.TRICUBIC)
                        write3D(tricubic_path, tricubic_img)
                    tricubic.append(tricubic_path)

                    if max_samples and index > max_samples:
                        break

        assert "test" not in split_types
        datablock = DataBlock(
            blocks=(ImageBlock3D, ImageBlock3D),
            splitter=FuncSplitter(is_validation_image),
            get_y=get_y_3d,
        )

        dataloaders = DataLoaders.from_dblock(
            datablock, 
            source=tricubic,
            bs=batch_size,
        )

        dataloaders.c = 1

        return dataloaders    

    def model(
        self,
        video_unet:bool = False, 
        pretrained:bool = True,
        initial_features:int = ta.Param(
            64,
            tune=True, 
            tune_min=16,
            tune_max=256,
            help="The number of features after the initial CNN layer."
        ),
        growth_factor:int = ta.Param(
            2.0,
            tune=True, 
            tune_min=1.0,
            tune_max=4.0,
            log=True,
            help="The factor to grow the number of convolutional filters each time the model downscales."
        ),
        # more should be added
    ):
        if video_unet:
            return VideoUnet3d(in_channels=1, out_channels=1, pretrained=pretrained)

        return ResidualUNet(in_channels=1, out_channels=1, initial_features=initial_features, growth_factor=growth_factor, dim=3)

    def build_learner_func(self):
        return Learner

    def learner_kwargs(
        self,
        output_dir: Path = ta.Param("./outputs", help="The location of the output directory."),
        **kwargs,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        return dict(
            loss_func=self.loss_func(),
            metrics=self.metrics(),
            path=output_dir,
        )

    def get_items(self, directory):
        directory = Path(directory)
        return list(directory.glob("*.mat"))

    def validate_individual(self, csv, do_nothing:bool=False, item_dir: Path = ta.Param(None, help="The dir with the images to upscale."), **kwargs):
        path = call_func(self.pretrained_local_path, **kwargs)
        learner = load_learner(path)
        if do_nothing:
            learner.model = DoNothing() 
                 
        with open(csv, 'w') as f:
            items = self.get_items(Path(item_dir).expanduser().resolve())
            print("image", "loss", "l2", "psnr", sep=",", file=f)
            for item in items:
                dataloader = learner.dls.test_dl([item], with_labels=True)
                values = learner.validate(dl=dataloader)
                print(item, *values)
                print(item, *values, sep=",", file=f)

    def inference_dataloader(
        self, 
        learner, 
        items:List[Path] = None, 
        width:int = ta.Param(100, help="The width of the final volume."), 
        height:int = ta.Param(None, help="The height of the final volume."), 
        depth:int = ta.Param(None, help="The depth of the final volume."), 
        **kwargs
    ):
        if not items:
            items = []
        if isinstance(items, (Path, str)):
            items = [items]

        items = [Path(item) for item in items]
        self.items = items
        dataloader = learner.dls.test_dl(items, with_labels=True, **kwargs)

        height = height or width
        depth = depth or width
        dataloader.after_item = Pipeline( [InterpolateTransform(width, height, depth), ToTensor] )
        return dataloader

    def output_results(self, results, return_volumes=False, **kwargs):
        list_to_return = []
        for item, result in zip(self.items, results[0]):
            extension = item.name[item.name.rfind(".")+1:].lower() 
            stub = item.name[:-len(extension)]
            new_name = f"{stub}upscaled.{extension}"
            new_path = item.parent/new_name
            write3D(new_path, result[0].cpu().detach().numpy())            
                        
            console.print(f"Upscaled '{item}' ⮕ '{new_path}'")
            list_to_return.append(result[0] if return_volumes else new_path)

        return list_to_return
