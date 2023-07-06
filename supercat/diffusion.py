import re
import torchapp as ta
from typing import List
import torch
import random
from pathlib import Path
from fastai.callback.core import Callback, CancelBatchException
from fastai.data.block import DataBlock, TransformBlock
from fastai.data.core import DataLoaders, DisplayedTransform
import torch.nn.functional as F
from rich.progress import track
from fastai.data.transforms import get_image_files
import torchvision.transforms as T
from fastai.data.transforms import ToTensor
from fastai.data.core import Tensor
from enum import Enum
from fastcore.transform import Pipeline
from fastai.vision.augment import Resize
from fastai.data.transforms import FuncSplitter
from fastai.learner import load_learner
from PIL import Image
from functools import partial
from fastai.vision.data import ImageBlock, TensorImage
from fastai.vision.core import PILImageBW, TensorImageBW
from supercat.worley import WorleyNoise, WorleyNoiseTensor
from supercat.fractal import *
from fastai.vision.augment import Dihedral

from supercat.models import ResidualUNet
from supercat.transforms import ImageBlock3D


class DownsampleScale(Enum):
    X2 = "X2"
    X4 = "X4"


class DownsampleMethod(Enum):
    DEFAULT = "default"
    UNKNOWN = "unknown"


def is_validation_image(item:tuple):
    "Returns True if this image should be part of the validation set i.e. if the parent directory doesn't have the string `_train_` in it."
    return "_train_" not in item.parent.name


def get_y(item, pattern=r"_BI_.*"):
    dir_name = re.sub(pattern, "_HR", item.parent.name)            
    return item.parent.parent/dir_name/item.name


class RescaleImage(DisplayedTransform):
    order = 20 #Need to run after IntToFloatTensor
    
    def encodes(self, item:TensorImage): 
        return item.float()*2.0 - 1.0


class DihedralCallback(Callback):
    def before_batch(self):
        """
        x: (batch_size, c, d, h, w)
        """
        xb = self.xb[0]
        yb = self.yb[0]

        k = random.randint(0,7)

        if k in [1,3,4,7]: 
            xb = xb.flip(-1)
            yb = yb.flip(-1)
        
        if k in [2,4,5,7]:
            xb = xb.flip(-2)
            yb = yb.flip(-2)

        if k in [3,5,6,7]: 
            xb = xb.transpose(-1,-2)
            yb = yb.transpose(-1,-2)

        self.learn.xb = (xb,)
        self.learn.yb = (yb,)



class DDPMCallback(Callback):
    """
    Derived from https://wandb.ai/capecape/train_sd/reports/How-To-Train-a-Conditional-Diffusion-Model-From-Scratch--VmlldzoyNzIzNTQ1#using-fastai-to-train-your-diffusion-model
    """
    def __init__(self, n_steps:int=1000, s:float = 0.008):
        self.n_steps = n_steps
        self.s = s

        t = torch.arange(self.n_steps)
        self.alpha_bar = torch.cos((t/self.n_steps+self.s)/(1+self.s) * torch.pi * 0.5)**2
        self.alpha = self.alpha_bar/torch.cat([torch.ones(1), self.alpha_bar[:-1]])
        self.beta = 1.0 - self.alpha
        self.sigma = torch.sqrt(self.beta)

    def before_batch(self):
        """
        x: (batch_size, c, d, h, w)
        """
        lr = self.xb[0]
        hr = self.yb[0]

        noise = torch.randn_like(hr)

        batch_size = hr.shape[0]
        dim = len(hr.shape) - 2

        # lookup noise schedule
        t = torch.randint(0, self.n_steps, (batch_size,), dtype=torch.long) # select random timesteps
        if dim == 2:
            alpha_bar_t = self.alpha_bar[t, None, None, None]
        else:
            alpha_bar_t = self.alpha_bar[t, None, None, None, None]
        alpha_bar_t = alpha_bar_t.to(self.dls.device)
        
        # noisify the image
        xt =  torch.sqrt(alpha_bar_t) * hr + torch.sqrt(1-alpha_bar_t) * noise 
        
        # Stack input with low-resolution image (upscaled) and noise level
        self.learn.xb = (torch.cat([xt, lr, alpha_bar_t.repeat(1,1,*hr.shape[2:])], dim=1),)
        self.learn.yb = (noise,) # we are trying to predict the noise


class DDPMSamplerCallback(DDPMCallback):
    def before_batch(self):        
        lr = self.xb[0]
        
        # Generate a batch of random noise to start with
        xt = torch.randn_like(lr)
        
        outputs = [xt] 
        for t in track(reversed(range(self.n_steps)), total=self.n_steps, description="Performing diffusion steps for batch:"):
            z = torch.randn(xt.shape, device=xt.device) if t > 0 else torch.zeros(xt.shape, device=xt.device)
            alpha_t = self.alpha[t] # get noise level at current timestep
            alpha_bar_t = self.alpha_bar[t]
            sigma_t = self.sigma[t]
            model_input = torch.cat(
                [xt, lr, alpha_bar_t.repeat(1,1,*lr.shape[2:]).to(xt.device)], 
                dim=1,
            )
            predicted_noise = self.model(model_input)
            
            # predict x_(t-1) in accordance to Algorithm 2 in paper
            xt = 1/torch.sqrt(alpha_t) * (xt - (1-alpha_t)/torch.sqrt(1-alpha_bar_t) * predicted_noise)  + sigma_t*z 
            outputs.append(xt)

        self.learn.pred = (torch.stack(outputs, dim=1),)

        raise CancelBatchException


class SupercatDiffusion(ta.TorchApp):
    def get_items(self, directory):
        if self.dim == 2:
            return get_image_files(directory)
        
        directory = Path(directory)
        return list(directory.glob("*.mat"))            

    def dataloaders(
        self,
        dim:int = ta.Param(default=2, help="The dimension of the dataset. 2 or 3."),
        deeprock:Path = ta.Param(help="The path to the DeepRockSR dataset."), 
        downsample_scale:DownsampleScale = ta.Param(DownsampleScale.X4.value, help="Should it use the 2x or 4x downsampled images.", case_sensitive=False),
        downsample_method:DownsampleMethod = ta.Param(DownsampleMethod.UNKNOWN.value, help="Should it use the default method to downsample (bicubic) or a random kernel (UNKNOWN)."),
        batch_size:int = ta.Param(default=10, help="The batch size."),
        force:bool = ta.Param(default=False, help="Whether or not to force the conversion of the bicubic upscaling."),
        max_samples:int = ta.Param(default=None, help="If set, then the number of input samples for training/validation is truncated at this number."),
        include_sand:bool = ta.Param(default=False, help="Including DeepSand-SR dataset."),
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which Supercat uses in training and prediction.
        """
        assert deeprock is not None

        self.dim = dim
        deeprock = Path(deeprock)
        upscaled = []
        highres = []
        
        # sources = ["shuffled2D"]
        sources = [f"carbonate{dim}D",f"coal{dim}D",f"sandstone{dim}D"]
        if include_sand:
            sources.append(f"sand{dim}D")

        if isinstance(downsample_method, DownsampleMethod):
            downsample_method = downsample_method.value

        if isinstance(downsample_scale, DownsampleScale):
            downsample_scale = downsample_scale.value

        split_types = ["train","valid"] # There is also "test"
        # split_types = ["train","valid","test"] # hack

        UP = "BI" if dim == 2 else "TRI"
        
        for source in sources:
            for split_type in split_types:
                highres_dir = deeprock/source/f"{source}_{split_type}_HR"
                highres_split = self.get_items(highres_dir)
                highres.extend( highres_split )

                lowres_dir = deeprock/source/f"{source}_{split_type}_LR_{downsample_method}_{downsample_scale}"
                
                # We will save upscaled images
                upscale_dir = deeprock/source/f"{source}_{split_type}_{UP}_{downsample_method}_{downsample_scale}" 
                upscale_dir.mkdir(exist_ok=True)

                for index, highres_path in enumerate(highres_split):
                    upscale_path = upscale_dir/highres_path.name

                    if not upscale_path.exists() or force:
                        components = highres_path.name.split(".")
                        lowres_name = f'{components[0]}{downsample_scale.lower()}.{components[1]}'
                        lowres_path = lowres_dir/lowres_name
                        print(split_type, highres_path, upscale_path, lowres_path)
                        
                        # upscale with upscale interpolation
                        print("Upscaling")
                        if dim == 2:
                            highres_img = Image.open(highres_path)
                            lowres_img = Image.open(lowres_path)

                            # Convert to single channel
                            if lowres_img.mode == "RGB":
                                lowres_img = lowres_img.getchannel('R')
                                lowres_img.save(lowres_path)
                            if highres_img.mode == "RGB":
                                highres_img = highres_img.getchannel('R')
                                highres_img.save(highres_path)

                            upscale_img = lowres_img.resize(highres_img.size,Image.upscale)
                            if upscale_img.mode == "RGB":
                                upscale_img = upscale_img.getchannel('R')

                            upscale_img.save(upscale_path)
                        else:
                            raise NotImplementedError("Upscaling for 3D is not implemented yet.")

                    upscaled.append(upscale_path)

                    if max_samples and index > max_samples:
                        break

        if len(upscaled) == 0:
            raise ValueError("No images found.")

        if dim == 2:
            blocks = (ImageBlock(cls=PILImageBW), ImageBlock(cls=PILImageBW))
        else:
            blocks = (ImageBlock3D, ImageBlock3D,)

        datablock = DataBlock(
            blocks=blocks,
            splitter=FuncSplitter(is_validation_image),
            get_y=get_y if dim == 2 else partial(get_y, pattern=r"_TRI_.*"),
            batch_tfms=[RescaleImage],
        )

        dataloaders = DataLoaders.from_dblock(
            datablock, 
            source=upscaled,
            bs=batch_size,
        )

        dataloaders.c = 1

        return dataloaders
    
    def extra_callbacks(self, diffusion:bool=True):
        self.diffusion = diffusion
        callbacks = [DihedralCallback()]
        if self.diffusion:
            callbacks.append(DDPMCallback())
        return callbacks
    
    def inference_callbacks(self, diffusion:bool=True):
        callbacks = []
        if diffusion:
            callbacks.append(DDPMSamplerCallback())
        return callbacks
    
    def model(self, pretrained:Path=None):
        if pretrained:
            learner = load_learner(pretrained)
            return learner.model

        return ResidualUNet(dim=self.dim, in_channels=3 if self.diffusion else 1)

    def loss_func(self):
        """
        Returns the loss function to use with the model.
        """
        return F.smooth_l1_loss

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
        dataloader.transform = dataloader.transform[:1] # ignore the get_y function
        height = height or width
        dataloader.after_item = Pipeline( [Resize(height, width), ToTensor] )

        return dataloader

    def output_results(
        self, 
        results, 
        output_dir: Path = ta.Param("./outputs", help="The location of the output directory."),
        fps:float=ta.Param(30.0, help="The frames per second to use when generating the gif."),
        **kwargs,
    ):
        output_dir = Path(output_dir)
        print(f"Saving {len(results[0])} generated images:")

        transform = T.ToPILImage()
        transform(torch.clip(results[1][0]/2.0 + 0.5, min=0.0, max=1.0)).save("results1.png")
        output_dir.mkdir(exist_ok=True, parents=True)
        images = []
        for index, image in enumerate(results[0][0]):
            path = output_dir/f"image.{index}.png"
            
            image = transform(torch.clip(image[0]/2.0 + 0.5, min=0.0, max=1.0))
            image.save(path)
            images.append(image)
        print(f"\t{path}")
        images[-1].save(output_dir/f"final.tif")

        images[0].save(output_dir/f"image.gif", save_all=True, append_images=images[1:], fps=fps)


if __name__ == "__main__":
    SupercatDiffusion.main()
