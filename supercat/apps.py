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
from fastcore.transform import Pipeline
from fastai.vision.augment import Resize
from fastai.data.transforms import FuncSplitter
from fastai.learner import load_learner
from PIL import Image
from functools import partial
from fastai.vision.data import ImageBlock, TensorImage
from fastai.vision.core import PILImageBW, TensorImageBW
from supercat.noise.apps import * # remove this

from supercat.models import ResidualUNet, calc_initial_features_residualunet
from supercat.transforms import ImageBlock3D, RescaleImage, write3D, read3D, InterpolateTransform, RescaleImageMinMax, CropTransform
from supercat.enums import DownsampleScale, DownsampleMethod, PaddingMode
from supercat.diffusion import DDPMCallback, DDPMSamplerCallback
from skimage.transform import resize as skresize

from rich.console import Console
console = Console()


def is_validation_image(item:tuple):
    "Returns True if this image should be part of the validation set i.e. if the parent directory doesn't have the string `_train_` in it."
    return "_train_" not in item.parent.name


def get_y(item, pattern=r"_BI_.*"):
    dir_name = re.sub(pattern, "_HR", item.parent.name)            
    return item.parent.parent/dir_name/item.name


class Supercat(ta.TorchApp):
    in_channels = 1

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
                            components = highres_path.name.split(".")
                            lowres_name = f'{components[0]}{downsample_scale.lower()}.{components[1]}'
                            lowres_path = lowres_dir/lowres_name
                            print(split_type, highres_path, upscale_path, lowres_path)
                            
                            # upscale with tricubic interpolation
                            print("Upscaling with tricubic")
                            highres_img = read3D(highres_path)
                            lowres_img = read3D(lowres_path)

                            tricubic_img = skresize(lowres_img, highres_img.shape, order=3)
                            write3D(upscale_path, tricubic_img)

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
        
    def model(
        self, 
        pretrained:Path=None,
        initial_features:int = ta.Param(
            None,
            help="The number of features after the initial CNN layer. If not set then it is derived from the MACC."
        ),
        growth_factor:float = ta.Param(
            2.0,
            tune=True, 
            tune_min=1.0,
            tune_max=4.0,
            tune_log=True,
            help="The factor to grow the number of convolutional filters each time the model downscales."
        ),
        kernel_size:int = ta.Param(
            3,
            tune=True, 
            tune_choices=[3,5,7],
            help="The size of the kernel in the convolutional layers."
        ),
        stub_kernel_size:int = ta.Param(
            7,
            tune=True, 
            tune_choices=[5,7,9],
            help="The size of the kernel in the initial stub convolutional layer."
        ),
        downblock_layers:int = ta.Param(
            4,
            tune=True, 
            tune_min=2,
            tune_max=5,
            help="The number of layers to downscale (and upscale) in the UNet."
        ),
        attn_layers:str = ta.Param(
            "",
            help="Whether or not to use self attention in the model. Specify the indices of the layers, seperated with ',', to include self attention layer. Index starts from 0."
        ),
        position_emb_dim:int = ta.Param(
            None,
            help="The dimension of the positional embedding. If not set, the model will not be conditioned on positional info."
        ),
        affine:bool = ta.Param(
            False,
            help="Whether or not to use affine transformations in feature wise transformation."
        ),
        macc:int = ta.Param(
            default=132_000,
            help=(
                "The approximate number of multiply or accumulate operations in the model per pixel/voxel. " +
                "Used to set initial_features if it is not provided explicitly."
            ),
        ),
        padding_mode: PaddingMode = ta.Param(
            PaddingMode.REFLECT.value, 
            help="The padding mode for convolution layers", 
            case_sensitive=False
        )
    ):
        if pretrained:
            learner = load_learner(pretrained)
            return learner.model

        dim  = getattr(self, "dim", 3)
        attn_layers = tuple(map(int, filter(None, attn_layers.split(','))))

        if not initial_features:
            assert macc

            initial_features = calc_initial_features_residualunet(
                macc=macc,
                dim=dim,
                growth_factor=growth_factor,
                kernel_size=kernel_size,
                stub_kernel_size=stub_kernel_size,
                downblock_layers=downblock_layers,
            )

        return ResidualUNet(
            dim=dim,
            padding_mode=padding_mode,
            in_channels=self.in_channels,
            out_channels=1,
            initial_features=initial_features,
            growth_factor=growth_factor,
            kernel_size=kernel_size,
            downblock_layers=downblock_layers,
            attn_layers=attn_layers,
            position_emb_dim=position_emb_dim,
            use_affine=affine,
        )


    def loss_func(self):
        """
        Returns the loss function to use with the model.
        """
        return F.smooth_l1_loss

    def inference_dataloader(
        self, 
        learner, 
        dim:int = ta.Param(default=2, help="The dimension of the dataset. 2 or 3."),
        items:List[Path] = None, 
        item_dir: Path = ta.Param(None, help="A directory with images to upscale."), 
        width:int = ta.Param(500, help="The width of the final image/volume."), 
        height:int = ta.Param(None, help="The height of the final image/volume."), 
        depth:int = ta.Param(None, help="The depth of the final image/volume."), 
        start_x:int=None,
        end_x:int=None,
        start_y:int=None,
        end_y:int=None,
        start_z:int=None,
        end_z:int=None,        
        **kwargs
    ):  
        self.dim = dim

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
        depth = depth or width
        
        interpolation = InterpolateTransform(depth=depth, height=height, width=width, dim=dim)
        crop_transform = CropTransform(
            start_x=start_x, end_x=end_x,
            start_y=start_y, end_y=end_y,
            start_z=start_z, end_z=end_z,
        )
        self.rescaling = RescaleImageMinMax()
        dataloader.after_item = Pipeline( [crop_transform, interpolation, self.rescaling, ToTensor] )
        if isinstance(dataloader.after_batch[1], RescaleImage):
            dataloader.after_batch = Pipeline( *(dataloader.after_batch[:1] + dataloader.after_batch[2:]) ) if dim == 2 else Pipeline([])

        return dataloader

    def output_results(
        self, 
        results, 
        return_data:bool=False, 
        output_dir: Path = ta.Param(None, help="The location of the output directory. If not given then it uses the directory of the item."),
        suffix:str = ta.Param("", help="The file extension for the output file."),
        **kwargs,
    ):
        list_to_return = []
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)

        for item, result in zip(self.items, results[0]):
            my_suffix = suffix or item.suffix
            if my_suffix[0] != ".":
                my_suffix = "." + my_suffix

            new_name = item.with_suffix("").name + f".upscaled{my_suffix}"
            my_output_dir = output_dir or item.parent
            new_path = my_output_dir/new_name

            dim = len(result.shape) - 1
            if dim == 2:
                # hack get extrema to rescale
                data = np.asarray(Image.open(item).convert('L'))
                min, max = Image.open(item).convert('L').getextrema()
                result[0] = self.rescaling.decodes(result[0], min, max)

                pixels = torch.clip(result[0], min=0, max=255)
                im = Image.fromarray( pixels.cpu().detach().numpy().astype('uint8') )
                im.save(new_path)
            else:
                # hack get extrema to rescale
                data = read3D(item)
                min, max = data.min(), data.max()
                result[0] = self.rescaling.decodes(result[0], min, max)

                write3D(new_path, result[0].cpu().detach().numpy())            
                            
            list_to_return.append(result[0] if return_data else new_path)
            console.print(f"Upscaled '{item}' ⮕ '{new_path}'")

        return list_to_return
    
    def pretrained_location(
        self,
        dim:int = ta.Param(default=2, help="The dimension of the dataset. 2 or 3."),
    ) -> str:
        assert dim in [2,3]
        if dim == 2:
            return f"https://github.com/rbturnbull/supercat/releases/download/v0.2.1/supercat-{dim}D.0.2.pkl"
        return f"https://github.com/rbturnbull/supercat/releases/download/v0.3.0/supercat-{dim}D.0.3.pkl"        


class SupercatDiffusion(Supercat):
    in_channels = 2
    
    def extra_callbacks(self):
        return [DDPMCallback()]
    
    def inference_callbacks(self):
        return [DDPMSamplerCallback()]        

    def pretrained_location(
        self,
        dim:int = ta.Param(default=2, help="The dimension of the dataset. 2 or 3."),
    ) -> str:
        assert dim in [2,3]
        if dim == 2:
            return f"https://github.com/rbturnbull/supercat/releases/download/v0.2.1/supercat-diffusion-{dim}D.0.2.pkl"
        return f"https://github.com/rbturnbull/supercat/releases/download/v0.3.0/supercat-diffusion-{dim}D.0.3.pkl"

    # def output_results(
    #     self, 
    #     results, 
    #     output_dir: Path = ta.Param("./outputs", help="The location of the output directory."),
    #     diffusion_gif:bool=False,        
    #     diffusion_gif_fps:float=ta.Param(120.0, help="The frames per second to use when generating the gif."),
    #     **kwargs,
    # ):
    #     breakpoint()
    #     # final_results = [[result[-1] for result in results[0][0]]]
    #     to_return = super().output_results(results, output_dir=output_dir, **kwargs)

    #     if diffusion_gif:
    #         assert self.dim == 2

    #         output_dir = Path(output_dir)
    #         print(f"Saving {len(results[0])} generated images:")

    #         transform = T.ToPILImage()
    #         output_dir.mkdir(exist_ok=True, parents=True)
    #         images = []
    #         for index, image in enumerate(results[0][0]):
    #             path = output_dir/f"image.{index}.png"
                
    #             image = transform(torch.clip(image[0]/2.0 + 0.5, min=0.0, max=1.0))
    #             images.append(image)
    #         print(f"\t{path}")
    #         images[0].save(output_dir/f"image.gif", save_all=True, append_images=images[1:], fps=diffusion_gif_fps)

    #     return to_return

if __name__ == "__main__":
    SupercatDiffusion.main()
