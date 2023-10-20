import torchapp as ta
import torch
from pathlib import Path
from fastai.callback.core import Callback
from fastai.data.block import DataBlock, TransformBlock
from fastai.data.core import DataLoaders
from fastai.data.transforms import IndexSplitter
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T

from supercat.models import ResidualUNet, calc_initial_features_residualunet
from supercat.diffusion import DDPMCallback, DDPMSamplerCallback, wandb_process

from supercat.noise.fractal import FractalNoiseTensor
from supercat.noise.worley import WorleyNoiseTensor

class ShrinkCallBack(Callback):
    def __init__(self, factor:int=4, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def before_batch(self):
        hr = self.xb[0]
        lr_shape = tuple(s//self.factor for s in hr.shape[2:])
        mode = "bilinear" if len(lr_shape) == 2 else "trilinear"
        lr = F.interpolate(hr, lr_shape, mode=mode)
        pseudo_hr = F.interpolate(lr, hr.shape[2:], mode=mode)

        self.learn.xb = (pseudo_hr,)
        self.learn.yb = (hr,)


class NoiseTensorGenerator():
    def __init__(self, shape, worley_density:int=0, fractal_proportion:float=0.5):
        self.fractal = FractalNoiseTensor(shape)
        
        if not worley_density:
            worley_density = 200 if len(shape) == 2 else 40
        
        self.worley = WorleyNoiseTensor(shape, density=worley_density)
        self.shape = shape
        self.fractal_proportion = fractal_proportion
    
    def __call__(self, *args, **kwargs):
        return self.fractal(*args, **kwargs) if np.random.rand() < self.fractal_proportion else self.worley(*args, **kwargs)


class NoiseSR(ta.TorchApp):    
    def dataloaders(
        self,
        dim:int = ta.Param(default=2, help="The dimension of the dataset. 2 or 3."),
        depth:int=ta.Param(default=500, help="The depth of the noise image."),
        width:int=ta.Param(default=500, help="The width of the noise image."),
        height:int=ta.Param(default=500, help="The height of the noise image."),
        batch_size:int=ta.Param(default=16, help="The batch size for training."),
        item_count:int=ta.Param(default=1024, help="The height of the noise image."),
        worley_density:int=0,
        fractal_proportion:float=0.5,
    ):

        shape = (height, width) if dim == 2 else (depth, height, width)
        self.shape = shape
        self.dim = dim

        datablock = DataBlock(
            blocks=(TransformBlock),
            get_x=NoiseTensorGenerator(shape, worley_density=worley_density, fractal_proportion=fractal_proportion),
            splitter=IndexSplitter(list(range(batch_size))),
        )

        dataloaders = DataLoaders.from_dblock(
            datablock,
            source=range(item_count),
            bs=batch_size,
        )

        return dataloaders
    
    def extra_callbacks(self, diffusion:bool=ta.Param(True, help="Whether or not to create diffusion model.")):
        self.diffusion = diffusion
        callbacks = [ShrinkCallBack(factor=4)]
        if self.diffusion:
            callbacks.append(DDPMCallback())
        return callbacks
    
    def inference_callbacks(self, diffusion:bool=ta.Param(True, help="Whether or not to create diffusion model.")):
        callbacks = [ShrinkCallBack(factor=4)]
        if diffusion:
            callbacks.append(DDPMSamplerCallback())
        return callbacks

    def model(
        self,
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

    ):
        dim = getattr(self, "dim", 2)
        diffusion = getattr(self, "diffusion", False)
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
            in_channels=2 if diffusion else 1,
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

    def inference_dataloader(self, learner, **kwargs):
        dataloader = learner.dls.test_dl([0], **kwargs) # output single test image
        return dataloader

    def output_results(
        self, 
        results, 
        output_dir: Path = ta.Param("./outputs", help="The location of the output directory."),
        fps:float=ta.Param(30.0, help="The frames per second to use when generating the gif."),
        **kwargs,
    ):
        output_dir = Path(output_dir)
        print(f"Saving {len(results)} generated images:")

        transform = T.ToPILImage()
        output_dir.mkdir(exist_ok=True, parents=True)
        images = []
        for index, image in enumerate(results[0]):
            path = output_dir/f"image.{index}.jpg"
            
            image = transform(torch.clip(image[0]/2.0 + 0.5, min=0.0, max=1.0))
            image.save(path)
            images.append(image)
        print(f"\t{path}")
        images[0].save(output_dir/f"image.gif", save_all=True, append_images=images[1:], fps=fps)

    def monitor(self):
        return "train_loss"


if __name__ == "__main__":
    NoiseSR.main()
