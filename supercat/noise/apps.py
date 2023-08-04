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

from supercat.models import ResidualUNet
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
        dim:int=2,
        depth:int=500,
        width:int=500,
        height:int=500,
        batch_size:int=16,
        item_count:int=1024,
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
    
    def extra_callbacks(self, diffusion:bool=True):
        self.diffusion = diffusion
        callbacks = [ShrinkCallBack(factor=4)]
        if self.diffusion:
            callbacks.append(DDPMCallback())
        return callbacks
    
    def inference_callbacks(self, diffusion:bool=True):
        callbacks = [ShrinkCallBack(factor=4)]
        if diffusion:
            callbacks.append(DDPMSamplerCallback())
        return callbacks
    
    def model(self):
        return ResidualUNet(dim=self.dim, in_channels=3 if self.diffusion else 1)

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
