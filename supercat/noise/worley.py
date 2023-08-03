import torchapp as ta
import torch
from pathlib import Path
from fastai.callback.core import Callback, CancelBatchException
from fastai.data.block import DataBlock, TransformBlock
from fastai.data.core import DataLoaders
import torch.nn.functional as F
import numpy as np
from rich.progress import track
import torchvision.transforms as T


from supercat.models import ResidualUNet
from supercat.diffusion import DDPMCallback, DDPMSamplerCallback, wandb_process


class WorleyNoise:
    """
    Derived from:
        https://stackoverflow.com/a/65704227
        https://stackoverflow.com/q/65703414
    """
    def __init__(self, shape, density, n:int=0, seed=None):
        np.random.seed(seed)

        self.density = density
        self.shape = shape
        self.dims = len(self.shape)
        self.coords = [np.arange(s) for s in self.shape]
        self.points = None
        self.n = n
        
    def __call__(self):
        self.points = np.random.rand(self.density, self.dims) 

        for i, size in enumerate(self.shape):
            self.points[:, i] *= size

        axes = list(range(1, self.dims+1))
        squared_d = sum(
            np.expand_dims(
                np.power(self.points[:, i, np.newaxis] - self.coords[i], 2), 
                axis=axes[:i]+axes[i+1:]
            )
            for i in range(self.dims)
        )

        if self.n == 0:
            return np.sqrt(squared_d.min(axis=0))
        elif self.n is None:
            return np.sqrt(squared_d)
        return  np.sqrt(np.sort(squared_d, axis=0)[self.n])


class WorleyNoiseTensor(WorleyNoise):
    def __call__(self, *args):
        x = super().__call__()
        x = torch.from_numpy(x).float()
        x = x/x.max()*2.0 - 1.0
        x = x.unsqueeze(0)
        
        return x


# class WorleySR(ta.TorchApp):
#     def build_generator(self, shape):
#         return WorleyNoiseTensor(shape=shape, density=200)
    
#     def dataloaders(
#         self,
#         dim:int=2,
#         depth:int=500,
#         width:int=500,
#         height:int=500,
#         batch_size:int=16,
#         item_count:int=1024,
#     ):

#         shape = (height, width) if dim == 2 else (depth, height, width)
#         self.shape = shape
#         self.dim = dim

#         datablock = DataBlock(
#             blocks=(TransformBlock),
#             get_x=self.build_generator(shape),
#         )

#         dataloaders = DataLoaders.from_dblock(
#             datablock,
#             source=range(item_count),
#             bs=batch_size,
#         )

#         return dataloaders
    
#     def extra_callbacks(self, diffusion:bool=True):
#         self.diffusion = diffusion
#         callbacks = [ShrinkCallBack(factor=4)]
#         if self.diffusion:
#             callbacks.append(DDPMCallback())
#         return callbacks
    
#     def inference_callbacks(self, diffusion:bool=True):
#         callbacks = [ShrinkCallBack(factor=4)]
#         if diffusion:
#             callbacks.append(DDPMSamplerCallback())
#         return callbacks
    
#     def model(self):
#         return ResidualUNet(dim=self.dim, in_channels=3 if self.diffusion else 1)

#     def loss_func(self):
#         """
#         Returns the loss function to use with the model.
#         """
#         return F.smooth_l1_loss

#     def inference_dataloader(self, learner, **kwargs):
#         dataloader = learner.dls.test_dl([0], **kwargs) # output single test image
#         return dataloader

#     def output_results(
#         self, 
#         results, 
#         output_dir: Path = ta.Param("./outputs", help="The location of the output directory."),
#         fps:float=ta.Param(30.0, help="The frames per second to use when generating the gif."),
#         **kwargs,
#     ):
#         output_dir = Path(output_dir)
#         print(f"Saving {len(results)} generated images:")

#         transform = T.ToPILImage()
#         output_dir.mkdir(exist_ok=True, parents=True)
#         images = []
#         for index, image in enumerate(results[0]):
#             path = output_dir/f"image.{index}.jpg"
            
#             image = transform(torch.clip(image[0]/2.0 + 0.5, min=0.0, max=1.0))
#             image.save(path)
#             images.append(image)
#         print(f"\t{path}")
#         images[0].save(output_dir/f"image.gif", save_all=True, append_images=images[1:], fps=fps)


# if __name__ == "__main__":
#     WorleySR.main()
