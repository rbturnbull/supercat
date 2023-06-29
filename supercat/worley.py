import torchapp as ta
import torch
from fastai.callback.core import Callback
from fastai.data.block import DataBlock, TransformBlock
from fastai.data.core import DataLoaders
import torch.nn.functional as F
import numpy as np

from supercat.models import ResidualUNet

class ShrinkCallBack(Callback):
    def __init__(self, factor:int=2, **kwargs):
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


class WorleySR(ta.TorchApp):
    def extra_callbacks(self):
        return [ShrinkCallBack(factor=2)]

    def dataloaders(
        self,
        dim:int=2,
        depth:int=500,
        width:int=500,
        height:int=500,
        batch_size:int=16,
        density:int=200,
        item_count:int=1000,
    ):

        shape = (height, width) if dim == 2 else (depth, height, width)
        self.shape = shape
        self.dim = dim

        generator = WorleyNoiseTensor(
            shape=shape, 
            density=density,
        )

        datablock = DataBlock(
            blocks=(TransformBlock),
            get_x=generator,
        )

        dataloaders = DataLoaders.from_dblock(
            datablock,
            source=range(item_count),
            bs=batch_size,
        )

        return dataloaders
    
    def model(self):
        return ResidualUNet(dim=self.dim)

    def loss_func(self):
        """
        Returns the loss function to use with the model.
        """
        return F.smooth_l1_loss


if __name__ == "__main__":
    WorleySR.main()
