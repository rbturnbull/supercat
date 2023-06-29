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

        self.learn.pred = (outputs,)

        raise CancelBatchException


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
    
    def extra_callbacks(self, diffusion:bool=True):
        self.diffusion = diffusion
        callbacks = [ShrinkCallBack(factor=2)]
        if self.diffusion:
            callbacks.append(DDPMCallback())
        return callbacks
    
    def inference_callbacks(self, diffusion:bool=True):
        callbacks = [ShrinkCallBack(factor=2)]
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


if __name__ == "__main__":
    WorleySR.main()
