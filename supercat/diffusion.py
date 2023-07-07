import torch
from rich.progress import track
from fastai.callback.core import Callback, CancelBatchException


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

        # self.learn.pred = (torch.stack(outputs, dim=1),)
        self.learn.pred = (xt,)

        raise CancelBatchException

