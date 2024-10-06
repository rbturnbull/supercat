import torch
import lightning as L
from rich.progress import track


class DDPMCallback(L.Callback):
    """
    Derived from https://wandb.ai/capecape/train_sd/reports/How-To-Train-a-Conditional-Diffusion-Model-From-Scratch--VmlldzoyNzIzNTQ1#using-fastai-to-train-your-diffusion-model
    """
    def __init__(self, n_steps:int=1000, s:float = 0.008):
        self.n_steps = n_steps
        self.s = s
        """
        Based on https://arxiv.org/abs/2102.09672,
        the consine noise scheduler is defined as:
        
        T = num_step
        s = offset
        
        f(t) = cos(( t / T + s ) / ( 1 + s ) * pi * 0.5 ) ^ 2

        alpha_bar(t) = f(t)/f(0); t = {1, ..., T}
        
        alpha(t) = alpha_bar(t) / alpha_bar(t-1); t = {2,..., T}
                   alpha_bar(t); t = 1
        
        However, when T is large, calculating alpha_bar(t) with f(t)/f(0) will
        case alpha_bar(1) and alpha(1) converge to 1 due to precision issue
        
        As a result, defining alpha_bar(t) = f(t) scalces up alpha_bar(t) and sovle
        the issue.
        """
        t = torch.arange(self.n_steps + 1)
        self.alpha_bar = torch.cos((t/self.n_steps+self.s)/(1+self.s) * torch.pi * 0.5)**2
        self.alpha = self.alpha_bar/torch.cat([torch.ones(2), self.alpha_bar[1:-1]])
        self.beta = 1.0 - self.alpha
        self.sigma = torch.sqrt(self.beta)

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        """
        x: (batch_size, c, d, h, w)
        """
        lr, hr, residual = batch

        noise = torch.randn_like(hr)

        batch_size = hr.shape[0]
        dim = len(hr.shape) - 2

        # lookup noise schedule
        if self.training:
            t = torch.randint(1, self.n_steps + 1, (batch_size,), dtype=torch.long) # select random timesteps
        else:
            # if validation, use a spread of timesteps that is deterministic
            # so valdiation results can be properly compared
            t = torch.linspace(1, self.n_steps, batch_size, dtype=torch.long)

        if dim == 2:
            alpha_bar_t = self.alpha_bar[t, None, None, None]
        else:
            alpha_bar_t = self.alpha_bar[t, None, None, None, None]
        alpha_bar_t = alpha_bar_t.to(self.dls.device)

        # noisify the image
        xt =  torch.sqrt(alpha_bar_t) * hr + torch.sqrt(1-alpha_bar_t) * noise 

        # Stack input with low-resolution image (upscaled) at channel dim,
        # then pass the stacked image along with the noise level as tuple to the model
        input = torch.cat([xt, lr], dim=1)

        return input, alpha_bar_t.view((batch_size, 1)), hr, noise


class DDPMSamplerCallback(L.Callback):
    def before_batch(self):
        lr = self.xb[0]
        batch_size = lr.shape[0]

        # Generate a batch of random noise to start with
        xt = torch.randn_like(lr)

        outputs = [xt]
        for t in track(reversed(range(1, self.n_steps)), total=self.n_steps, description="Performing diffusion steps for batch:"):
            z = torch.randn(xt.shape, device=xt.device) if t > 1 else torch.zeros(xt.shape, device=xt.device)
            alpha_t = self.alpha[t] # get noise level at current timestep
            alpha_bar_t = self.alpha_bar[t]
            sigma_t = self.sigma[t]

            predicted_noise = self.model(torch.cat([xt, lr], dim=1), torch.full((batch_size, 1), alpha_bar_t, device=xt.device))

            # predict x_(t-1) in accordance to Algorithm 2 in paper
            xt = (1/torch.sqrt(alpha_t)) * (xt - ((1-alpha_t)/torch.sqrt(1-alpha_bar_t)) * predicted_noise)  + sigma_t*z
            outputs.append(xt)

        # self.learn.pred = (torch.stack(outputs, dim=1),)
        self.learn.pred = (xt,)

        raise CancelBatchException

