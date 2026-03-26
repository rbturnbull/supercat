import torch
import torch.nn as nn

from .diffusion import create_diffusion


class DiffusionPredictionModel(nn.Module):
    def __init__(self, diffusion_model: nn.Module, num_sampling_steps: int = 250):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.diffusion = create_diffusion(str(num_sampling_steps))

    def forward(self, x):
        device = x.device
        z = torch.randn( x.shape, device=device)
        conditioned = x
        model_kwargs = dict(conditioned=conditioned)
        samples = self.diffusion.p_sample_loop(
            self.diffusion_model.forward, z.shape, z,
            clip_denoised=True, model_kwargs=model_kwargs,
            progress=True, device=device
        )
        return samples
