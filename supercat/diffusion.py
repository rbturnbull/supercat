import torch
from rich.progress import track
from fastai.callback.core import Callback, CancelBatchException
import wandb
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from fastcore.dispatch import typedispatch

from supercat.visualization import add_volume_face_traces

@typedispatch
def wandb_process(x, y, samples, outs, preds):
    table = wandb.Table(columns=["Sample"])
    index = 0
    for (sample_input, sample_target), prediction in zip(samples, outs):
        xt = sample_input[0]
        lr = sample_input[1]
        alpha_bar_t = sample_input[2,0,0]
        noise = sample_target[0]
        dim = len(xt.shape)

        if dim == 3:
            alpha_bar_t = alpha_bar_t[0]

        
        #xt =  torch.sqrt(alpha_bar_t) * hr + torch.sqrt(1-alpha_bar_t) * noise 
        hr = (xt - torch.sqrt(1-alpha_bar_t) * noise)/torch.sqrt(alpha_bar_t)
        hr_predicted = (xt - torch.sqrt(1-alpha_bar_t) * prediction[0][0])/torch.sqrt(alpha_bar_t)


        specs = None
        if dim == 3:
            specs = [[{'type':"surface"}, {'type':"surface"}, {'type':"surface"}, {'type':"surface"}, ]]

        fig = make_subplots(
            rows=1, 
            cols=4,
            subplot_titles=(
                "Low Resolution", 
                f"Noise: {alpha_bar_t}", 
                "Prediction (Single Shot)",
                "Ground Truth",
            ),
            vertical_spacing = 0.02,
            horizontal_spacing = 0.02,
            specs=specs,
        )

        if dim == 2:
            def add_trace(z, col):
                fig.add_trace( go.Heatmap(z=lr, zmin=-1.0, zmax=1.0, autocolorscale=False, showscale=False), row=1, col=col)        
            add_trace(lr, 1)
            add_trace(noise, 2)
            add_trace(hr_predicted, 3)
            add_trace(hr, 4)
            fig.update_traces(
                zmax=-1.0,
                zmin=1.0,
            )
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
        else:
            add_volume_face_traces(fig, lr, row=1, col=1)
            add_volume_face_traces(fig, noise, row=1, col=2)
            add_volume_face_traces(fig, hr_predicted, row=1, col=3)
            add_volume_face_traces(fig, hr, row=1, col=4)
            fig.update_layout(coloraxis=dict(colorscale='gray', cmin=-1.0, cmax=1.0, showscale=False), showlegend=False)
            axis = dict(showgrid=False, showticklabels=False, showaxeslabels=False, title="", showbackground=False)
            scene = dict(
                xaxis=axis,
                yaxis=axis,
                zaxis=axis,
            )
            fig.update_layout(
                scene1=scene,
                scene2=scene,
                scene3=scene,
                scene4=scene,
            )

        # fig.write_html("plotly.html", auto_play = False) 
        fig.update_layout(
            height=400,
            width=1200,
        )

        filename = f"plotly{index}.png"
        fig.write_image(filename) 
        table.add_data(
            wandb.Image(filename),
        )
        index += 1
        
    return {"Predictions": table}


class DDPMCallback(Callback):
    """
    Derived from https://wandb.ai/capecape/train_sd/reports/How-To-Train-a-Conditional-Diffusion-Model-From-Scratch--VmlldzoyNzIzNTQ1#using-fastai-to-train-your-diffusion-model
    """
    def __init__(self, n_steps:int=1000, s:float = 0.008):
        self.n_steps = n_steps
        self.s = s

        t = torch.arange(self.n_steps + 1)
        self.alpha_bar = torch.cos(( t / self.n_steps + self.s ) / ( 1 + self.s ) * torch.pi * 0.5 ) ** 2
        self.alpha_bar = self.alpha_bar / self.alpha_bar[0]
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
        if self.training:
            t = torch.randint(0, self.n_steps + 1, (batch_size,), dtype=torch.long) # select random timesteps
        else:
            # Use a spread of timesteps that is deterministic so validation results are comparable
            t = torch.linspace(0, self.n_steps, batch_size, dtype=torch.long)

        if dim == 2:
            alpha_bar_t = self.alpha_bar[t, None, None, None]
        else:
            alpha_bar_t = self.alpha_bar[t, None, None, None, None]
        alpha_bar_t = alpha_bar_t.to(self.dls.device)

        # noisify the image
        xt =  torch.sqrt(alpha_bar_t) * hr + torch.sqrt(1-alpha_bar_t) * noise 

        # Stack input with low-resolution image (upscaled) at channel dim,
        # then pass the stacked image along with the noise level as tuple to the model
        self.learn.xb = (torch.cat([xt, lr], dim=1), alpha_bar_t.sqrt().view((batch_size, 1)))
        self.learn.yb = (noise,) # we are trying to predict the noise


class DDPMSamplerCallback(DDPMCallback):
    def before_batch(self):
        lr = self.xb[0]
        batch_size = lr.shape[0]

        # Generate a batch of random noise to start with
        xt = torch.randn_like(lr)

        outputs = [xt]
        for t in track(reversed(range(self.n_steps + 1)), total=self.n_steps, description="Performing diffusion steps for batch:"):
            z = torch.randn(xt.shape, device=xt.device) if t > 0 else torch.zeros(xt.shape, device=xt.device)
            alpha_t = self.alpha[t] # get noise level at current timestep
            alpha_bar_t = self.alpha_bar[t]
            sigma_t = self.sigma[t]

            predicted_noise = self.model(torch.cat([xt, lr], dim=1), torch.full((batch_size, 1), alpha_bar_t.sqrt(),  device=xt.device))

            # predict x_(t-1) in accordance to Algorithm 2 in paper
            xt = 1/torch.sqrt(alpha_t) * (xt - (1-alpha_t)/torch.sqrt(1-alpha_bar_t) * predicted_noise)  + sigma_t*z
            outputs.append(xt)

        # self.learn.pred = (torch.stack(outputs, dim=1),)
        self.learn.pred = (xt,)

        raise CancelBatchException

