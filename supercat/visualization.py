from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import PIL
from PIL import Image
from plotly.subplots import make_subplots
import numpy as np

from .transforms import read3D


def format_fig(fig):
    fig.update_layout(
        plot_bgcolor="white",
        title_font_color="black",
        font=dict(
            family="Linux Libertine Display O",
            size=18,
            color="black",
        ),
    )


def render_volume(volume, width: int = 600, height: int = 600, title: str = "Volume") -> go.Figure:
    """
    Renders a volume with a number of cross-sections that can be animated.

    Based on https://plotly.com/python/visualizing-mri-volume-slices/ by Emilia Petrisor

    Args:
        volume (ndarray): The volume to be rendered.
        width (int, optional): The width of the figure. Defaults to 600.
        height (int, optional): The height of the figure. Defaults to 600.
        height (str, optional): The title of the figure. Defaults to 'Volume'.

    Returns:
        plotly.graph_objects.Figure: A plotly figure representing the volume.
    """
    r, c = volume[0].shape

    nb_frames = len(volume)

    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        z=k * np.ones((r, c)),
        surfacecolor=np.flipud(volume[k]),
        cmin=0, cmax=1.0
    ),
        # you need to name the frame for the animation to behave properly
        name=str(k)
    )
        for k in range(nb_frames)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=0.0 * np.ones((r, c)),
        surfacecolor=np.flipud(volume[0]),
        colorscale='Gray',
        cmin=0, cmax=1.0,
        colorbar=dict(thickness=20, ticklen=4)
    ))

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        scene=dict(
            zaxis=dict(range=[0, nb_frames-1], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders
    )

    return fig


    # upscaled_images = app(
    #     items=downscaled_images, 
    #     item_dir=[], 
    #     pretrained=str(pretrained),
    #     width=500, 
    #     height=500,
    #     return_images=True,
    # )

def comparison_plot(originals, downscaled_images, upscaled_images, titles, crops):
    fig = make_subplots(
        rows=len(originals), 
        cols=5,
        subplot_titles=(
            "Original", 
            "Cropped", 
            "Downscaled",
            "Upscaled",
            "Difference",
        ),
        vertical_spacing = 0.02,
        horizontal_spacing = 0.02,
    )
    
    for row, (original, downscaled, upscaled, title, crop) in enumerate(zip(originals, downscaled_images, upscaled_images, titles, crops)):
        original_im = Image.open(original)
        downscaled_im = Image.open(downscaled).resize( (original_im.size[0], original_im.size[1]), resample=PIL.Image.Resampling.NEAREST)

        crop_x = crop[0:2]
        crop_y = (crop[3], crop[2])

        if isinstance(upscaled, (Path, str)):
            upscaled = Image.open(upscaled)

        difference = np.asarray(upscaled).astype(int) - np.asarray(original_im.convert("RGB"))[:,:,0].astype(int)
        # squared_error = np.power(difference.astype(float)/255, 2.0)

        fig.add_trace( go.Image(z=np.asarray(original_im.convert("RGB"))), row=row+1, col=1)
        fig.add_trace( go.Image(z=np.asarray(original_im.convert("RGB"))), row=row+1, col=2)
        fig.add_trace( go.Image(z=np.asarray(downscaled_im.convert("RGB"))), row=row+1, col=3)
        fig.add_trace( go.Image(z=np.asarray(upscaled.convert("RGB")).astype(int)), row=row+1, col=4)
        fig.add_trace( go.Heatmap(z=difference.astype(float)/255, coloraxis="coloraxis"), row=row+1, col=5)

        update_dict = {
            f"yaxis{1+row*5}_title":title,
            f"xaxis{2+row*5}_range":(crop_x[0],crop_x[1]),
            f"yaxis{2+row*5}_range":(crop_y[0],crop_y[1]),
            f"xaxis{3+row*5}_range":(crop_x[0],crop_x[1]),
            f"yaxis{3+row*5}_range":(crop_y[0],crop_y[1]),
            f"xaxis{4+row*5}_range":(crop_x[0],crop_x[1]),
            f"yaxis{4+row*5}_range":(crop_y[0],crop_y[1]),
            f"xaxis{5+row*5}_range":(crop_x[0],crop_x[1]),
            f"yaxis{5+row*5}_range":(crop_y[0],crop_y[1]),
        }
        fig.update_layout(**update_dict)
        fig.add_shape(type="rect",
            x0=crop_x[0], y0=crop_y[0], x1=crop_x[1], y1=crop_y[1],
            line=dict(color="Red"),
            row=row+1, 
            col=1,
        )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        height=150 + 200 * len(originals),
        width=1200,
    )
    format_fig(fig)

    fig.update_layout(coloraxis=dict(colorscale='Rainbow'), showlegend=False)
    fig.update_annotations(font_size=24)

    return fig    


def add_volume_face_traces(fig, volume, coloraxis="coloraxis", **kwargs):
    """ Adds six faces of a volume to a plotly figure. """
    x1 = np.zeros(volume.shape[0], dtype=int)
    y1 = np.arange(volume.shape[1])
    z1 = np.arange(volume.shape[2])
    surfacecolor = volume[x1[0],:,:]
    fig.add_trace(
        go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)), surfacecolor=surfacecolor, text=surfacecolor, coloraxis=coloraxis, name="left"),
        **kwargs
    )

    # RIGHT x = max
    x1[:] = volume.shape[0] - 1
    surfacecolor = volume[x1[0],:,:]
    fig.add_trace(
        go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)), surfacecolor=surfacecolor, text=surfacecolor, coloraxis=coloraxis, name="right"),
        **kwargs
    )

    # BACK y = 0
    x1 = np.arange(volume.shape[0])
    y1 = np.zeros(volume.shape[1], dtype=int)
    surfacecolor = volume[:,y1[0],:].T
    fig.add_trace(
        go.Surface(x=x1, y=y1, z=np.array([z1] * len(y1)).T, surfacecolor=surfacecolor, text=surfacecolor, coloraxis=coloraxis, name="back"),
        **kwargs
    )

    # FRONT y = max
    y1[:] = volume.shape[1] - 1
    surfacecolor = volume[:,y1[0],:].T
    fig.add_trace(
        go.Surface(x=x1, y=y1, z=np.array([z1] * len(y1)).T, surfacecolor=surfacecolor, text=surfacecolor, coloraxis=coloraxis, name="front"),
        **kwargs
    )

    # BOTTOM z = 0
    x1 = np.arange(volume.shape[0])
    y1 = np.arange(volume.shape[1])
    z1 = np.zeros((volume.shape[0],volume.shape[1]), dtype=int)
    surfacecolor = volume[:,:,z1[0,0]].T
    fig.add_trace(
        go.Surface(x=x1, y=y1, z=z1, surfacecolor=surfacecolor, text=surfacecolor, coloraxis=coloraxis, name="bottom"),
        **kwargs
    )

    # TOP z = 0
    z1[:,:] = volume.shape[2] - 1
    surfacecolor = volume[:,:,z1[0,0]].T
    fig.add_trace(
        go.Surface(x=x1, y=y1, z=z1, surfacecolor=surfacecolor, text=surfacecolor, coloraxis=coloraxis, name="top"),
        **kwargs
    )
    return fig


def comparison_plot3D(originals, downscaled_volumes, upscaled_volumes, titles):
    fig = make_subplots(
        rows=len(originals), 
        cols=3,
        subplot_titles=(
            "Original", 
            "Downscaled",
            "Upscaled",
            # "Difference",
        ),
        vertical_spacing = 0.02,
        horizontal_spacing = 0.02,
        specs=[[{'type':"surface"}, {'type':"surface"}, {'type':"surface"}, ]]*len(originals), # hack
    )

    axis = dict(showgrid=False, showticklabels=False, showaxeslabels=False, title="", showbackground=False)
    scene = dict(
            xaxis=axis,
            yaxis=axis,
            zaxis=axis,
    )

    for row, (original, downscaled, upscaled, title) in enumerate(zip(originals, downscaled_volumes, upscaled_volumes, titles)):
        original = read3D(original) if isinstance(original, (str, Path)) else original
        downscaled = read3D(downscaled) if isinstance(downscaled, (str, Path)) else downscaled
        upscaled = read3D(upscaled) if isinstance(upscaled, (str, Path)) else upscaled

        # upscaled = (upscaled - upscaled.mean())/upscaled.std()
        # upscaled = upscaled * downscaled.std() + downscaled.mean()
        # breakpoint()
        # upscaled *= 255.0
        # breakpoint()

        add_volume_face_traces(fig, original, row=row+1, col=1)
        add_volume_face_traces(fig, downscaled, row=row+1, col=2)
        add_volume_face_traces(fig, upscaled, row=row+1, col=3)
        # add_volume_face_traces(fig, upscaled-original, row=row+1, col=4, coloraxis="coloraxis2")

        scenes = {f"scene{row*4+column}":scene for column in range(1,5)}
        fig.update_layout(**scenes)

        fig.add_annotation(
            text=title,
            xref="paper", 
            yref="paper",
            x=0.0, 
            y=1.0-1.0*row/len(originals)-0.5/len(originals), 
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            textangle=-90

        )



    fig.update_layout(coloraxis=dict(colorscale='gray', cmin=0.0, cmax=1.0, showscale=False), showlegend=False)
    fig.update_layout(coloraxis2=dict(colorscale='Rainbow'), showlegend=False)

    fig.update_layout(
        scene1=scene,
        scene2=scene,
        scene3=scene,
        scene4=scene,
    )
    fig.update_layout(
        height=250 + 200 * len(originals),
        width=1200,
    )
    format_fig(fig)
    return fig
