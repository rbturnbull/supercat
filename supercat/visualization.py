from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import PIL
from PIL import Image
from plotly.subplots import make_subplots
import numpy as np

from .transforms import read3D


class DivergeColorGradient:
    """
    A class to generate a diverging color gradient for data visualization.

    Data will be presented by a gradient defined by RGB colors color_high, color_mid, and color_low.
    color_high represents the color of z_max, color_low represents the color of z_min, and color_mid represents the color of the data mid-point.
    The gradient will be divided into num_steps steps.
    """

    def __init__(
        self,
        color_high: list[float, float, float],
        color_mid: list[float, float, float],
        color_low: list[float, float, float],
        z_max: float,
        z_min: float,
        num_steps: int,
    ):
        """
        Initialize a DivergeColorGradient object with color information and data range.

        Args:
            color_high (list): RGB color representing the high end of the gradient.
            color_mid (list): RGB color representing the middle of the gradient.
            color_low (list): RGB color representing the low end of the gradient.
            z_max (float): Maximum data value for color_high.
            z_min (float): Minimum data value for color_low.
            num_steps (int): Number of steps in the gradient.
        """
        self.color_high = np.array(color_high, dtype=np.float16)
        self.color_low = np.array(color_low, dtype=np.float16)
        self.color_mid = np.array(color_mid, dtype=np.float16)
        self.num_steps = num_steps
        self.z_max = z_max
        self.z_min = z_min
        self.z_mid = (z_max + z_min) / 2

    def full_colorscale(self, color_high, color_mid, color_low, num_steps):
        """
        Creates a full colorscale from color_high to color_low with num_steps steps.

        Args:
            color_high (list): RGB color representing the high end of the gradient.
            color_mid (list): RGB color representing the middle of the gradient.
            color_low (list): RGB color representing the low end of the gradient.
            num_steps (int): Number of steps in the gradient.

        Returns:
            numpy.ndarray: A numpy array representing the full gradient.
        """
        # Create a gradient from start to mid
        gradient1 = np.linspace(color_high, color_mid, num=((num_steps // 2) + 1))

        # Create a gradient from mid to end
        gradient2 = np.linspace(color_mid, color_low, num=((num_steps // 2) + 1))

        # Combine the two gradients
        return np.vstack((gradient1, gradient2[1:, :]))

    def section_colorscale(self, z_high, z_low, num_steps):
        """
        Creates a sub-range colorscale from z_high to z_low with num_steps steps.

        Args:
            z_high (float): High end of the data range for the sub-range gradient.
            z_low (float): Low end of the data range for the sub-range gradient.
            num_steps (int): Number of steps in the gradient.

        Returns:
            numpy.ndarray: A numpy array representing the sub-range gradient.
        """
        z_high = z_high if z_high <= self.z_max else self.z_max
        z_low = z_low if z_low >= self.z_min else self.z_min

        if z_high > self.z_mid and z_low >= self.z_mid:
            high_scalar, low_scalar = self.z_max - self.z_mid
        elif z_high <= self.z_mid and z_low < self.z_mid:
            high_scalar, low_scalar = self.z_min - self.z_mid
        else:
            high_scalar = self.z_max - self.z_mid
            low_scalar = self.z_min - self.z_mid

            num_steps_high = np.int(
                num_steps * (z_high - self.z_mid) / (z_high - z_low)
            )
            num_steps_low = num_steps - num_steps_high

        high_normalizer = (self.color_high - self.color_mid) / high_scalar
        low_normalizer = (self.color_low - self.color_mid) / low_scalar
        new_color_high = self.color_mid + (z_high - self.z_mid) * high_normalizer
        new_color_low = self.color_mid + (z_low - self.z_mid) * low_normalizer

        if high_scalar != low_scalar:
            gradient1 = np.linspace(new_color_high, self.color_mid, num_steps_high)
            gradient2 = np.linspace(self.color_mid, new_color_low, num_steps_low)

            return np.vstack((gradient1, gradient2))
        else:
            return np.linspace(new_color_high, new_color_low, num_steps)

    def format_output(self, colorscale):
        """
        Format the gradient into a list with step values and corresponding RGB colors.

        Args:
            colorscale (numpy.ndarray): A numpy array representing the gradient.

        Returns:
            list: A list of step-value, RGB color pairs.
        """
        output = [
            [step, f"rgb({color[0]},{color[1]},{color[2]})"]
            for step, color in zip(np.linspace(0, 1, len(colorscale)), colorscale)
        ]
        return output


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

def comparison_plot(
    originals: list[str | Path],
    downscaled_images: list[str | Path],
    upscaled_images:list[str | Path | np.ndarray],
    titles: list[str],
    crops: list[tuple[tuple[int, int], tuple[int, int]]],
    ):
    """
    Args:
        originals:
            A list of paths to the original images.
        downscaled_images:
            A list of paths to the downsampled images.
        upscaled_images:
            A list of paths to the upscaled images.
        titles:
            A list of titles for the images.
        crops:
            A list of tuples of the cropping area of the images.
            The format is ((x, x_width), (y, y_height)).
    Returns:
        A plotly figure object of the comparison plot.
    """
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
    colorscale_generator = DivergeColorGradient(
        color_high=[210.0, 0.0, 5.0],
        color_mid=[255.0, 255.0, 255.0],
        color_low=[5.0, 48.0, 210.0],
        z_max=2.0,
        z_min=-2.0,
        num_steps=11,
    )

    data_z_max = 0.0
    data_z_min = 0.0
    for row, (original, downscaled, upscaled, title, crop) in enumerate(zip(originals, downscaled_images, upscaled_images, titles, crops)):
        original_im = Image.open(original)
        downscaled_im = Image.open(downscaled).resize( (original_im.size[0], original_im.size[1]), resample=PIL.Image.Resampling.NEAREST)

        crop_x = crop[0]
        crop_y = crop[1]
        crop_x_0, crop_x_1 = crop_x[0], crop_x[0] + crop_x[1]
        crop_y_0, crop_y_1 = crop_y[0], crop_y[0] + crop_y[1]

        if isinstance(upscaled, (Path, str)):
            upscaled = Image.open(upscaled)

        difference = np.asarray(upscaled).astype(int) - np.asarray(original_im.convert("RGB"))[:,:,0].astype(int)
        difference = difference.astype(float)/255
        data_z_max = max(data_z_max, difference.max())
        data_z_min = min(data_z_min, difference.min())
        # squared_error = np.power(difference.astype(float)/255, 2.0)

        fig.add_trace( go.Image(z=np.asarray(original_im.convert("RGB"))), row=row+1, col=1)
        fig.add_trace( go.Image(z=np.asarray(original_im.convert("RGB"))), row=row+1, col=2)
        fig.add_trace( go.Image(z=np.asarray(downscaled_im.convert("RGB"))), row=row+1, col=3)
        fig.add_trace( go.Image(z=np.asarray(upscaled.convert("RGB")).astype(int)), row=row+1, col=4)
        fig.add_trace(
            go.Heatmap(
                z=difference,
                coloraxis="coloraxis",
                zauto=False,
                zmax=0.5,
                zmin=-0.5,
            ),
            row=row+1,
            col=5,
        )

        update_dict = {
            f"yaxis{1+row*5}_title":title,
            f"xaxis{2+row*5}_range":(crop_x_0,crop_x_1),
            f"yaxis{2+row*5}_range":(crop_y_0,crop_y_1),
            f"xaxis{3+row*5}_range":(crop_x_0,crop_x_1),
            f"yaxis{3+row*5}_range":(crop_y_0,crop_y_1),
            f"xaxis{4+row*5}_range":(crop_x_0,crop_x_1),
            f"yaxis{4+row*5}_range":(crop_y_0,crop_y_1),
            f"xaxis{5+row*5}_range":(crop_x_0,crop_x_1),
            f"yaxis{5+row*5}_range":(crop_y_0,crop_y_1),
        }
        fig.update_layout(**update_dict)
        fig.add_shape(type="rect",
            x0=crop_x_0, y0=crop_y_0, x1=crop_x_1, y1=crop_y_1,
            line=dict(color="Red"),
            row=row+1, 
            col=1,
        )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgb(170,170,170)')
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        height=150 + 200 * len(originals),
        width=1200,
    )
    format_fig(fig)
    custom_colorscale = colorscale_generator.section_colorscale(z_high=data_z_max, z_low=data_z_min, num_steps=11)
    fig.update_layout(coloraxis=dict(colorscale=colorscale_generator.format_output(custom_colorscale)), showlegend=False)
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


def comparison_plot3D(
    originals: list[str | Path | np.ndarray],
    downscaled_volumes: list[str | Path | np.ndarray],
    upscaled_volumes: list[str | Path | np.ndarray],
    titles: list[str],
    ):
    """
    Args:
        originals:
            A list of original images. The elements can be either paths to the images or the images themselves.
        downscaled_volumes:
            A list of downsampled images. The elements can be either paths to the images or the images themselves.
        upscaled_volumes:
            A list of upscaled images. The elements can be either paths to the images or the images themselves.
        titles:
            A list of titles for the images.
    Returns:
        A plotly figure object of the comparison plot.
    """
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


def visualize_result(
    dim: int,
    num_image: int,
    deeprock_path: str,
    inference_path: str,
    output_path: str,
    sources: list[str],
    split_type: str,
    downsample_method: str,
    downsample_scale: int,
    image_crops: tuple[tuple[int, int], tuple[int, int]],
) -> dict:
    """
    Args:
        dim:
            The dimension of the input images
        num_image:
            Number of images to be selected from each source for the visualization.
        deeprock_path:
            The path of the deeprock images.  This directory holds the original images and the downsampled images.
        inference_path:
            The path of the inference images.  This directory holds the upscaled images.
        output_path:
            The path to save the the output images.
        sources:
            A list of image sourcess/types.
        split_type:
            The split type of the images.  This will either 'train', 'valid' or 'test'.
        downsample_method:
            The method used to downsample the images.  This will either 'default' or 'unknown'.
        downsample_scale:
            The scale of the downsampled and upscaled the images. This will either 2 or 4.
        image_crops:
            The croping area of the image. used only for 2D images.  The format is ((x, x_width), (y, y_height))

    Returns:
        A dictionary of the images' paths.
        The structure of the dictionary is as follows:
        {
            "source1": {
                "original": [path1, path2, ...],
                "downscale": [path1, path2, ...],
                "upscale": [path1, path2, ...],
                "title": [title1, title2, ...],
            },
            "source2": {...}
        }

        Result images will be saved at output_path with the name {source}-compare.png
    """
    file_extension = ".png" if dim == 2 else ".mat"
    sources = [f"{src}{dim}D" for src in sources]

    deeprock_path = Path(deeprock_path)
    inference_path = Path(inference_path)
    output_path = Path(output_path)

    images = dict()
    for source in sources:
        upscale_path = inference_path/source
        upscale_images = sorted(upscale_path.glob("*" + file_extension))
        upscale_images = list(upscale_images)
        if len(upscale_images) == 0:
            raise Exception(f"No images found in {upscale_path}.  Please check the path.")
        try:
            upscale_images = upscale_images[:num_image]
        except Exception:
            raise IndexError(f"Parameter num_image exceed the number of images in {upscale_path}. Only {len(upscale_images)} images found.")

        original_path = deeprock_path/source/f"{source}_{split_type}_HR"
        original_images = [
            original_path/f"{upscale_image.name.split(f'x{downsample_scale}', 1)[0]}{file_extension}"
            for upscale_image in upscale_images
        ]

        downscale_path = deeprock_path/source/f"{source}_{split_type}_LR_{downsample_method}_X{downsample_scale}"
        downscale_images = [
            downscale_path/f"{upscale_image.name.split('.', 1)[0]}{file_extension}"
            for upscale_image in upscale_images
        ]

        title = [
            f"{source}/{original_image.name.split('.', 1)[0]}"
            for original_image in original_images
        ]
        images[source[:-2]] = {
            "original": original_images,
            "downscale": downscale_images,
            "upscale": upscale_images,
            "title": title,
        }

        output_path.mkdir(exist_ok=True, parents=True)
        if dim == 2:
            comparison_plot(
                original_images,
                downscale_images,
                upscale_images,
                title,
                image_crops,
            ).write_image(output_path/f"{source}-compare.png")
        elif dim == 3:
            comparison_plot3D(
                original_images, downscale_images, upscale_images, title
            ).write_image(output_path/f"{source}-compare.png")

    return images
