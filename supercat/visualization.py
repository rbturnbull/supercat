import numpy as np
import plotly.graph_objects as go


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
