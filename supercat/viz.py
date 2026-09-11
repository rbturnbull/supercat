from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots
import numpy as np
import typer

app = typer.Typer()

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


@app.command()
def comparison(
    hr:list[Path] = typer.Option(..., help="Paths to the original high-resolution images"),
    lr:list[Path] = typer.Option(..., help="Paths to the downscaled low-resolution images"),
    sr:list[Path] = typer.Option(..., help="Paths to the upscaled super-resolution images"),
    titles:list[str] = typer.Option(..., help="Titles for each row, in the same order as the images"),
    output:Path = typer.Option(None, help="Path to save the resulting figure (e.g., comparison.html)"),
):
    assert len(hr) == len(lr) == len(sr) == len(titles), "The number of original, downscaled, upscaled images and titles must be the same"

    fig = make_subplots(
        rows=len(hr), 
        cols=4,
        subplot_titles=(
            "Original", 
            "Downscaled",
            "Upscaled",
            "Difference",
        ),
        vertical_spacing = 0.02,
        horizontal_spacing = 0.02,
    )

    def read(x):
        if isinstance(x, (Path, str)):
            x = str(x)
            if x.endswith(".mat"):
                x_im = read3D(x)
                x_im = x_im[x_im.shape[0]//2]
            else:
                x_im = np.asarray(Image.open(x).convert("L")).astype(int)
        return x_im
    
    
    for row, (original, downscaled, upscaled, title) in enumerate(zip(hr, lr, sr, titles)):
        original_im = read(original)
        downscaled_im = read(downscaled) #.resize( (original_im.size[0], original_im.size[1]), resample=PIL.Image.Resampling.NEAREST)
        upscaled = read(upscaled)

        difference = upscaled - original_im
        # squared_error = np.power(difference.astype(float)/255, 2.0)

        fig.add_trace( go.Heatmap(z=np.asarray(original_im).astype(int), colorscale="gray", showscale=False, zmin=0, zmax=255), row=row+1, col=1)
        fig.add_trace( go.Heatmap(z=np.asarray(downscaled_im).astype(int), colorscale="gray", showscale=False, zmin=0, zmax=255), row=row+1, col=2)
        fig.add_trace( go.Heatmap(z=np.asarray(upscaled).astype(int), colorscale="gray", showscale=False, zmin=0, zmax=255), row=row+1, col=3)
        fig.add_trace( go.Heatmap(z=difference, coloraxis="coloraxis2"), row=row+1, col=4)

        update_dict = {
            f"yaxis{1+row*4}_title":title,
        }
        fig.update_layout(**update_dict)
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        height=150 + 240 * len(hr),
        width=1200,
    )
    format_fig(fig)

    # fig.update_layout(coloraxis1=dict(colorscale='gray'), showlegend=False)
    fig.update_layout(coloraxis2=dict(colorscale='Rainbow'), showlegend=False)
    fig.update_annotations(font_size=24)

    if output is not None:
        print(f"Saving figure to {output}")
        if output.suffix.lower() in [".html", ".htm"]:
            fig.write_html(output)
        else:
            fig.write_image(output, scale=2)

    return fig    

if __name__ == "__main__":
    app()