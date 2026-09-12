from unittest.mock import Mock

import hdf5storage
import numpy as np
import plotly.graph_objects as go
import pytest
from PIL import Image
from typer.testing import CliRunner

from supercat import viz


@pytest.fixture
def image_paths(tmp_path):
    arrays = {
        "hr": np.array([[0, 200, 255], [100, 50, 20]], dtype=np.uint8),
        "lr": np.array([[10, 100, 150], [80, 40, 30]], dtype=np.uint8),
        "sr": np.array([[20, 150, 250], [110, 50, 0]], dtype=np.uint8),
    }
    paths = {}
    for name, values in arrays.items():
        paths[name] = tmp_path / f"{name}.png"
        Image.fromarray(values).save(paths[name])
    return paths, arrays


def comparison_options(paths, **overrides):
    options = {name: [path] for name, path in paths.items()}
    options.update(titles=["Sandstone"], output=None)
    options.update(overrides)
    return options


def test_format_fig_sets_theme_without_removing_content():
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    fig.update_layout(title="Existing title", width=700)
    viz.format_fig(fig)
    assert fig.layout.plot_bgcolor == "white"
    assert fig.layout.title.font.color == "black"
    assert fig.layout.font.family == "Linux Libertine Display O"
    assert fig.layout.font.size == 18
    assert fig.layout.font.color == "black"
    assert fig.layout.title.text == "Existing title"
    assert fig.layout.width == 700
    assert list(fig.data[0].y) == [3, 4]


@pytest.mark.parametrize("string_paths", [False, True])
def test_comparison_contains_images_and_signed_difference(image_paths, string_paths):
    paths, arrays = image_paths
    if string_paths:
        paths = {name: str(path) for name, path in paths.items()}
    fig = viz.comparison(**comparison_options(paths))
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4
    for trace, name in zip(fig.data[:3], ["hr", "lr", "sr"]):
        assert isinstance(trace, go.Heatmap)
        np.testing.assert_array_equal(trace.z, arrays[name])
        assert trace.zmin == 0
        assert trace.zmax == 255
        assert trace.showscale is False
    expected_difference = np.array([[20, -50, -5], [10, 0, -20]])
    np.testing.assert_array_equal(fig.data[3].z, expected_difference)
    assert fig.data[3].coloraxis == "coloraxis2"


def test_comparison_layout_labels_and_axes(image_paths):
    paths, _ = image_paths
    fig = viz.comparison(**comparison_options(paths))
    assert [annotation.text for annotation in fig.layout.annotations] == [
        "Original",
        "Downscaled",
        "Upscaled",
        "Difference",
    ]
    assert all(annotation.font.size == 24 for annotation in fig.layout.annotations)
    assert fig.layout.yaxis.title.text == "Sandstone"
    assert fig.layout.height == 390
    assert fig.layout.width == 1200
    assert fig.layout.showlegend is False
    for axis in fig.select_xaxes():
        assert axis.showticklabels is False
    for axis in fig.select_yaxes():
        assert axis.showticklabels is False
    assert fig.layout.plot_bgcolor == "white"


def test_comparison_multiple_rows_preserve_order(image_paths):
    paths, arrays = image_paths
    fig = viz.comparison(
        hr=[paths["hr"], paths["sr"]],
        lr=[paths["lr"], paths["lr"]],
        sr=[paths["sr"], paths["hr"]],
        titles=["Sandstone", "Carbonate"],
        output=None,
    )
    assert len(fig.data) == 8
    assert fig.layout.height == 630
    assert fig.layout.yaxis.title.text == "Sandstone"
    assert fig.layout.yaxis5.title.text == "Carbonate"
    np.testing.assert_array_equal(fig.data[0].z, arrays["hr"])
    np.testing.assert_array_equal(fig.data[4].z, arrays["sr"])
    np.testing.assert_array_equal(fig.data[7].z, -np.asarray(fig.data[3].z))
    assert fig.data[4].xaxis == "x5"
    assert fig.data[7].xaxis == "x8"


def test_comparison_converts_rgb_to_grayscale(tmp_path):
    path = tmp_path / "rgb.png"
    Image.new("RGB", (3, 2), color=(255, 0, 0)).save(path)
    fig = viz.comparison(hr=[path], lr=[path], sr=[path], titles=["RGB"], output=None)
    np.testing.assert_array_equal(fig.data[0].z, np.full((2, 3), 76))
    np.testing.assert_array_equal(fig.data[3].z, np.zeros((2, 3)))


def test_comparison_accepts_smaller_low_resolution_image(image_paths, tmp_path):
    paths, _ = image_paths
    path = tmp_path / "small.png"
    Image.new("L", (1, 1), color=100).save(path)
    fig = viz.comparison(**comparison_options(paths, lr=[path]))
    assert np.asarray(fig.data[1].z).shape == (1, 1)
    assert np.asarray(fig.data[3].z).shape == (2, 3)


@pytest.mark.parametrize("field", ["hr", "lr", "sr", "titles"])
def test_comparison_rejects_mismatched_row_counts(image_paths, field):
    paths, _ = image_paths
    with pytest.raises(AssertionError, match="must be the same"):
        viz.comparison(**comparison_options(paths, **{field: []}))


@pytest.mark.parametrize("suffix", [".html", ".htm", ".HTML"])
def test_comparison_writes_html_without_static_renderer(
    image_paths, tmp_path, suffix, monkeypatch, capsys
):
    paths, _ = image_paths
    output = tmp_path / f"comparison{suffix}"
    static_export = Mock()
    monkeypatch.setattr(go.Figure, "write_image", static_export)
    viz.comparison(**comparison_options(paths, output=output))
    html = output.read_text()
    assert "Plotly.newPlot" in html
    assert "Sandstone" in html
    static_export.assert_not_called()
    assert f"Saving figure to {output}" in capsys.readouterr().out


@pytest.mark.parametrize("suffix", [".png", ".pdf", ".SVG"])
def test_comparison_requests_static_export_at_double_scale(
    image_paths, tmp_path, suffix, monkeypatch
):
    paths, _ = image_paths
    output = tmp_path / f"comparison{suffix}"
    export = Mock()
    monkeypatch.setattr(go.Figure, "write_image", export)
    viz.comparison(**comparison_options(paths, output=output))
    export.assert_called_once_with(output, scale=2)


def test_comparison_without_output_does_not_export(image_paths, monkeypatch):
    paths, _ = image_paths
    html, static = Mock(), Mock()
    monkeypatch.setattr(go.Figure, "write_html", html)
    monkeypatch.setattr(go.Figure, "write_image", static)
    viz.comparison(**comparison_options(paths))
    html.assert_not_called()
    static.assert_not_called()


def test_comparison_cli_writes_html(image_paths, tmp_path):
    paths, _ = image_paths
    output = tmp_path / "cli.html"
    result = CliRunner().invoke(
        viz.app,
        [
            "--hr",
            str(paths["hr"]),
            "--lr",
            str(paths["lr"]),
            "--sr",
            str(paths["sr"]),
            "--titles",
            "Sandstone",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Plotly.newPlot" in output.read_text()


def test_comparison_uses_middle_slice_of_mat_volume(tmp_path):
    path = tmp_path / "volume.mat"
    volume = np.arange(60, dtype=np.uint8).reshape(3, 4, 5)
    hdf5storage.savemat(str(path), {"temp": volume}, format="7.3")
    fig = viz.comparison(
        hr=[path], lr=[path], sr=[path], titles=["Volume"], output=None
    )
    np.testing.assert_array_equal(fig.data[0].z, volume[1])
    np.testing.assert_array_equal(fig.data[3].z, np.zeros((4, 5)))


def test_module_entry_point_runs_the_cli_app(monkeypatch):
    import runpy
    import typer

    launched = Mock()
    monkeypatch.setattr(
        typer.Typer, "__call__", lambda self, *args, **kwargs: launched(self)
    )
    namespace = runpy.run_path(viz.__file__, run_name="__main__")
    launched.assert_called_once_with(namespace["app"])


def test_comparison_accepts_arrays_without_reading_files():
    array = np.array([[0, 100], [200, 255]])
    fig = viz.comparison(
        hr=[array], lr=[array], sr=[array + 1], titles=["Array"], output=None
    )
    np.testing.assert_array_equal(fig.data[0].z, array)
    np.testing.assert_array_equal(fig.data[3].z, np.ones((2, 2)))
