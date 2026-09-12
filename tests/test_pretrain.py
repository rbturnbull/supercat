from pathlib import Path

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch
from PIL import Image

from supercat import pretrain

from supercat.pretrain import PretrainMovieDataset, PretrainImagesDataset

K400_PATH = Path(__file__).parent / "k400"
IMAGENET_PATH = Path(__file__).parent / "imagenet"


def test_pretrain_movie_dataset():
    dataset = PretrainMovieDataset(K400_PATH, size=8)
    assert len(dataset) == 2, "Dataset should not be empty"
    for item in dataset:
        lr_t, hr_t = item

        assert lr_t.shape == hr_t.shape == (1, 8, 8, 8)
        assert torch.isfinite(lr_t).all()
        assert torch.isfinite(hr_t).all()
        assert hr_t.min() >= -1.0
        assert hr_t.max() <= 1.0


def test_pretrain_image_dataset():
    dataset = PretrainImagesDataset(IMAGENET_PATH, size=8)
    assert len(dataset) == 2, "Dataset should not be empty"
    for item in dataset:
        lr_t, hr_t = item

        assert lr_t.shape == hr_t.shape == (1, 8, 8)
        assert torch.isfinite(lr_t).all()
        assert torch.isfinite(hr_t).all()
        assert hr_t.min() >= -1.0
        assert hr_t.max() <= 1.0


@pytest.mark.parametrize("channels", [None, 1, 3, 4])
def test_normalize_video_frame(channels):
    if channels is None:
        frame = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        expected = frame.copy()
    elif channels == 1:
        frame = np.array([[[1], [2]], [[3], [4]]], dtype=np.uint8)
        expected = frame[..., 0].copy()
    else:
        pixel = [10, 20, 33] + ([255] if channels == 4 else [])
        frame = np.tile(np.array(pixel, dtype=np.uint8), (2, 3, 1))
        expected = np.full((2, 3), 21, dtype=np.uint8)
    original = frame.copy()
    result = pretrain._normalize_video_frame(frame)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == frame.dtype
    np.testing.assert_array_equal(frame, original)


@pytest.mark.parametrize("shape", [(3,), (1, 2, 3, 4)])
def test_normalize_video_frame_rejects_invalid_dimensions(shape):
    with pytest.raises(ValueError, match="Unsupported frame shape"):
        pretrain._normalize_video_frame(np.zeros(shape))


def test_ffmpeg_binary_uses_path_lookup(monkeypatch):
    which = Mock(return_value="/mock/ffmpeg")
    monkeypatch.setattr(pretrain.shutil, "which", which)
    assert pretrain._ffmpeg_binary() == "/mock/ffmpeg"
    which.assert_called_once_with("ffmpeg")


@pytest.fixture
def ffmpeg_mock(monkeypatch):
    pretrain._read_video_via_ffmpeg.cache_clear()
    monkeypatch.setattr(pretrain, "_ffmpeg_binary", lambda: "/mock/ffmpeg")
    run = Mock()
    monkeypatch.setattr(pretrain.subprocess, "run", run)
    yield run
    pretrain._read_video_via_ffmpeg.cache_clear()


def test_probe_ffmpeg_extracts_final_frame_count(ffmpeg_mock, tmp_path):
    path = tmp_path / "movie.mp4"
    ffmpeg_mock.return_value = SimpleNamespace(
        stdout="frame= 1\n",
        stderr="Video: h264, yuv420p, 640x480 [SAR 1:1]\nframe= 25\n",
    )
    assert pretrain._probe_ffmpeg_video(path) == (25, 480, 640)
    ffmpeg_mock.assert_called_once_with(
        [
            "/mock/ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "output,message",
    [
        ("frame= 10", "Could not determine video size"),
        ("Video: h264, yuv420p, 640x480 [SAR 1:1]", "Could not determine frame count"),
    ],
)
def test_probe_ffmpeg_rejects_incomplete_metadata(
    ffmpeg_mock, tmp_path, output, message
):
    ffmpeg_mock.return_value = SimpleNamespace(stdout="", stderr=output)
    with pytest.raises(RuntimeError, match=message):
        pretrain._probe_ffmpeg_video(tmp_path / "movie.mp4")


def test_probe_ffmpeg_requires_binary(ffmpeg_mock, monkeypatch, tmp_path):
    monkeypatch.setattr(pretrain, "_ffmpeg_binary", lambda: None)
    with pytest.raises(RuntimeError, match="ffmpeg binary not found"):
        pretrain._probe_ffmpeg_video(tmp_path / "movie.mp4")
    ffmpeg_mock.assert_not_called()


@pytest.mark.parametrize("reported_count", [2, 9])
def test_read_ffmpeg_decodes_actual_frames_and_caches(
    ffmpeg_mock, monkeypatch, tmp_path, reported_count
):
    probe = Mock(return_value=(reported_count, 3, 4))
    monkeypatch.setattr(pretrain, "_probe_ffmpeg_video", probe)
    frames = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    ffmpeg_mock.return_value = SimpleNamespace(stdout=frames.tobytes())
    path = str(tmp_path / "movie.mp4")
    result = pretrain._read_video_via_ffmpeg(path)
    np.testing.assert_array_equal(result, frames)
    assert pretrain._read_video_via_ffmpeg(path) is result
    ffmpeg_mock.assert_called_once()
    probe.assert_called_once_with(Path(path))
    command = ffmpeg_mock.call_args.args[0]
    assert command[0] == "/mock/ffmpeg"
    assert command[command.index("-pix_fmt") + 1] == "gray"
    assert ffmpeg_mock.call_args.kwargs == {"capture_output": True, "check": True}


def test_read_ffmpeg_rejects_partial_frame(ffmpeg_mock, monkeypatch, tmp_path):
    monkeypatch.setattr(pretrain, "_probe_ffmpeg_video", lambda path: (2, 3, 4))
    ffmpeg_mock.return_value = SimpleNamespace(stdout=b"12345")
    with pytest.raises(RuntimeError, match="Unexpected ffmpeg output size"):
        pretrain._read_video_via_ffmpeg(str(tmp_path / "movie.mp4"))


def test_read_ffmpeg_requires_binary_after_probe(ffmpeg_mock, monkeypatch, tmp_path):
    monkeypatch.setattr(pretrain, "_probe_ffmpeg_video", lambda path: (2, 3, 4))
    monkeypatch.setattr(pretrain, "_ffmpeg_binary", lambda: None)
    with pytest.raises(RuntimeError, match="ffmpeg binary not found"):
        pretrain._read_video_via_ffmpeg(str(tmp_path / "movie.mp4"))
    ffmpeg_mock.assert_not_called()


@pytest.fixture
def fake_backends(monkeypatch):
    # Substitute optional modules so no installed codec or external process is needed.
    imageio = ModuleType("imageio")
    iio = ModuleType("imageio.v3")
    imageio.v3 = iio
    iio.improps = Mock()
    iio.imiter = Mock()
    skvideo = ModuleType("skvideo")
    skio = ModuleType("skvideo.io")
    skvideo.io = skio
    skio.ffprobe = Mock()
    skio.vreader = Mock(
        side_effect=lambda *args, **kwargs: iter([np.zeros((1, 2, 3, 1))])
    )
    for name, module in [
        ("imageio", imageio),
        ("imageio.v3", iio),
        ("skvideo", skvideo),
        ("skvideo.io", skio),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(pretrain, "_ffmpeg_binary", lambda: None)
    return iio, skio


def test_video_backend_prefers_imageio(fake_backends, tmp_path):
    iio, skio = fake_backends
    path = tmp_path / "movie.mp4"
    assert pretrain._video_backend(path) == ("imageio", iio)
    iio.improps.assert_called_once_with(path)
    skio.vreader.assert_not_called()


@pytest.mark.parametrize("missing_imageio", [False, True])
def test_video_backend_falls_back_to_skvideo(
    fake_backends, monkeypatch, tmp_path, missing_imageio
):
    iio, skio = fake_backends
    if missing_imageio:
        monkeypatch.setitem(sys.modules, "imageio.v3", None)
    else:
        iio.improps.side_effect = OSError("unsupported")
    assert pretrain._video_backend(tmp_path / "movie.mp4") == (
        "skvideo",
        (skio.ffprobe, skio.vreader),
    )


@pytest.mark.parametrize(
    "failure", [AssertionError, OSError, StopIteration, ImportError]
)
def test_video_backend_falls_back_to_ffmpeg(
    fake_backends, monkeypatch, tmp_path, failure
):
    iio, skio = fake_backends
    iio.improps.side_effect = OSError("unsupported")
    if failure is ImportError:
        monkeypatch.setitem(sys.modules, "skvideo.io", None)
    else:
        skio.vreader.side_effect = failure
    monkeypatch.setattr(pretrain, "_ffmpeg_binary", lambda: "/mock/ffmpeg")
    assert pretrain._video_backend(tmp_path / "movie.mp4") == ("ffmpeg_cli", None)


def test_video_backend_reports_no_decoder(fake_backends, tmp_path):
    iio, skio = fake_backends
    iio.improps.side_effect = OSError("unsupported")
    skio.vreader.side_effect = OSError("unsupported")
    with pytest.raises(RuntimeError, match="No supported video backend"):
        pretrain._video_backend(tmp_path / "movie.mp4")


@pytest.mark.parametrize("backend", ["imageio", "skvideo", "ffmpeg_cli"])
def test_iter_video_frames_yields_grayscale_frames(monkeypatch, tmp_path, backend):
    frames = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    if backend == "imageio":
        reader = SimpleNamespace(imiter=lambda path: iter(frames[..., None]))
    elif backend == "skvideo":
        reader = (
            Mock(),
            lambda *args, **kwargs: (frame[None, ..., None] for frame in frames),
        )
    else:
        reader = None
        monkeypatch.setattr(
            pretrain, "_read_video_via_ffmpeg", Mock(return_value=frames)
        )
    monkeypatch.setattr(pretrain, "_video_backend", lambda path: (backend, reader))
    actual = list(pretrain.iter_video_frames(tmp_path / "movie.mp4"))
    np.testing.assert_array_equal(np.stack(actual), frames)


@pytest.mark.parametrize("count", [2, float("inf")])
def test_video_shape_imageio_counts_unknown_length(monkeypatch, tmp_path, count):
    reader = SimpleNamespace(
        improps=Mock(return_value=SimpleNamespace(shape=(count, 3, 4, 3))),
        imiter=Mock(return_value=iter([None, None])),
    )
    monkeypatch.setattr(pretrain, "_video_backend", lambda path: ("imageio", reader))
    assert pretrain.video_shape(tmp_path / "movie.mp4") == (2, 3, 4)
    assert reader.imiter.call_count == (0 if count == 2 else 1)


def test_video_shape_rejects_incomplete_imageio_shape(monkeypatch, tmp_path):
    reader = SimpleNamespace(improps=lambda path: SimpleNamespace(shape=(3, 4)))
    monkeypatch.setattr(pretrain, "_video_backend", lambda path: ("imageio", reader))
    with pytest.raises(ValueError, match="Unsupported video shape"):
        pretrain.video_shape(tmp_path / "movie.mp4")


@pytest.mark.parametrize("count", ["2", "N/A", None])
def test_video_shape_skvideo_metadata_and_frame_count_fallback(
    monkeypatch, tmp_path, count
):
    metadata = {"@height": "3", "@width": "4"}
    if count is not None:
        metadata["@nb_frames"] = count
    probe = Mock(return_value={"video": metadata})
    monkeypatch.setattr(
        pretrain, "_video_backend", lambda path: ("skvideo", (probe, Mock()))
    )
    frames = Mock(return_value=iter([None, None]))
    monkeypatch.setattr(pretrain, "iter_video_frames", frames)
    assert pretrain.video_shape(tmp_path / "movie.mp4") == (2, 3, 4)
    assert frames.call_count == (0 if count == "2" else 1)


def test_video_shape_ffmpeg_uses_decoded_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(pretrain, "_video_backend", lambda path: ("ffmpeg_cli", None))
    monkeypatch.setattr(
        pretrain, "_read_video_via_ffmpeg", lambda path: np.zeros((2, 3, 4))
    )
    assert pretrain.video_shape(tmp_path / "movie.mp4") == (2, 3, 4)


def test_reflect_padding_repeats_safely_for_large_target():
    image = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    result = pretrain._pad_to_size_with_reflect(image, 5)
    expected = torch.tensor(
        [
            [
                [1.0, 2.0, 1.0, 2.0, 1.0],
                [3.0, 4.0, 3.0, 4.0, 3.0],
                [1.0, 2.0, 1.0, 2.0, 1.0],
                [3.0, 4.0, 3.0, 4.0, 3.0],
                [1.0, 2.0, 1.0, 2.0, 1.0],
            ]
        ]
    )
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(image, expected[:, :2, :2])


@pytest.mark.parametrize("dim", [2, 3])
def test_padding_singleton_dimensions_replicates(dim):
    image = torch.full((1,) + (1,) * dim, 7.0)
    result = pretrain._pad_to_size_with_reflect(image, 4, spatial_dims=dim)
    torch.testing.assert_close(result, torch.full((1,) + (4,) * dim, 7.0))


def test_padding_preserves_larger_dimensions():
    image = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    assert pretrain._pad_to_size_with_reflect(image, 3) is image


@pytest.mark.parametrize(
    "shape,dim,message",
    [((1, 2, 3), 1, "spatial_dims must be"), ((2,), 2, "image has")],
)
def test_padding_rejects_invalid_dimensions(shape, dim, message):
    with pytest.raises(ValueError, match=message):
        pretrain._pad_to_size_with_reflect(torch.zeros(shape), 4, dim)


@pytest.mark.parametrize(
    "finder,suffixes",
    [
        (pretrain.find_images, [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]),
        (pretrain.find_movies, [".mp4", ".avi", ".mov", ".mkv"]),
    ],
)
def test_file_discovery_recurses_and_filters_extensions(tmp_path, finder, suffixes):
    nested = tmp_path / "nested"
    nested.mkdir()
    for index, suffix in enumerate(suffixes):
        (nested / f"sample{index}{suffix}").touch()
    (nested / f"uppercase{suffixes[0].upper()}").touch()
    (nested / "ignored.txt").touch()
    found = finder(tmp_path)
    assert {path.name for path in found} == {
        *(f"sample{index}{suffix}" for index, suffix in enumerate(suffixes)),
        f"uppercase{suffixes[0].upper()}",
    }
    assert len(found) == len(suffixes) + 1


def test_find_files_requires_directory(tmp_path):
    with pytest.raises(ValueError, match="not a valid directory"):
        pretrain.find_files(tmp_path / "missing", [".png"])


@pytest.mark.parametrize("limit", [0, 1, 3])
def test_movie_discovery_respects_limit(tmp_path, limit):
    for name in ["a.mp4", "b.mp4", "c.avi"]:
        (tmp_path / name).touch()
    found = pretrain.find_movies(tmp_path, max_items=limit)
    assert len(found) == limit
    assert all(path.is_file() for path in found)


@pytest.fixture
def synthetic_image(tmp_path):
    pixels = np.arange(9 * 11, dtype=np.uint8).reshape(9, 11)
    Image.fromarray(pixels).save(tmp_path / "image.png")
    return tmp_path, torch.tensor(pixels.astype(np.float32) * 2 / 255 - 1)[None]


def test_image_dataset_center_crops(synthetic_image):
    path, image = synthetic_image
    lr, hr = PretrainImagesDataset(path, scale=2, max_size=4)[0]
    torch.testing.assert_close(hr, image[:, 2:6, 3:7])
    assert lr.shape == hr.shape == (1, 4, 4)


def test_image_dataset_trims_odd_dimensions(synthetic_image):
    path, image = synthetic_image
    lr, hr = PretrainImagesDataset(path, scale=2)[0]
    torch.testing.assert_close(hr, image[:, :8, :10])
    assert lr.shape == hr.shape


def test_image_dataset_random_crop_and_paired_augmentation(
    synthetic_image, monkeypatch
):
    path, image = synthetic_image
    randint = Mock(side_effect=[0, 0, 4])
    monkeypatch.setattr(pretrain.np.random, "randint", randint)
    lr, hr = PretrainImagesDataset(path, scale=1, size=4, augment=True)[0]
    torch.testing.assert_close(hr, image[:, :4, :4].flip(-1))
    torch.testing.assert_close(lr, hr)


@pytest.fixture
def synthetic_movie(tmp_path, monkeypatch):
    (tmp_path / "movie.mp4").touch()
    pixels = np.arange(7 * 9 * 11, dtype=np.uint16).reshape(7, 9, 11).astype(np.uint8)
    monkeypatch.setattr(pretrain, "video_shape", lambda path: pixels.shape)
    monkeypatch.setattr(pretrain, "iter_video_frames", lambda path: iter(pixels))
    return tmp_path, torch.tensor(pixels.astype(np.float32) * 2 / 255 - 1)[None]


def test_movie_dataset_center_crops_frames_and_space(synthetic_movie):
    path, image = synthetic_movie
    lr, hr = PretrainMovieDataset(path, scale=2, max_size=4)[0]
    torch.testing.assert_close(hr, image[:, 1:5, 2:6, 3:7])
    assert lr.shape == hr.shape == (1, 4, 4, 4)


def test_movie_dataset_trims_odd_dimensions_and_moves_channel(synthetic_movie):
    path, image = synthetic_movie
    lr, hr = PretrainMovieDataset(path, scale=2, channel_first=False)[0]
    torch.testing.assert_close(hr, image[:, :6, :8, :10].movedim(0, -1))
    assert lr.shape == hr.shape == (6, 8, 10, 1)


def test_movie_dataset_random_crop_and_paired_augmentation(
    synthetic_movie, monkeypatch
):
    path, image = synthetic_movie
    monkeypatch.setattr(pretrain.random, "randint", lambda *args: 0)
    monkeypatch.setattr(pretrain.np.random, "randint", lambda *args: 4)
    lr, hr = PretrainMovieDataset(path, scale=1, size=4, augment=True)[0]
    torch.testing.assert_close(hr, image[:, :4, :4, :4].flip(-1))
    torch.testing.assert_close(lr, hr)


@pytest.mark.parametrize("min_size", [0, 4])
def test_movie_dataset_returns_zeros_for_decode_failure(
    synthetic_movie, monkeypatch, capsys, min_size
):
    path, _ = synthetic_movie
    monkeypatch.setattr(
        pretrain, "video_shape", Mock(side_effect=OSError("decode failed"))
    )
    lr, hr = PretrainMovieDataset(path, min_size=min_size)[0]
    expected = torch.zeros((1,) + (min_size or 16,) * 3)
    torch.testing.assert_close(lr, expected)
    torch.testing.assert_close(hr, expected)
    assert "decode failed" in capsys.readouterr().out


@pytest.mark.parametrize("dataset_class", [PretrainImagesDataset, PretrainMovieDataset])
def test_dataset_requires_path(dataset_class):
    with pytest.raises(AssertionError, match="Path must be provided"):
        dataset_class(None)


@pytest.mark.parametrize(
    "app_class,dataset_name,max_size",
    [
        (pretrain.SupercatPretrainImage, "PretrainImagesDataset", 224),
        (pretrain.SupercatPretrainMovie, "PretrainMovieDataset", 100),
    ],
)
def test_pretrain_app_dataset_defaults(
    monkeypatch, tmp_path, app_class, dataset_name, max_size
):
    training, validation = object(), object()
    factory = Mock(side_effect=[training, validation])
    monkeypatch.setattr(pretrain, dataset_name, factory)
    result = app_class().datasets(
        training=tmp_path / "train", validation=tmp_path / "valid"
    )
    assert result == (training, validation)
    extra = {"max_items": None} if dataset_name == "PretrainMovieDataset" else {}
    assert factory.call_args_list == [
        call(
            path=tmp_path / "train",
            scale=4,
            min_size=16,
            max_size=max_size,
            augment=True,
            **extra,
        ),
        call(
            path=tmp_path / "valid",
            scale=4,
            min_size=16,
            max_size=max_size,
            augment=False,
            **extra,
        ),
    ]


def test_pretrain_movie_app_forwards_limits_and_options(monkeypatch, tmp_path):
    factory = Mock()
    monkeypatch.setattr(pretrain, "PretrainMovieDataset", factory)
    pretrain.SupercatPretrainMovie().datasets(
        training=tmp_path / "train",
        validation=tmp_path / "valid",
        scale=2,
        min_size=8,
        max_size=32,
        augment=False,
        max_training_items=10,
        max_validation_items=3,
    )
    assert factory.call_args_list == [
        call(
            path=tmp_path / "train",
            scale=2,
            min_size=8,
            max_size=32,
            augment=False,
            max_items=10,
        ),
        call(
            path=tmp_path / "valid",
            scale=2,
            min_size=8,
            max_size=32,
            augment=False,
            max_items=3,
        ),
    ]


@pytest.mark.parametrize(
    "app_class", [pretrain.SupercatPretrainImage, pretrain.SupercatPretrainMovie]
)
@pytest.mark.parametrize("missing", ["training", "validation"])
def test_pretrain_app_requires_dataset_paths(tmp_path, app_class, missing):
    options = dict(training=tmp_path, validation=tmp_path)
    options[missing] = None
    with pytest.raises(AssertionError, match="path must be provided"):
        app_class().datasets(**options)
