import functools
import math
from pathlib import Path
import random
import re
import shutil
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import cluey
from cluey import method
from torch.utils.data import Dataset
from widitapp import WiDiTApp

from .data import read_image_as_tensor, TRANSFORMATIONS_2D, TRANSFORMATIONS_3D


def _normalize_video_frame(frame: np.ndarray) -> np.ndarray:
    """Convert a decoded frame to a 2D grayscale array."""
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[-1] == 1:
        return frame[..., 0]
    if frame.ndim == 3:
        # Match PIL's default grayscale conversion closely enough for training.
        return np.round(frame[..., :3].mean(axis=-1)).astype(frame.dtype)
    raise ValueError(f"Unsupported frame shape: {frame.shape}")


def _ffmpeg_binary() -> str | None:
    return shutil.which("ffmpeg")


def _probe_ffmpeg_video(path: Path) -> tuple[int, int, int]:
    ffmpeg = _ffmpeg_binary()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg binary not found")

    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr

    size_match = re.search(r"Video:.*?,\s+(\d+)x(\d+)\s+\[", output)
    if size_match is None:
        raise RuntimeError(f"Could not determine video size for {path}")
    frame_width = int(size_match.group(1))
    frame_height = int(size_match.group(2))

    frame_matches = re.findall(r"frame=\s*(\d+)", output)
    if not frame_matches:
        raise RuntimeError(f"Could not determine frame count for {path}")
    frame_count = int(frame_matches[-1])

    return frame_count, frame_height, frame_width


@functools.lru_cache(maxsize=8)
def _read_video_via_ffmpeg(path: str) -> np.ndarray:
    video_path = Path(path)
    frame_count, frame_height, frame_width = _probe_ffmpeg_video(video_path)
    ffmpeg = _ffmpeg_binary()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg binary not found")

    proc = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-vf",
            "format=gray",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        capture_output=True,
        check=True,
    )

    video = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_size = frame_height * frame_width
    actual_frame_count, remainder = divmod(video.size, frame_size)
    if remainder != 0:
        raise RuntimeError(f"Unexpected ffmpeg output size for {video_path}")

    if actual_frame_count != frame_count:
        frame_count = actual_frame_count

    return video.reshape(frame_count, frame_height, frame_width)


def _video_backend(path: Path):
    try:
        import imageio.v3 as iio
    except ImportError:
        iio = None

    if iio is not None:
        try:
            iio.improps(path)
            return "imageio", iio
        except OSError:
            pass

    try:
        from skvideo.io import ffprobe, vreader
    except ImportError:
        ffprobe = None
        vreader = None

    if ffprobe is not None and vreader is not None:
        try:
            next(vreader(str(path), as_grey=True))
            return "skvideo", (ffprobe, vreader)
        except (AssertionError, OSError, StopIteration):
            pass

    if _ffmpeg_binary() is not None:
        return "ffmpeg_cli", None

    raise RuntimeError(
        "No supported video backend is available. Install `imageio[ffmpeg]` "
        "or make an `ffmpeg` binary available on PATH."
    )


def iter_video_frames(path: Path):
    backend, reader_impl = _video_backend(path)
    if backend == "imageio":
        for frame in reader_impl.imiter(path):
            yield _normalize_video_frame(np.asarray(frame))
        return

    if backend == "ffmpeg_cli":
        for frame in _read_video_via_ffmpeg(str(path)):
            yield frame
        return

    _, vreader = reader_impl
    for frame in vreader(str(path), as_grey=True):
        yield np.asarray(frame[0, :, :, 0])


def video_shape(path: Path) -> tuple[int, int, int]:
    backend, reader_impl = _video_backend(path)

    if backend == "imageio":
        props = reader_impl.improps(path)
        shape = props.shape
        if len(shape) < 3:
            raise ValueError(
                f"Unsupported video shape from imageio for {path}: {shape}"
            )
        frame_count, frame_height, frame_width = shape[:3]
        if not math.isfinite(float(frame_count)):
            frame_count = sum(1 for _ in reader_impl.imiter(path))
        return int(frame_count), int(frame_height), int(frame_width)

    if backend == "ffmpeg_cli":
        video = _read_video_via_ffmpeg(str(path))
        frame_count, frame_height, frame_width = video.shape
        return int(frame_count), int(frame_height), int(frame_width)

    ffprobe, _ = reader_impl
    metadata = ffprobe(str(path))
    video_metadata = metadata.get("video", {})

    frame_height = int(video_metadata["@height"])
    frame_width = int(video_metadata["@width"])

    try:
        frame_count = int(video_metadata["@nb_frames"])
    except (KeyError, ValueError):
        frame_count = sum(1 for _ in iter_video_frames(path))

    return frame_count, frame_height, frame_width


def _pad_to_size_with_reflect(
    image: torch.Tensor,
    target_size: int,
    spatial_dims: int = 2,
) -> torch.Tensor:
    """
    Pad the trailing spatial dimensions to ``target_size`` using safe reflect steps.

    For each spatial dimension, pad only on the "right" side until that dimension
    reaches ``target_size``. Uses reflect padding where possible, and falls back to
    replicate padding when the size along a dimension is 1.

    Examples
    --------
    2D expected input shape:
        (..., H, W)

    3D expected input shape:
        (..., D, H, W)
    """
    if spatial_dims not in (2, 3):
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

    if image.ndim < spatial_dims:
        raise ValueError(
            f"image has {image.ndim} dims, but spatial_dims={spatial_dims}"
        )

    # Iterate from last spatial dim backwards:
    # 2D: -1, -2
    # 3D: -1, -2, -3
    for dim in range(-1, -spatial_dims - 1, -1):
        while image.shape[dim] < target_size:
            remaining = target_size - image.shape[dim]
            current = image.shape[dim]

            if current <= 1:
                # Reflection needs at least 2 pixels along the padded dimension.
                mode = "replicate"
                pad_amount = remaining
            else:
                mode = "reflect"
                pad_amount = min(remaining, current - 1)

            # F.pad expects padding specified from the last dimension moving left:
            # 2D: (W_left, W_right, H_left, H_right)
            # 3D: (W_left, W_right, H_left, H_right, D_left, D_right)
            pad = [0] * (2 * spatial_dims)

            # For dim=-1 -> index 1
            # For dim=-2 -> index 3
            # For dim=-3 -> index 5
            pad_index = 2 * (-dim) - 1
            pad[pad_index] = pad_amount

            image = F.pad(image, tuple(pad), mode=mode)

    return image


def find_files(
    base_path: Path, valid_extensions, max_items: int | None = None
) -> list[Path]:
    """Find all files under ``base_path`` matching any of the given extensions."""
    if not base_path.is_dir():
        raise ValueError(f"{base_path} is not a valid directory")

    files = []
    for ext in valid_extensions:
        files.extend(base_path.rglob(f"*{ext.lower()}"))
        files.extend(base_path.rglob(f"*{ext.upper()}"))

        if max_items is not None and len(files) >= max_items:
            return files[:max_items]

    return files


def find_images(base_path: Path) -> list[Path]:
    return find_files(base_path, [".jpg", ".jpeg", ".png", ".bmp", ".tiff"])


def find_movies(base_path: Path, max_items: int | None = None) -> list[Path]:
    return find_files(base_path, [".mp4", ".avi", ".mov", ".mkv"], max_items=max_items)


class PretrainImagesDataset(Dataset):
    def __init__(
        self,
        path: Path,
        scale: int = 4,
        channel_first: bool = True,
        augment: bool = False,
        size: int = 0,
        min_size: int = 0,
        max_size: int = 0,
    ):
        assert path is not None, "Path must be provided"
        self.path = Path(path)
        self.scale = int(scale)
        self.channel_first = channel_first
        self.augment = augment
        self.size = size
        if size:
            min_size = size
            max_size = size
        self.min_size = min_size
        self.max_size = max_size

        self.items = find_images(path)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.items[idx]
        hr_t = read_image_as_tensor(path)

        if self.max_size:
            if hr_t.shape[-1] > self.max_size:
                start = (
                    np.random.randint(0, hr_t.shape[-1] - self.max_size)
                    if self.augment
                    else hr_t.shape[-1] // 2 - self.max_size // 2
                )
                hr_t = hr_t[..., start : start + self.max_size]
            if hr_t.shape[-2] > self.max_size:
                start = (
                    np.random.randint(0, hr_t.shape[-2] - self.max_size)
                    if self.augment
                    else hr_t.shape[-2] // 2 - self.max_size // 2
                )
                hr_t = hr_t[..., start : start + self.max_size, :]

        if self.min_size:
            hr_t = _pad_to_size_with_reflect(hr_t, self.min_size)

        # Ensure dimensions are even
        if hr_t.shape[-1] % 2 != 0:
            hr_t = hr_t[..., :-1]
        if hr_t.shape[-2] % 2 != 0:
            hr_t = hr_t[..., :-1, :]

        lr_t = torch.nn.functional.interpolate(
            hr_t.unsqueeze(0),
            scale_factor=1 / self.scale,
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        lr_t = torch.nn.functional.interpolate(
            lr_t.unsqueeze(0),
            size=hr_t.shape[1:],
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        if self.augment:
            transformation = TRANSFORMATIONS_2D[
                np.random.randint(0, len(TRANSFORMATIONS_2D))
            ]
            hr_t = transformation(hr_t)
            lr_t = transformation(lr_t)

        return lr_t, hr_t


class PretrainMovieDataset(Dataset):
    def __init__(
        self,
        path: Path,
        scale: int = 4,
        channel_first: bool = True,
        augment: bool = False,
        size: int = 0,
        min_size: int = 0,
        max_size: int = 0,
        max_items: int | None = None,
    ):
        assert path is not None, "Path must be provided"
        self.path = Path(path)
        self.scale = int(scale)
        self.channel_first = channel_first
        self.augment = augment
        self.size = size
        if size:
            min_size = size
            max_size = size
        self.min_size = min_size
        self.max_size = max_size

        self.items = find_movies(path, max_items=max_items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.items[idx]

        try:
            frame_count, frame_height, frame_width = video_shape(path)

            depth = self.max_size or frame_count
            height = self.max_size or frame_height
            width = self.max_size or frame_width

            frame_start = (
                random.randint(0, max(frame_count - depth, 0))
                if self.augment
                else max(frame_count // 2 - depth // 2, 0)
            )
            frame_end = min(frame_start + depth, frame_count)

            y_start = (
                random.randint(0, max(frame_height - height, 0))
                if self.augment
                else max(frame_height // 2 - height // 2, 0)
            )
            y_end = min(y_start + height, frame_height)
            x_start = (
                random.randint(0, max(frame_width - width, 0))
                if self.augment
                else max(frame_width // 2 - width // 2, 0)
            )
            x_end = min(x_start + width, frame_width)

            image = np.zeros(
                (frame_end - frame_start, y_end - y_start, x_end - x_start),
                dtype=np.uint8,
            )

            for i, frame in enumerate(iter_video_frames(path)):
                if i < frame_start:
                    continue
                if i >= frame_end:
                    break
                image[i - frame_start, :, :] = frame[y_start:y_end, x_start:x_end]

            hr_t = torch.tensor(image / 255.0 * 2 - 1.0, dtype=torch.float32).unsqueeze(
                0
            )
        except Exception as e:
            print(f"Failed to load video {path}: {e}")
            zeros = torch.zeros(
                (1, self.min_size or 16, self.min_size or 16, self.min_size or 16),
                dtype=torch.float32,
            )
            return zeros, zeros

        # Ensure dimensions are even
        if hr_t.shape[-1] % 2 != 0:
            hr_t = hr_t[..., :-1]
        if hr_t.shape[-2] % 2 != 0:
            hr_t = hr_t[..., :-1, :]
        if hr_t.shape[-3] % 2 != 0:
            hr_t = hr_t[..., :-1, :, :]

        # Pad to the target size if needed, since some videos may be smaller than the requested crop size.
        if self.min_size:
            hr_t = _pad_to_size_with_reflect(hr_t, self.min_size, spatial_dims=3)

        lr_t = torch.nn.functional.interpolate(
            hr_t.unsqueeze(0),
            scale_factor=1 / self.scale,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        lr_t = torch.nn.functional.interpolate(
            lr_t.unsqueeze(0),
            size=hr_t.shape[1:],
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        if self.augment:
            transformation = TRANSFORMATIONS_3D[
                np.random.randint(0, len(TRANSFORMATIONS_3D))
            ]
            hr_t = transformation(hr_t)
            lr_t = transformation(lr_t)

        if not self.channel_first:
            hr_t = hr_t.movedim(0, -1)
            lr_t = lr_t.movedim(0, -1)

        return lr_t, hr_t


class SupercatPretrainImage(WiDiTApp):
    @method
    def datasets(
        self,
        training: Path = cluey.Option(
            None, help="Path to the training image directory"
        ),
        validation: Path = cluey.Option(
            None, help="Path to the validation image directory"
        ),
        scale: int = cluey.Option(4, help="Scale factor for downsampling the images"),
        augment: bool = cluey.Option(
            True,
            help="Apply random cropping and augmentation to the training image samples",
        ),
        min_size: int = cluey.Option(
            16,
            help="Minimum size for each spatial dimension after padding (0 disables padding)",
        ),
        max_size: int = cluey.Option(
            224,
            help="Maximum crop size for each spatial dimension (0 disables cropping)",
        ),
        **kwargs,
    ) -> tuple[Dataset, Dataset]:
        """Build training and validation datasets for 2D image pretraining."""
        assert training is not None, "Training path must be provided"
        assert validation is not None, "Validation path must be provided"
        training_dataset = PretrainImagesDataset(
            path=training,
            scale=scale,
            min_size=min_size,
            max_size=max_size,
            augment=augment,
        )
        validation_dataset = PretrainImagesDataset(
            path=validation,
            scale=scale,
            min_size=min_size,
            max_size=max_size,
            augment=False,
        )
        return training_dataset, validation_dataset


class SupercatPretrainMovie(WiDiTApp):
    @method
    def datasets(
        self,
        training: Path = cluey.Option(
            None, help="Path to the training video directory"
        ),
        validation: Path = cluey.Option(
            None, help="Path to the validation video directory"
        ),
        scale: int = cluey.Option(4, help="Scale factor for downsampling the images"),
        augment: bool = cluey.Option(
            True,
            help="Apply random cropping and augmentation to the training video samples",
        ),
        min_size: int = cluey.Option(
            16,
            help="Minimum frame count, height, and width after padding (0 disables padding)",
        ),
        max_size: int = cluey.Option(
            100,
            help="Maximum frame count, height, and width per crop (0 uses the full video)",
        ),
        max_training_items: int | None = cluey.Option(
            None, help="Maximum number of training videos (None uses all videos)"
        ),
        max_validation_items: int | None = cluey.Option(
            None, help="Maximum number of validation videos (None uses all videos)"
        ),
        **kwargs,
    ) -> tuple[Dataset, Dataset]:
        """Build training and validation datasets for 3D video pretraining."""
        assert training is not None, "Training path must be provided"
        assert validation is not None, "Validation path must be provided"

        training_dataset = PretrainMovieDataset(
            path=training,
            scale=scale,
            min_size=min_size,
            max_size=max_size,
            augment=augment,
            max_items=max_training_items,
        )
        validation_dataset = PretrainMovieDataset(
            path=validation,
            scale=scale,
            min_size=min_size,
            max_size=max_size,
            augment=False,
            max_items=max_validation_items,
        )
        return training_dataset, validation_dataset
