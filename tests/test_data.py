import hdf5storage
import numpy as np
import pytest
import torch
from PIL import Image

from supercat import data


def save_mat(path, values, key="temp"):
    path.parent.mkdir(parents=True, exist_ok=True)
    hdf5storage.savemat(str(path), {key: values}, format="7.3")


@pytest.fixture
def deeprock_pair(tmp_path):
    def create(dim, partition="train", scale=2, category="sandstone", name="sample"):
        prefix = f"{category}{dim}D"
        root = tmp_path / prefix
        suffix = ".png" if dim == 2 else ".mat"
        hr_path = root / f"{prefix}_{partition}_HR" / f"{name}{suffix}"
        hr_path.parent.mkdir(parents=True, exist_ok=True)
        if dim == 2:
            lr_path = root / f"{prefix}_{partition}_BI_unknown_X{scale}" / f"{name}.png"
            lr_path.parent.mkdir(parents=True, exist_ok=True)
            hr = np.arange(64, dtype=np.uint8).reshape(8, 8)
            lr = hr + 20
            Image.fromarray(hr).save(hr_path)
            Image.fromarray(lr).save(lr_path)
        else:
            lr_path = root / f"{prefix}_{partition}_LR_default_X{scale}" / f"{name}x{scale}.mat"
            hr = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
            lr = np.full((4 // scale,) * 3, 127.5, dtype=np.float32)
            save_mat(hr_path, hr)
            save_mat(lr_path, lr)
        return hr_path, lr_path, hr, lr

    return create


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".jpeg", ".PNG"])
def test_read_image_converts_rgb_to_channel_first_grayscale(tmp_path, suffix):
    path = tmp_path / f"image{suffix}"
    Image.new("RGB", (6, 4), color=(255, 0, 0)).save(path)
    result = data.read_image(str(path))
    with Image.open(path) as image:
        expected = np.asarray(image.convert("L"), dtype=np.float32)[None]
    assert result.shape == (1, 4, 6)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)


def test_read_image_resizes_2d_image(tmp_path):
    path = tmp_path / "image.png"
    Image.new("L", (6, 4), color=100).save(path)
    result = data.read_image(path, size=(8, 10))
    assert result.shape == (1, 8, 10)
    np.testing.assert_array_equal(result, 100)


def test_read_mat_preserves_volume(tmp_path):
    path = tmp_path / "volume.mat"
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    save_mat(path, values)
    np.testing.assert_array_equal(data.read_mat(path), values)
    np.testing.assert_array_equal(data.read_image(path), values[None])


def test_read_image_resizes_volume(tmp_path):
    path = tmp_path / "volume.mat"
    save_mat(path, np.full((2, 3, 4), 100, dtype=np.float32))
    result = data.read_image(path, size=(4, 6, 8))
    assert result.shape == (1, 4, 6, 8)
    np.testing.assert_allclose(result, 100)


def test_read_image_rejects_non_volume_mat(tmp_path):
    path = tmp_path / "image.mat"
    save_mat(path, np.zeros((4, 4)))
    with pytest.raises(AssertionError, match="Expected 3D array"):
        data.read_image(path)


def test_read_image_rejects_unsupported_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported image format: .txt"):
        data.read_image(tmp_path / "image.txt")


@pytest.mark.parametrize("exists", [False, True])
def test_read_mat_reports_unreadable_file(tmp_path, exists):
    path = tmp_path / "broken.mat"
    if exists:
        path.write_text("not a MATLAB file")
    with pytest.raises(IOError, match="Error reading 3D file"):
        data.read_mat(path)


def test_read_mat_reports_missing_key(tmp_path):
    path = tmp_path / "volume.mat"
    save_mat(path, np.zeros((2, 3, 4)), key="other")
    with pytest.raises(Exception, match="expected key temp not found") as error:
        data.read_mat(path)
    assert "other" in str(error.value)


@pytest.mark.parametrize("as_tensor", [False, True])
def test_transform_scale_maps_intensity_endpoints(as_tensor):
    values = np.array([0, 127.5, 255], dtype=np.float32)
    if as_tensor:
        values = torch.from_numpy(values)
    np.testing.assert_allclose(data.transform_scale(values), [-1, 0, 1])


def test_read_image_as_tensor_normalizes_and_forwards_size(tmp_path):
    path = tmp_path / "image.png"
    Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8)).save(path)
    result = data.read_image_as_tensor(path)
    torch.testing.assert_close(result, torch.tensor([[[-1.0, 1.0], [1.0, -1.0]]]))
    assert result.is_contiguous()
    resized = data.read_image_as_tensor(path, size=(4, 6))
    assert resized.shape == (1, 4, 6)
    assert resized.dtype == torch.float32


@pytest.mark.parametrize("dtype", [np.float32, np.uint8])
@pytest.mark.parametrize("function,shape", [
    (data.downscale_tricubic_rescale, (2, 3, 4)),
    (data.upscale_tricubic_rescale, (8, 12, 16)),
])
def test_rescale_preserves_constant_intensity_and_dtype(function, shape, dtype):
    volume = np.full((4, 6, 8), 100, dtype=dtype)
    result = function(volume, factor=2)
    assert result.shape == shape
    assert result.dtype == volume.dtype
    np.testing.assert_allclose(result, 100, atol=1)
    np.testing.assert_array_equal(volume, 100)


@pytest.mark.parametrize("dim,transforms", [(2, data.TRANSFORMATIONS_2D), (3, data.TRANSFORMATIONS_3D)])
def test_augmentations_preserve_values_and_channels(dim, transforms):
    sample = torch.arange(2 * 3 ** dim).reshape((2,) + (3,) * dim)
    original = sample.clone()
    for transform in transforms:
        result = transform(sample)
        assert result.shape == sample.shape
        for channel in range(2):
            torch.testing.assert_close(result[channel].flatten().sort().values, sample[channel].flatten())
    torch.testing.assert_close(sample, original)


@pytest.mark.parametrize("dataset_class", [data.Deeprock2D, data.Deeprock3D])
def test_dataset_rejects_empty_directory(tmp_path, dataset_class):
    with pytest.raises(FileNotFoundError, match="No .* files found"):
        dataset_class(tmp_path)


def test_2d_dataset_loads_normalized_pairs(tmp_path, deeprock_pair):
    _, _, hr, lr = deeprock_pair(2)
    dataset = data.Deeprock2D(tmp_path, scale=2)
    assert len(dataset) == 1
    actual_lr, actual_hr = dataset[0]
    np.testing.assert_allclose(actual_hr.numpy(), hr[None] * (2.0 / 255) - 1, atol=1e-7)
    np.testing.assert_allclose(actual_lr.numpy(), lr[None] * (2.0 / 255) - 1, atol=1e-7)


@pytest.mark.parametrize("channel_first", [True, False])
def test_3d_dataset_upsamples_pairs(tmp_path, deeprock_pair, channel_first):
    _, _, hr, _ = deeprock_pair(3)
    dataset = data.Deeprock3D(tmp_path, scale=2, channel_first=channel_first)
    assert len(dataset) == 1
    lr_tensor, hr_tensor = dataset[0]
    expected = hr[None] if channel_first else hr[..., None]
    assert lr_tensor.shape == hr_tensor.shape == expected.shape
    np.testing.assert_allclose(hr_tensor.numpy(), expected * (2.0 / 255) - 1, atol=1e-7)
    torch.testing.assert_close(lr_tensor, torch.zeros_like(lr_tensor))


@pytest.mark.parametrize("dim,dataset_class", [(2, data.Deeprock2D), (3, data.Deeprock3D)])
def test_dataset_augmentation_applies_same_flip_to_pair(tmp_path, deeprock_pair, monkeypatch, dim, dataset_class):
    deeprock_pair(dim)
    if dim == 3:
        _, lr_path, _, _ = deeprock_pair(dim)
        save_mat(lr_path, np.arange(8, dtype=np.float32).reshape(2, 2, 2) * 20)
    baseline = dataset_class(tmp_path, scale=2)[0]
    # Transformation 4 is a flip of the final spatial axis in both lists.
    monkeypatch.setattr(data.np.random, "randint", lambda *args: 4)
    augmented = dataset_class(tmp_path, scale=2, augment=True)[0]
    for actual, original in zip(augmented, baseline):
        torch.testing.assert_close(actual, original.flip(-1))


@pytest.mark.parametrize("dim,builder", [(2, data.build_datasets2D), (3, data.build_datasets3D)])
@pytest.mark.parametrize("augment", [False, True])
def test_build_datasets_selects_partitions_and_training_augmentation(tmp_path, deeprock_pair, dim, builder, augment):
    train_path, *_ = deeprock_pair(dim, partition="train")
    valid_path, *_ = deeprock_pair(dim, partition="valid")
    training, validation = builder(tmp_path, scale=2, train_augment=augment)
    assert training.hr_items == [train_path]
    assert validation.hr_items == [valid_path]
    assert training.scale == validation.scale == 2
    assert training.augment is augment
    assert validation.augment is False


@pytest.mark.parametrize("dim,dataset_class", [(2, data.Deeprock2D), (3, data.Deeprock3D)])
def test_dataset_reports_missing_low_resolution_pair(tmp_path, deeprock_pair, dim, dataset_class):
    _, lr_path, *_ = deeprock_pair(dim)
    lr_path.unlink()
    with pytest.raises(OSError):
        dataset_class(tmp_path, scale=2)[0]


def test_3d_dataset_rejects_mismatched_shapes(tmp_path, deeprock_pair):
    _, lr_path, *_ = deeprock_pair(3)
    save_mat(lr_path, np.zeros((3, 3, 3), dtype=np.float32))
    with pytest.raises(AssertionError, match="LR shape"):
        data.Deeprock3D(tmp_path, scale=2)[0]


@pytest.mark.parametrize("bad_high_resolution", [False, True])
def test_3d_dataset_rejects_out_of_range_intensities(tmp_path, deeprock_pair, bad_high_resolution):
    hr_path, lr_path, hr, lr = deeprock_pair(3)
    path, values = (hr_path, hr) if bad_high_resolution else (lr_path, lr)
    save_mat(path, np.full_like(values, 300))
    with pytest.raises(AssertionError, match="gives range"):
        data.Deeprock3D(tmp_path, scale=2)[0]


@pytest.mark.parametrize("dim,dataset_class", [(2, data.Deeprock2D), (3, data.Deeprock3D)])
def test_dataset_rejects_high_resolution_path_outside_hr_directory(tmp_path, deeprock_pair, dim, dataset_class):
    hr_path, *_ = deeprock_pair(dim)
    dataset = dataset_class(tmp_path, scale=2)
    prefix = f"sandstone{dim}D"
    unexpected = tmp_path / prefix / f"{prefix}_train_LR" / hr_path.name
    with pytest.raises(ValueError, match=f"Unexpected parent '{prefix}_train_LR'"):
        dataset._hr_to_lr_path(unexpected)
