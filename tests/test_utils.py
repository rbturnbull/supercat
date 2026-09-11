import numpy as np
import pytest
import torch

from supercat.utils import (
    distance_to_boundary,
    generate_intervals,
    generate_overlapping_intervals,
    write_image,
)


@pytest.mark.parametrize(
    "n,k,expected",
    [
        (10, 2, [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]),
        (10, 3, [(0, 3), (3, 6), (6, 9)]),
        (11, 3, [(1, 4), (4, 7), (7, 10)]),
        (5, 5, [(0, 5)]),
        (5, 6, []),
        (5, 0, []),
        (0, 5, []),
        (0, 0, []),
    ],
)
def test_generate_intervals(n, k, expected):
    assert generate_intervals(n, k) == expected


@pytest.mark.parametrize("variable_size", [False, True])
@pytest.mark.parametrize("total,size,overlap", [(20, 5, 2), (23, 8, 2), (8, 8, 2), (24, 8, 0)])
def test_overlapping_intervals_cover_range(total, size, overlap, variable_size):
    intervals = generate_overlapping_intervals(
        total, size, overlap, variable_size=variable_size
    )

    assert intervals[0][0] == 0
    assert intervals[-1][1] == total
    covered = set()
    for start, end in intervals:
        assert 0 <= start < end <= total
        assert end - start <= size
        if not variable_size:
            assert end - start == size
        covered.update(range(start, end))
    assert covered == set(range(total))
    for previous, current in zip(intervals, intervals[1:]):
        assert current[0] > previous[0]
        assert previous[1] - current[0] >= overlap


def test_overlapping_intervals_example():
    assert generate_overlapping_intervals(20, 5, 2) == [
        (0, 5), (3, 8), (6, 11), (9, 14), (12, 17), (15, 20)
    ]


def test_overlapping_intervals_variable_size_reduces_tile_size():
    assert generate_overlapping_intervals(10, 8, 2, variable_size=True) == [(0, 6), (4, 10)]


def test_overlapping_intervals_empty_range():
    assert generate_overlapping_intervals(0, 5, 2) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (5, None), (5, 5), (5, 6)])
def test_overlapping_intervals_reject_invalid_parameters(size, overlap):
    with pytest.raises(AssertionError):
        generate_overlapping_intervals(20, size, overlap)


def test_overlapping_intervals_unchecked_short_range():
    assert generate_overlapping_intervals(3, 5, 1, check=False) == [(0, 3)]


def test_distance_to_boundary_concentric_layers():
    result = distance_to_boundary(5, 5, 5)
    expected = torch.zeros((5, 5, 5), dtype=torch.int64)
    expected[1:4, 1:4, 1:4] = 1
    expected[2, 2, 2] = 2
    torch.testing.assert_close(result, expected)


def test_distance_to_boundary_rectangular_volume():
    result = distance_to_boundary(5, 7, 9)
    assert result.shape == (5, 7, 9)
    assert result[2, 3, 4] == 2
    assert result[2, 1, 4] == 1
    assert result[2, 3, 1] == 1
    for axis in range(3):
        assert torch.count_nonzero(result.select(axis, 0)) == 0
        assert torch.count_nonzero(result.select(axis, result.shape[axis] - 1)) == 0
        torch.testing.assert_close(result, result.flip(axis))


@pytest.mark.parametrize("shape", [(1, 4, 5), (4, 1, 5), (4, 5, 1), (0, 4, 5)])
def test_distance_to_boundary_degenerate_volume(shape):
    result = distance_to_boundary(*shape)
    assert result.shape == shape
    assert torch.count_nonzero(result) == 0


def test_write_tensor_preserves_values_and_creates_directories(tmp_path):
    data = torch.tensor([[[-2.0, 0.25, 3.0]]], dtype=torch.float64)
    path = tmp_path / "nested" / "prediction.PT"
    write_image(data, str(path))
    torch.testing.assert_close(torch.load(path, weights_only=True), data)


@pytest.mark.parametrize("suffix", [".tif", ".png", ".mat"])
@pytest.mark.parametrize("as_tensor", [False, True])
def test_write_image_clips_and_scales_without_mutating_input(tmp_path, suffix, as_tensor):
    data = np.tile(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32), (5, 1))
    original = data.copy()
    source = torch.from_numpy(data) if as_tensor else data
    path = tmp_path / "nested" / f"prediction{suffix}"
    write_image(source, path)

    if suffix == ".tif":
        import tifffile
        actual = tifffile.imread(path)
    elif suffix == ".mat":
        import hdf5storage
        actual = hdf5storage.loadmat(str(path))["temp"]
    else:
        from PIL import Image
        with Image.open(path) as image:
            actual = np.asarray(image)

    expected = np.tile(np.array([0, 0, 127, 255, 255], dtype=np.uint8), (5, 1))
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(data, original)


def test_write_nrrd_preserves_volume_values(tmp_path):
    nrrd = pytest.importorskip("nrrd")
    data = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    path = tmp_path / "nested" / "prediction.nrrd"
    write_image(data, path)
    actual, _ = nrrd.read(str(path))
    np.testing.assert_array_equal(actual, data.numpy())


def test_write_nrrd_registers_half_precision_type(tmp_path, monkeypatch):
    import sys
    import types

    writer = types.ModuleType("nrrd.writer")
    writer._TYPEMAP_NUMPY2NRRD = {"f4": "float"}
    nrrd = types.ModuleType("nrrd")
    nrrd.writer = writer
    written = {}
    nrrd.write = lambda path, data: written.update(path=path, data=data)
    monkeypatch.setitem(sys.modules, "nrrd", nrrd)
    monkeypatch.setitem(sys.modules, "nrrd.writer", writer)

    data = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    path = tmp_path / "nested" / "prediction.NRRD"
    write_image(data, path)

    assert writer._TYPEMAP_NUMPY2NRRD == {"f4": "float", "f2": "float16"}
    assert written["path"] == str(path)
    np.testing.assert_array_equal(written["data"], data.numpy())
    assert path.parent.is_dir()
