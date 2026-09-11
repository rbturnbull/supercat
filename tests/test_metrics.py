import numpy as np
import pytest
import torch

from supercat.metrics import calc_porosity


def test_calc_porosity_splits_bimodal_image_evenly():
    data = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0], dtype=np.float32)
    assert calc_porosity(data) == pytest.approx(0.5)


def test_calc_porosity_counts_void_fraction_of_volume():
    volume = np.ones((4, 4, 5), dtype=np.float32)
    volume[0] = 0.0
    assert calc_porosity(volume) == pytest.approx(0.25)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_calc_porosity_accepts_tensors(dtype):
    values = [0.0, 0.1, 0.2, 0.8, 0.9, 1.0]
    tensor = torch.tensor(values, dtype=dtype)
    expected = calc_porosity(np.array(values, dtype=np.float32))
    assert calc_porosity(tensor) == pytest.approx(expected)
    torch.testing.assert_close(tensor, torch.tensor(values, dtype=dtype))


def test_calc_porosity_of_uniform_data_is_zero():
    assert calc_porosity(np.zeros((3, 3), dtype=np.float32)) == 0.0
