from supercat import transforms
import numpy as np
import tempfile
from pathlib import Path
import torch

def test_read_write_3D_mat():
    data = np.random.normal(size=(10, 10, 10))

    with tempfile.TemporaryDirectory() as tmpdirname:
        path = Path(tmpdirname)/'test.mat'
        transforms.write3D(path, data)

        result = transforms.read3D(path)

    assert np.allclose(data, result)


def test_read_write_3D_tif():
    data = np.random.normal(size=(10, 10, 10))

    with tempfile.TemporaryDirectory() as tmpdirname:
        path = Path(tmpdirname)/'test.tif'
        transforms.write3D(path, data)

        result = transforms.read3D(path)

    assert np.allclose(data, result)    


def test_interpolate_2D():
    transform = transforms.InterpolateTransform(width=20)

    data = np.random.normal(size=(10, 10))
    result = transform(data)
    assert result.shape == (1, 20, 20)