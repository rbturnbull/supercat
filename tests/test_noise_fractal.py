import numpy as np
import pytest
import torch
from supercat.noise.fractal import FractalNoiseTensor



# def test_fractal_checking():
#     seed = 42
#     density = 23
#     shape = (100, 100)
#     with pytest.raises(AssertionError):
#         FractalNoise(shape, density, n=0, seed=seed)

#     with pytest.raises(AssertionError):
#         FractalNoise(shape, density, n=24, seed=seed)


def test_fractal_noise_tensor_2D():
    shape = (100, 100)
    fn = FractalNoiseTensor(shape)

    assert fn.shape == shape
    assert fn.dim == 2
    
    output = fn()
    assert output.shape == (1,) + shape
    assert isinstance(output, torch.Tensor)
    assert output.min() < -0.99
    assert output.max() > 0.99


def test_fractal_noise_tensor_3D():
    shape = (25, 25, 25)
    fn = FractalNoiseTensor(shape)

    assert fn.shape == shape
    assert fn.dim == 3
    
    output = fn()
    assert output.shape == (1,) + shape
    assert isinstance(output, torch.Tensor)
    assert output.min() < -0.99
    assert output.max() > 0.99


