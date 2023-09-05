import numpy as np
import pytest
import torch
from supercat.noise.worley import WorleyNoise, WorleyNoiseTensor

def test_worley_noise():
    seed = 42
    density = 23
    shape = (100, 100)
    wn = WorleyNoise(shape, density, seed=seed)
    assert wn.shape == shape
    assert wn.density == density
    
    output = wn()
    assert output.shape == shape
    assert isinstance(output, np.ndarray)
    

def test_worley_noise_n_all():
    seed = 42
    density = 23
    shape = (100, 100)
    wn = WorleyNoise(shape, density, n=None, seed=seed)
    assert wn.shape == shape
    assert wn.density == density
    
    output = wn()
    assert output.shape == (density,) + shape
    assert isinstance(output, np.ndarray)
        

def test_worley_n_10():
    seed = 42
    density = 23
    shape = (100, 100)
    wn = WorleyNoise(shape, density, n=10, seed=seed)
    assert wn.shape == shape
    assert wn.density == density
    
    output = wn()
    assert output.shape == shape
    assert isinstance(output, np.ndarray)
    

def test_worley_checking():
    seed = 42
    density = 23
    shape = (100, 100)
    with pytest.raises(AssertionError):
        WorleyNoise(shape, density, n=0, seed=seed)

    with pytest.raises(AssertionError):
        WorleyNoise(shape, density, n=24, seed=seed)


def test_worley_noise_tensor():
    seed = 42
    density = 23
    shape = (100, 100)
    wn = WorleyNoiseTensor(shape, density, seed=seed)

    assert wn.shape == shape
    assert wn.density == density
    
    output = wn()
    assert output.shape == (1,) + shape
    assert isinstance(output, torch.Tensor)
    assert output.min() < -0.99
    assert output.max() > 0.99
    

