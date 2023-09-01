from torchapp.testing import TorchAppTestCase
from supercat.noise.apps import NoiseSR, NoiseTensorGenerator
import torch

class TestNoise(TorchAppTestCase):
    app_class = NoiseSR


def test_noise_tensor_generator():
    shape = (100, 100)
    noise_generator = NoiseTensorGenerator(shape)

    assert noise_generator.shape == shape
    
    output = noise_generator()
    assert output.shape == (1,) + shape
    assert isinstance(output, torch.Tensor)
    assert output.min() < -0.99
    assert output.max() > 0.99
