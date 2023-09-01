from torchapp.testing import TorchAppTestCase
from supercat.noise.apps import NoiseSR


class TestNoise(TorchAppTestCase):
    app_class = NoiseSR


