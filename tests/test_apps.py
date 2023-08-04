from torchapp.testing import TorchAppTestCase
from supercat.apps import Supercat, SupercatDiffusion


class TestSupercat(TorchAppTestCase):
    app_class = Supercat


class TestSupercatDiffusion(TorchAppTestCase):
    app_class = SupercatDiffusion


