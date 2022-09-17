from torchapp.testing import TorchAppTestCase
from supercat.apps import Supercat, Supercat3d


class TestSupercat(TorchAppTestCase):
    app_class = Supercat


class TestSupercat3d(TorchAppTestCase):
    app_class = Supercat3d
