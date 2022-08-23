from fastapp.testing import FastAppTestCase
from supercat.apps import Supercat, Supercat3d


class TestSupercat(FastAppTestCase):
    app_class = Supercat


class TestSupercat3d(FastAppTestCase):
    app_class = Supercat3d
