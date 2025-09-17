from pathlib import Path
from widitapp.apps import WiDiTApp
from cluey import main, tool, method

from .data import build_datasets3D, build_datasets2D

class Supercat(WiDiTApp):
    @method
    def datasets(
        self,
        dim:int=3,
        deeprock:Path=None,
        scale:int=4,
        **kwargs,
    ) -> tuple:
        """ Returns training and validation datasets """
        build_function = build_datasets2D if dim == 2 else build_datasets3D
        return build_function(deeprock=deeprock, scale=scale)

