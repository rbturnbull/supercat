from fastcore.transform import DisplayedTransform
from fastai.data.block import TransformBlock
import hdf5storage
import numpy as np
from pathlib import Path
from .interpolation import interpolate3D

DEEPROCK_HDF5_KEY = "temp"

def read3D(path:Path):
    data_dict = hdf5storage.loadmat(str(path))
    assert DEEPROCK_HDF5_KEY in data_dict
    return np.float32(data_dict[DEEPROCK_HDF5_KEY]/255.0)


def write3D(path, data):
    hdf5storage.savemat(str(path), {DEEPROCK_HDF5_KEY:data*255.0}, format='7.3', oned_as='column', store_python_metadata=True)


def unsqueeze(inputs):
    """Adds a dimension for the single channel."""
    return inputs.unsqueeze(dim=1)


def ImageBlock3D():
    return TransformBlock(
        type_tfms=read3D,
        batch_tfms=unsqueeze,
    )


class InterpolateTransform(DisplayedTransform):
    def __init__(self, width, height, depth):
        self.shape = (width, height, depth)

    def encodes(self, data:np.ndarray):
        data = interpolate3D(data, self.shape)
        return data
    