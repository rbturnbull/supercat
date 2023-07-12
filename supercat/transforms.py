from fastcore.transform import DisplayedTransform
from fastai.data.block import TransformBlock
import hdf5storage
import numpy as np
from pathlib import Path
from fastai.vision.data import TensorImage
from skimage.transform import resize as skresize
from skimage import io


DEEPROCK_HDF5_KEY = "temp"

def read3D(path:Path):
    path = Path(path)        
    if path.suffix == ".mat":
        data_dict = hdf5storage.loadmat(str(path))
        if DEEPROCK_HDF5_KEY not in data_dict:
            keys_found = ",".join(data_dict.keys())
            raise Exception(f"expected key {DEEPROCK_HDF5_KEY} not found in '{path}'.\nCheck the following keys: {keys_found}")

        result =  np.float32(data_dict[DEEPROCK_HDF5_KEY]/255.0)
    else:
        result = np.float32(io.imread(path))
        
    print("result.shape", result.shape)
    print(result.min(), result.max())

    result /= 255.0

    return result


def write3D(path:Path, data):
    path = Path(path)

    if path.suffix == ".mat":
        hdf5storage.savemat(str(path), {DEEPROCK_HDF5_KEY:data*255.0}, format='7.3', oned_as='column', store_python_metadata=True)
    else:
        io.imsave(path, data)


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
        return skresize(data, self.shape, order=3)    


class RescaleImage(DisplayedTransform):
    order = 20 #Need to run after IntToFloatTensor
    
    def encodes(self, item:TensorImage): 
        return item.float()*2.0 - 1.0


