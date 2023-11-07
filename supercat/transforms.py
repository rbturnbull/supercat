import random
from pathlib import Path

import hdf5storage
import numpy as np
import torch
import torchvision
from fastai.data.block import TransformBlock
from fastai.data.transforms import image_extensions
from fastai.vision.core import PILImageBW
from fastai.vision.data import TensorImage
from fastcore.transform import DisplayedTransform
from skimage import io
from skimage.transform import resize as skresize
from torchvision.io import VideoReader
from torchvision.transforms import v2

DEEPROCK_HDF5_KEY = "temp"

def read3D(path:Path):
    path = Path(path)
    if path.suffix == ".mat":
        try:
            data_dict = hdf5storage.loadmat(str(path))
        except Exception as err:
            raise IOError(f"Error reading 3D file '{path}':\n{err}")
        if DEEPROCK_HDF5_KEY not in data_dict:
            keys_found = ",".join(data_dict.keys())
            raise Exception(f"expected key {DEEPROCK_HDF5_KEY} not found in '{path}'.\nCheck the following keys: {keys_found}")

        result =  np.float32(data_dict[DEEPROCK_HDF5_KEY]/255.0)
    else:
        result = np.float32(io.imread(path))

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
    def __init__(self, width=None, *, depth=None, height=None, dim=2):
        self.width = width
        assert width != None
        
        self.height = height or width
        self.depth = depth or width

        self.dim = dim
        if dim == 3:
            self.shape = (depth, height, width)
        elif dim == 2:
            self.shape = (height, width)
        else:
            raise ValueError("dim must be 2 or 3")

    def encodes(self, data):
        if len(data.shape) == self.dim + 1:
            data = data.squeeze(0)

        result = skresize(data, self.shape, order=3)
        assert result.shape == self.shape
        return np.expand_dims(result, 0)


class RescaleImage(DisplayedTransform):
    order = 20 #Need to run after IntToFloatTensor
    
    def encodes(self, item): 
        if not isinstance(item, torch.Tensor):
            item = torch.tensor(item)

        return item.float()*2.0 - 1.0


class RescaleImageMinMax(DisplayedTransform):
    def __init__(self, rescaled_min=-0.95, rescaled_max=0.95):
        self.extrema = []
        self.rescaled_min = rescaled_min
        self.rescaled_max = rescaled_max
        self.factor = (self.rescaled_max - self.rescaled_min)

    def encodes(self, item): 
        
        
        if isinstance(item, PILImageBW):
            item = np.expand_dims(np.asarray(item), 0)

        if not isinstance(item, torch.Tensor):
            item = torch.tensor(item)
        
        
        min, max = item.min(), item.max()
        self.extrema.append( (min,max) )
        transformed_item = (item.float() - min) / (max-min) * self.factor + self.rescaled_min
        return transformed_item
    
    def decodes(self, item, min, max):
        return (item - self.rescaled_min)/self.factor * (max-min) + min


class CropTransform(DisplayedTransform):
    def __init__(self, start_x:int=None, end_x:int=None, start_y:int=None, end_y:int=None, start_z:int=None, end_z:int=None ):
        self.start_x = start_x or None
        self.end_x = end_x or None
        self.start_y = start_y or None
        self.end_y = end_y or None
        self.start_z = start_z or None
        self.end_z = end_z or None

    def encodes(self, data):
        if isinstance(data, PILImageBW):
            data = np.expand_dims(np.asarray(data), 0)

        if len(data.shape) == 3:
            return data[self.start_z:self.end_z,self.start_y:self.end_y,self.start_x:self.end_x]
        return data[self.start_y:self.end_y,self.start_x:self.end_x]        

class ImageVideoReader(DisplayedTransform):
    def __init__(self, shape) -> None:
        self.shape = list(shape)
        self.dim = len(shape)

    def _rotate_info(self, shape):
        """
        Returns the shape, degree and axis for rotation process.

        If the shape is 2D, no adjustment will be applied to shape,
        and the degree and axis will be 0 and [0, 0] respectively.
        """
        if self.dim == 2:
            return shape, 0, [0, 0]

        rotate_degree = random.randint(0, 2)
        rotate_axis = random.sample([0, 1, 2], 2)

        rotate_shape = shape.copy()
        rotate_shape[rotate_axis[0]] = shape[rotate_axis[1]] if rotate_degree == 1 else rotate_shape[rotate_axis[0]]
        rotate_shape[rotate_axis[1]] = shape[rotate_axis[0]] if rotate_degree == 1 else rotate_shape[rotate_axis[1]]

        return rotate_shape, rotate_degree, rotate_axis

    def encodes(self, item: Path):
        rotate_shape, rotate_degree, rotate_axis = self._rotate_info(self.shape)

        if item.suffix.lower() in image_extensions:
            pipeline = v2.Compose([
                PILImageBW.create,
                v2.PILToTensor(),
                v2.ToDtype(torch.float16),
                lambda x: x / 255.0 * 2 - 1.0,
                v2.Resize(rotate_shape[-2:], antialias=True),
            ])

            image = pipeline(item)

            if self.dim == 3:
                image = image.unsqueeze(dim=1).expand(1, *rotate_shape)
                image = torch.rot90(image, k=rotate_degree, dims=[axis + 1 for axis in rotate_axis])
        else:
            pipeline = v2.Compose([
                v2.Grayscale(),
                v2.ToDtype(torch.float16),
                lambda x: x / 255.0 * 2 - 1.0,
                v2.Resize(rotate_shape[-2:], antialias=True),
            ])

            video = VideoReader(item)
            image = [pipeline(frame["data"]) for frame in video]

            frame_start = random.randint(0, len(image) - rotate_shape[0])
            image = image[frame_start : frame_start + rotate_shape[0]]

            image = torch.stack(image, dim=1)
            image = torch.rot90(image, k=rotate_degree, dims=[axis + 1 for axis in rotate_axis])

        return image
