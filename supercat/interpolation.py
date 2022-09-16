from enum import Enum
import tricubic
import numpy as np

class InterpolationMethod(Enum):
    TRICUBIC = 0
    LINEAR = 1


def interpolate3D(image:np.ndarray, new_shape, method:InterpolationMethod = InterpolationMethod.TRICUBIC):
    interpolator = tricubic.tricubic(list(image), list(image.shape))
    upscaled = np.empty( new_shape )
    xs = np.linspace(0.0, image.shape[0]-1, num=new_shape[0])
    ys = np.linspace(0.0, image.shape[1]-1, num=new_shape[1])
    zs = np.linspace(0.0, image.shape[2]-1, num=new_shape[2])
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                upscaled[i,j,k] = interpolator.ip( [x,y,z] )

    return np.float32(upscaled)