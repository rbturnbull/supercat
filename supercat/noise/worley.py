from typing import Union
import numpy as np
import torch


class WorleyNoise:
    """
    Generates Worley Noise

    https://dl.acm.org/doi/10.1145/237170.237267
    """
    def __init__(self, shape, density:int, n:Union[int,None]=1, seed=None):
        """
        Args:
            shape (tuple): The shape of the output array.
            density (int): The number of points to use when generating the noise.
            n (int|None, optional): The 'n' in the formula to define the noise function. 
                The noise is generated from finding the distance to the nth closest point. 
                Defaults to 1.
                This must be between 1 and density (inclusive).
                If 'None' then it returns the distance to all points.
            seed (int, optional): The random seed to use. Defaults to None.
        """
        np.random.seed(seed)

        self.density = density
        self.shape = shape
        self.dims = len(self.shape)
        self.coords = [np.arange(s) for s in self.shape]
        self.points = None
        self.n = n

        if isinstance(n, int):
            assert n > 0, f"n must be greater than or equal to 0. Got {n}."
            assert n <= density, f"n must be less than density. Got {n} and {density}."
        
    def __call__(self):
        """    
        Adapted from:
                https://stackoverflow.com/a/65704227
                https://stackoverflow.com/q/65703414
        """
        self.points = np.random.rand(self.density, self.dims) 

        for i, size in enumerate(self.shape):
            self.points[:, i] *= size

        axes = list(range(1, self.dims+1))
        squared_d = sum(
            np.expand_dims(
                np.power(self.points[:, i, np.newaxis] - self.coords[i], 2), 
                axis=axes[:i]+axes[i+1:]
            )
            for i in range(self.dims)
        )

        if self.n == 1:
            return np.sqrt(squared_d.min(axis=0))
        elif self.n is None:
            return np.sqrt(squared_d)
        return  np.sqrt(np.sort(squared_d, axis=0)[self.n-1])


class WorleyNoiseTensor(WorleyNoise):
    def __call__(self, *args):
        """ Converts the output to a tensor and rescales it to be between -1 and 1. """
        x = super().__call__()
        x = torch.from_numpy(x).float()
        x = x/x.max()*2.0 - 1.0
        x = x.unsqueeze(0)
        
        return x

