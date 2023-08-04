from .worley import WorleyNoiseTensor
from .fractal import FractalNoiseTensor


class NoiseTensorGenerator():
    def __init__(self, shape):
        self.fractal = FractalNoiseTensor(shape)
        self.worley = WorleyNoiseTensor(shape)
        self.shape = shape
    
    def __call__(self, *args, **kwargs):
        return self.fractal(*args, **kwargs) if np.random.rand() < 0.5 else self.worley(*args, **kwargs)


class NoiseSR(WorleySR):
    def build_generator(self, shape):
        return NoiseTensorGenerator(shape=shape)


if __name__ == "__main__":
    NoiseSR.main()
