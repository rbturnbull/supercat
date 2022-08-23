import math
from fastai.metrics import mse

def psnr(input, target, max=2.0):
    """
    A metric to calculate the peak signal-to-noise ratio.

    It is given in eq. 3 of 
    PSNR = 10 \log_{10}(\frac{I^2}{L_{2_{Loss}}})
    where I = 2 because the HR and SR pixel values are between [-1,1].
    """
    L2 = mse(input, target)
    return 10 * math.log10(max**2/L2)

