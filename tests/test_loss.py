from supercat.noise.worley import WorleyNoise, WorleyNoiseTensor
import torch
import torch.nn.functional as F
from supercat.loss import EdgeLoss

def test_edge_loss2d():
    seed = 42
    density = 23
    shape = (100, 100)
    wn = WorleyNoiseTensor(shape, density, seed=seed)
    n_samples = 4

    target = torch.zeros( (n_samples,) + shape )
    for i in range(n_samples):
        target[i] = wn()

    loss_module = EdgeLoss(percentile=50, alpha=0.5)
    loss = loss_module( target, target )
    assert loss < 0.001

    assert 0.0034 < loss_module( target+0.1, target ) < F.smooth_l1_loss(target+0.1, target)


def test_edge_loss3d():
    seed = 42
    density = 23
    shape = (32, 32, 32)
    wn = WorleyNoiseTensor(shape, density, seed=seed)
    n_samples = 2

    target = torch.zeros( (n_samples,) + shape )
    for i in range(n_samples):
        target[i] = wn()

    loss_module = EdgeLoss(percentile=50, alpha=0.5)
    loss = loss_module( target, target )
    assert loss < 0.001
    assert 0.0036 < loss_module( target+0.1, target ) < F.smooth_l1_loss(target+0.1, target)
