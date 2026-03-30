from pathlib import Path

import torch

from supercat.pretrain import PretrainMovieDataset

K400_PATH = Path(__file__).parent/"k400"


def test_pretrain_movie_dataset():
    dataset = PretrainMovieDataset(K400_PATH, size=8)
    item = dataset[0]
    lr_t, hr_t = item

    assert lr_t.shape == hr_t.shape == (1, 8, 8, 8)
    assert torch.isfinite(lr_t).all()
    assert torch.isfinite(hr_t).all()
    assert hr_t.min() >= -1.0
    assert hr_t.max() <= 1.0
