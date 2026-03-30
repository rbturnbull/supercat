from pathlib import Path

import torch

from supercat.pretrain import PretrainMovieDataset, PretrainImagesDataset

K400_PATH = Path(__file__).parent/"k400"
IMAGENET_PATH = Path(__file__).parent/"imagenet"


def test_pretrain_movie_dataset():
    dataset = PretrainMovieDataset(K400_PATH, size=8)
    assert len(dataset) == 2, "Dataset should not be empty"
    for item in dataset:
         lr_t, hr_t = item

         assert lr_t.shape == hr_t.shape == (1, 8, 8, 8)
         assert torch.isfinite(lr_t).all()
         assert torch.isfinite(hr_t).all()
         assert hr_t.min() >= -1.0
         assert hr_t.max() <= 1.0


def test_pretrain_image_dataset():
    dataset = PretrainImagesDataset(IMAGENET_PATH, size=8)
    assert len(dataset) == 2, "Dataset should not be empty"
    for item in dataset:
         lr_t, hr_t = item

         assert lr_t.shape == hr_t.shape == (1, 8, 8)
         assert torch.isfinite(lr_t).all()
         assert torch.isfinite(hr_t).all()
         assert hr_t.min() >= -1.0
         assert hr_t.max() <= 1.0
