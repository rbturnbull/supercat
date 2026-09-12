from unittest.mock import Mock

import pytest
import torch
from torch import nn

from supercat import models


@pytest.fixture
def diffusion_factory(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(models, "create_diffusion", factory)
    return factory


@pytest.mark.parametrize("sampling_steps", [None, 10, 100])
def test_prediction_model_configures_sampling_steps(diffusion_factory, sampling_steps):
    model = nn.Identity()
    if sampling_steps is None:
        prediction = models.DiffusionPredictionModel(model)
        expected_steps = "250"
    else:
        prediction = models.DiffusionPredictionModel(model, sampling_steps)
        expected_steps = str(sampling_steps)

    diffusion_factory.assert_called_once_with(expected_steps)
    assert prediction.diffusion is diffusion_factory.return_value
    assert prediction.diffusion_model is model


@pytest.mark.parametrize("shape", [(2, 1, 8, 10), (2, 1, 4, 6, 8)])
def test_forward_passes_conditioning_and_returns_samples(diffusion_factory, shape):
    model = nn.Identity()
    prediction = models.DiffusionPredictionModel(model)
    inputs = torch.zeros(shape)
    original = inputs.clone()
    samples = torch.full(shape, 0.5)
    sampler = diffusion_factory.return_value.p_sample_loop
    sampler.return_value = samples

    result = prediction(inputs)

    assert result is samples
    sampler.assert_called_once()
    args, kwargs = sampler.call_args
    forward, noise_shape, noise = args
    assert forward == model.forward
    assert noise_shape == inputs.shape
    assert noise.shape == inputs.shape
    assert noise.device == inputs.device
    assert torch.isfinite(noise).all()
    assert noise.data_ptr() != inputs.data_ptr()
    assert kwargs["model_kwargs"]["conditioned"] is inputs
    assert kwargs["clip_denoised"] is True
    assert kwargs["progress"] is True
    assert kwargs["device"] == inputs.device
    torch.testing.assert_close(inputs, original)


def test_forward_noise_is_seeded_and_changes_between_calls(diffusion_factory):
    prediction = models.DiffusionPredictionModel(nn.Identity())
    inputs = torch.zeros(1, 1, 8, 8)
    sampler = diffusion_factory.return_value.p_sample_loop
    sampler.side_effect = lambda forward, shape, noise, **kwargs: noise

    # Keep this test's seed changes isolated from other tests.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        first = prediction(inputs)
        second = prediction(inputs)
        torch.manual_seed(42)
        repeated = prediction(inputs)

    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert not torch.equal(first, second)


def test_prediction_model_propagates_eval_and_dtype_to_wrapped_model(diffusion_factory):
    model = nn.Conv2d(1, 2, kernel_size=1)
    prediction = models.DiffusionPredictionModel(model)

    assert list(prediction.parameters()) == list(model.parameters())
    prediction.eval()
    assert not model.training
    prediction.train()
    assert model.training
    prediction.to(dtype=torch.float64)
    assert model.weight.dtype == torch.float64
