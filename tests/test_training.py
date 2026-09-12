"""Exercise WiDiTApp hooks with real criteria and mocked training/model backends."""

from unittest.mock import Mock
import inspect

import pytest
import torch
from cluey.testing import CliRunner
from widitapp import WiDiTApp
from widitapp import training as backend

from supercat.apps import Supercat
from supercat.pretrain import SupercatPretrainImage, SupercatPretrainMovie
from supercat.metrics import PorosityLoss
from supercat.training import AuxiliaryLoss

APPS = [Supercat, SupercatPretrainImage, SupercatPretrainMovie]


@pytest.fixture
def images():
    target = torch.tensor([[[[-1.0, -0.5], [0.5, 1.0]]], [[[-0.8, -0.2], [0.4, 0.9]]]])
    return target + 0.1, target


def test_installed_widitapp_has_both_hooks():
    assert callable(WiDiTApp.loss)
    assert callable(WiDiTApp.metrics)
    assert "loss" in WiDiTApp.train.methods_to_call
    assert "metrics" in WiDiTApp.train.methods_to_call


@pytest.mark.parametrize("app_class", APPS)
@pytest.mark.parametrize("diffusion", [False, True])
def test_zero_weight_returns_exact_parent_and_skips_auxiliary(
    app_class, diffusion, monkeypatch
):
    import supercat.training as adapters

    app = app_class()
    parent = None if diffusion else torch.nn.SmoothL1Loss()
    parent_factory = Mock(return_value=parent)
    parent_factory.__name__ = "loss"
    parent_factory.__signature__ = inspect.signature(WiDiTApp.loss.func)
    monkeypatch.setattr(WiDiTApp.loss, "func", parent_factory)
    auxiliary = Mock(side_effect=AssertionError("Auxiliary must not be constructed"))
    monkeypatch.setattr(adapters, "PorosityLoss", auxiliary)
    assert (
        app.loss(use_diffusion=diffusion, loss_fn="smoothl1", porosity_loss_weight=0)
        is parent
    )
    assert parent_factory.call_args.kwargs == {
        "loss_fn": "smoothl1",
        "use_diffusion": diffusion,
    }
    auxiliary.assert_not_called()


@pytest.mark.parametrize("app_class", APPS)
@pytest.mark.parametrize("diffusion", [False, True])
def test_hook_reuses_porosity_and_combines_scalar_objective(
    app_class, diffusion, images
):
    prediction, target = images
    app = app_class()
    criterion = app.loss(
        use_diffusion=diffusion,
        loss_fn="smoothl1",
        porosity_loss_weight=0.3,
        porosity_temperature=0.2,
    )
    assert isinstance(criterion, AuxiliaryLoss)
    assert isinstance(criterion.auxiliary, PorosityLoss)
    expected = 0.3 * PorosityLoss(temperature=0.2)(prediction, target)
    if not diffusion:
        assert isinstance(criterion.parent, torch.nn.SmoothL1Loss)
        expected += torch.nn.functional.smooth_l1_loss(prediction, target)
    else:
        assert criterion.parent is None
    result = criterion(prediction, target)
    assert result.ndim == 0
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("diffusion", [False, True])
def test_auxiliary_contributes_gradients_in_both_modes(diffusion, images):
    prediction, target = images
    prediction = prediction.clone().requires_grad_()
    criterion = Supercat().loss(
        use_diffusion=diffusion,
        loss_fn="smoothl1",
        porosity_loss_weight=0.3,
        porosity_temperature=0.2,
    )
    actual = torch.autograd.grad(criterion(prediction, target), prediction)[0]
    reference = 0.3 * PorosityLoss(temperature=0.2)(prediction, target)
    if not diffusion:
        reference += torch.nn.functional.smooth_l1_loss(prediction, target)
    expected = torch.autograd.grad(reference, prediction)[0]
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0


def test_combiner_means_unreduced_parent_and_auxiliary(images):
    prediction, target = images
    parent = torch.nn.SmoothL1Loss(reduction="none")
    auxiliary = PorosityLoss(reduction="none")
    result = AuxiliaryLoss(parent, auxiliary, 0.2)(prediction, target)
    assert result.ndim == 0
    torch.testing.assert_close(
        result,
        parent(prediction, target).mean() + 0.2 * auxiliary(prediction, target).mean(),
    )


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_invalid_porosity_loss_weight_is_rejected(weight):
    with pytest.raises(ValueError, match="porosity_loss_weight"):
        Supercat().loss(porosity_loss_weight=weight)


@pytest.mark.parametrize("app_class", APPS)
@pytest.mark.parametrize("diffusion", [False, True])
def test_metrics_keep_parent_and_compute_independent_batch_means(
    app_class, diffusion, images
):
    prediction, target = images
    metrics = app_class().metrics(use_diffusion=diffusion, porosity_temperature=0.2)
    assert set(metrics) == (
        {"porosity_loss"} if diffusion else {"mse", "smoothl1", "porosity_loss"}
    )
    metric = metrics["porosity_loss"]
    assert isinstance(metric, PorosityLoss)
    with torch.no_grad():
        first = metric(prediction, target)
        expected = torch.stack(
            [metric(p[None], t[None]) for p, t in zip(prediction, target)]
        ).mean()
        torch.testing.assert_close(first, expected)
        assert metric(target, target).item() == 0
        torch.testing.assert_close(metric(prediction, target), first)
    assert first.ndim == 0 and not first.requires_grad


def test_metrics_preserve_custom_parent_entries(monkeypatch):
    app = Supercat()
    inherited = torch.nn.L1Loss()
    parent_factory = Mock(return_value={"custom_parent": inherited})
    parent_factory.__name__ = "metrics"
    parent_factory.__signature__ = inspect.signature(WiDiTApp.metrics.func)
    monkeypatch.setattr(WiDiTApp.metrics, "func", parent_factory)
    metrics = app.metrics(use_diffusion=False)
    assert metrics["custom_parent"] is inherited
    assert isinstance(metrics["porosity_loss"], PorosityLoss)


@pytest.mark.parametrize("app_class", APPS)
@pytest.mark.parametrize("diffusion", [False, True])
@pytest.mark.parametrize("cli", [False, True])
def test_train_forwards_inherited_and_custom_options(
    app_class, diffusion, cli, monkeypatch
):
    app = app_class()
    app.model = Mock(return_value=torch.nn.Identity())
    app.dataloaders = Mock(return_value=(object(), None))
    train = Mock()
    monkeypatch.setattr(backend, "train", train)
    if cli:
        result = CliRunner().invoke(
            app.tools_app,
            [
                "train",
                "--use-diffusion" if diffusion else "--no-use-diffusion",
                "--loss-fn",
                "smoothl1",
                "--porosity-loss-weight",
                "0.4",
                "--porosity-temperature",
                "0.2",
            ],
        )
        assert result.exit_code == 0, (result.output, result.exception)
    else:
        app.train(
            use_diffusion=diffusion,
            loss_fn="smoothl1",
            porosity_loss_weight=0.4,
            porosity_temperature=0.2,
        )
    train.assert_called_once()
    options = train.call_args.kwargs
    key = "diffusion_loss_fn" if diffusion else "loss_fn"
    assert options["use_diffusion"] is diffusion
    assert ("loss_fn" if diffusion else "diffusion_loss_fn") not in options
    criterion = options[key]
    assert criterion.weight == 0.4
    assert criterion.auxiliary.temperature == 0.2 and not criterion.auxiliary.hard_mask
    assert (
        (criterion.parent is None)
        if diffusion
        else isinstance(criterion.parent, torch.nn.SmoothL1Loss)
    )
    metric = options["metrics"]["porosity_loss"]
    assert metric.hard_mask is True


@pytest.mark.parametrize("diffusion", [False, True])
def test_validation_hook_reports_unclipped_predictions_without_changing_objective(
    diffusion, images
):
    from types import SimpleNamespace

    _, target = images
    prediction = target + 1.5  # Explicitly outside [-1, 1].
    app = Supercat()
    criterion = app.loss(use_diffusion=diffusion, porosity_loss_weight=0)
    metric = app.metrics(use_diffusion=diffusion)["porosity_loss"]
    observed = []
    metric.register_forward_pre_hook(
        lambda module, args: observed.append((args[0].clone(), torch.is_grad_enabled()))
    )
    model = Mock()
    model.return_value = prediction
    diffusion_backend = Mock(num_timesteps=10)
    diffusion_backend.training_losses.return_value = {
        "mse": torch.tensor([2.0, 4.0]),
        "pred_xstart": prediction,
    }
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        is_main_process=True,
        reduce=lambda tensor, reduction: tensor,
    )
    result = backend._run_validation_loop(
        accelerator,
        model,
        diffusion_backend,
        [(target, target)],
        torch.device("cpu"),
        torch.float32,
        diffusion,
        criterion,
        extra_criteria={"porosity_loss": metric},
    )
    assert result["loss"] == pytest.approx(
        3.0 if diffusion else criterion(prediction, target).item()
    )
    assert result["porosity_loss"] == pytest.approx(
        PorosityLoss(hard_mask=True)(prediction, target).item()
    )
    torch.testing.assert_close(observed[0][0], prediction)
    assert observed[0][1] is False
    _, payload = backend.build_val_log_payload(
        result, use_diffusion=diffusion, epoch=1, train_steps=2
    )
    assert payload["val/porosity_loss"] == result["porosity_loss"]
    assert payload["val/loss"] == result["loss"]


def test_real_diffusion_adds_auxiliary_once_and_preserves_gradients(images):
    from widitapp.diffusion import create_diffusion

    _, target = images
    diffusion = create_diffusion("10")
    value = torch.nn.Parameter(torch.tensor(0.05))

    def model(x, timestep, **kwargs):
        return torch.cat([torch.ones_like(x) * value, torch.zeros_like(x)], dim=1)

    timesteps = torch.zeros(len(target), dtype=torch.long)
    noise = torch.full_like(target, 0.1)
    baseline = diffusion.training_losses(model, target, timesteps, noise=noise)
    criterion = Supercat().loss(
        use_diffusion=True, porosity_loss_weight=0.3, porosity_temperature=0.2
    )
    result = diffusion.training_losses(
        model,
        target,
        timesteps,
        noise=noise,
        image_loss_fn=criterion,
        return_pred_xstart=True,
    )
    auxiliary = criterion(result["pred_xstart"], target)
    torch.testing.assert_close(result["loss"], baseline["loss"] + auxiliary)
    assert result["pred_xstart"].max() > 1  # Clean estimate must not be clipped.
    gradient = torch.autograd.grad(result["image_loss"].mean(), value)[0]
    assert torch.isfinite(gradient) and gradient.abs() > 0


@pytest.mark.parametrize("app_class", APPS)
def test_cli_has_temperature_but_no_mask_switch(app_class):
    app = app_class()
    help_result = CliRunner().invoke(app.tools_app, ["train", "--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "--porosity-temperature" in help_result.output
    assert "--porosity-hard-mask" not in help_result.output
    result = CliRunner().invoke(app.tools_app, ["train", "--porosity-hard-mask"])
    assert result.exit_code != 0


@pytest.mark.parametrize("app_class", APPS)
@pytest.mark.parametrize("diffusion", [False, True])
def test_training_mask_is_soft_and_metric_is_hard_independent_of_temperature(
    app_class, diffusion, images
):
    prediction, target = images
    app = app_class()
    losses, measurements = [], []
    for temperature in [0.05, 0.5]:
        criterion = app.loss(
            use_diffusion=diffusion,
            porosity_loss_weight=1,
            porosity_temperature=temperature,
        )
        assert criterion.auxiliary.hard_mask is False
        losses.append(criterion.auxiliary(prediction, target))
        metric = app.metrics(use_diffusion=diffusion, porosity_temperature=temperature)[
            "porosity_loss"
        ]
        assert metric.hard_mask is True
        with torch.no_grad():
            measurements.append(metric(prediction, target))
    assert not torch.isclose(losses[0], losses[1])
    torch.testing.assert_close(measurements[0], measurements[1], rtol=0, atol=0)
    torch.testing.assert_close(
        measurements[0], PorosityLoss(hard_mask=True)(prediction, target)
    )
