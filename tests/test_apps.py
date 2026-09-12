from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, Mock, call

import pytest
import torch
import widit
import widitapp.training

from supercat import apps, data, metrics, models, utils


@pytest.fixture
def app(monkeypatch):
    # Prediction changes global gradient mode; restore it after each test.
    enabled = torch.is_grad_enabled()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "manual_seed", Mock())
    instance = apps.Supercat()
    yield instance
    torch.set_grad_enabled(enabled)


@pytest.fixture
def prediction_dependencies(app, monkeypatch):
    image = torch.zeros(1, 4, 4, 4)
    prediction = torch.full((4, 4, 4), 0.5)
    model = Mock(out_channels=1)
    dependencies = SimpleNamespace(
        image=image,
        prediction=prediction,
        model=model,
        read=Mock(return_value=image),
        load=Mock(return_value=model),
        wrap=Mock(),
        generate=Mock(return_value=prediction),
        write=Mock(),
        porosity=Mock(return_value=0.25),
    )
    monkeypatch.setattr(data, "read_image_as_tensor", dependencies.read)
    monkeypatch.setattr(widit, "load_model", dependencies.load)
    monkeypatch.setattr(models, "DiffusionPredictionModel", dependencies.wrap)
    monkeypatch.setattr(app, "generate_prediction", dependencies.generate)
    monkeypatch.setattr(utils, "write_image", dependencies.write)
    monkeypatch.setattr(metrics, "calc_porosity", dependencies.porosity)
    return dependencies


@pytest.fixture
def prediction_options(tmp_path):
    return dict(
        input=tmp_path / "input.mat",
        output=tmp_path / "nested" / "output.tif",
        checkpoint=tmp_path / "checkpoint.pt",
        size=4,
        overlap=1,
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("augment", [False, True])
def test_datasets_selects_builder(app, monkeypatch, tmp_path, dim, augment):
    builders = {2: Mock(return_value=(object(), object())), 3: Mock()}
    monkeypatch.setattr(apps, "build_datasets2D", builders[2])
    monkeypatch.setattr(apps, "build_datasets3D", builders[3])
    result = app.datasets(dim=dim, deeprock=tmp_path, scale=2, augment=augment)
    assert result is builders[dim].return_value
    builders[dim].assert_called_once_with(
        deeprock=tmp_path, scale=2, train_augment=augment
    )
    builders[5 - dim].assert_not_called()


def test_datasets_resolves_option_defaults(app, monkeypatch, tmp_path):
    builder = Mock()
    monkeypatch.setattr(apps, "build_datasets3D", builder)
    app.datasets(deeprock=tmp_path)
    builder.assert_called_once_with(deeprock=tmp_path, scale=4, train_augment=True)


@pytest.mark.parametrize("diffusion", [False, True])
@pytest.mark.parametrize("cuda", [False, True])
def test_predict_loads_model_and_saves_result(
    app, prediction_dependencies, prediction_options, monkeypatch, diffusion, cuda
):
    deps = prediction_dependencies
    deps.model.out_channels = 2 if diffusion else 1
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    app.predict(**prediction_options)

    deps.read.assert_called_once_with(prediction_options["input"], size=(4, 4, 4))
    deps.load.assert_called_once_with(prediction_options["checkpoint"])
    selected_model = deps.wrap.return_value if diffusion else deps.model
    if diffusion:
        deps.wrap.assert_called_once_with(deps.model, 250)
        torch.manual_seed.assert_called_once_with(42)
    else:
        deps.wrap.assert_not_called()
        torch.manual_seed.assert_not_called()
    device = "cuda" if cuda else "cpu"
    selected_model.to.assert_called_once_with(device=device)
    selected_model.eval.assert_called_once_with()
    deps.generate.assert_called_once_with(
        input_image=deps.image,
        model=selected_model,
        size_i=4,
        size_j=4,
        size_k=4,
        overlap_i=1,
        overlap_j=1,
        overlap_k=1,
        spatial_dims=3,
        device=device,
        single_crop=False,
    )
    deps.write.assert_called_once_with(deps.prediction, prediction_options["output"])
    assert prediction_options["output"].parent.is_dir()


def test_predict_forwards_axis_and_sampling_overrides(
    app, prediction_dependencies, prediction_options
):
    deps = prediction_dependencies
    deps.model.out_channels = 2
    deps.read.return_value = torch.zeros(1, 6, 5, 4)
    app.predict(
        **prediction_options,
        size_i=4,
        size_j=5,
        size_k=6,
        overlap_i=2,
        overlap_j=3,
        overlap_k=4,
        num_sampling_steps=10,
        seed=7,
        single_crop=True,
    )
    deps.read.assert_called_once_with(prediction_options["input"], size=(6, 5, 4))
    deps.wrap.assert_called_once_with(deps.model, 10)
    torch.manual_seed.assert_called_once_with(7)
    kwargs = deps.generate.call_args.kwargs
    assert [kwargs[f"size_{axis}"] for axis in "ijk"] == [4, 5, 6]
    assert [kwargs[f"overlap_{axis}"] for axis in "ijk"] == [2, 3, 4]
    assert kwargs["single_crop"] is True


def test_predict_accepts_2d_input(app, prediction_dependencies, prediction_options):
    prediction_dependencies.read.return_value = torch.zeros(1, 4, 4)
    app.predict(**prediction_options)
    assert prediction_dependencies.generate.call_args.kwargs["spatial_dims"] == 2


@pytest.mark.parametrize("command", ["predict", "porosity_distribution"])
@pytest.mark.parametrize("missing", ["input", "output", "checkpoint"])
def test_prediction_requires_paths(
    app, prediction_dependencies, prediction_options, command, missing
):
    prediction_options[missing] = None
    with pytest.raises(AssertionError, match="Must provide"):
        getattr(app, command)(**prediction_options)
    prediction_dependencies.load.assert_not_called()
    prediction_dependencies.generate.assert_not_called()
    prediction_dependencies.write.assert_not_called()


@pytest.mark.parametrize("command", ["predict", "porosity_distribution"])
@pytest.mark.parametrize(
    "shape,message",
    [
        ((4, 4), "Input image must have 3 or 4 dimensions"),
        ((1, 4, 4, 5), "i dimension"),
        ((1, 4, 5, 4), "j dimension"),
        ((1, 5, 4, 4), "k dimension"),
    ],
)
def test_prediction_rejects_invalid_shape(
    app, prediction_dependencies, prediction_options, command, shape, message
):
    prediction_dependencies.read.return_value = torch.zeros(shape)
    with pytest.raises(AssertionError, match=message):
        getattr(app, command)(**prediction_options)
    prediction_dependencies.load.assert_not_called()
    prediction_dependencies.generate.assert_not_called()


def test_porosity_returns_and_prints_measurement(
    app, prediction_dependencies, tmp_path, capsys
):
    deps = prediction_dependencies
    path = tmp_path / "image.mat"
    assert app.porosity(input=path) == 0.25
    deps.read.assert_called_once_with(path)
    deps.porosity.assert_called_once_with(deps.image)
    assert "Porosity: 0.25" in capsys.readouterr().out


def test_porosity_distribution_writes_consecutive_seeds(
    app, prediction_dependencies, prediction_options
):
    deps = prediction_dependencies
    deps.model.out_channels = 2
    deps.porosity.side_effect = [0.1, 0.2, 0.3]
    app.porosity_distribution(
        **prediction_options, count=3, seed=10, num_sampling_steps=20
    )
    assert (
        prediction_options["output"].read_text()
        == "seed,porosity\n10,0.1\n11,0.2\n12,0.3\n"
    )
    deps.read.assert_called_once_with(prediction_options["input"], size=(4, 4, 4))
    deps.wrap.assert_called_once_with(deps.model, 20)
    assert torch.manual_seed.call_args_list == [call(10), call(11), call(12)]
    assert deps.generate.call_count == deps.porosity.call_count == 3
    deps.porosity.assert_called_with(deps.prediction)
    deps.write.assert_not_called()


def test_porosity_distribution_resumes_existing_csv(
    app, prediction_dependencies, prediction_options, capsys
):
    deps = prediction_dependencies
    deps.model.out_channels = 2
    output = prediction_options["output"]
    output.parent.mkdir()
    original = "seed,porosity\n\nnot-a-seed,0.1\n10,0.1\n12,0.3\n"
    output.write_text(original)
    app.porosity_distribution(**prediction_options, count=3, seed=10)
    assert output.read_text() == original + "11,0.25\n"
    torch.manual_seed.assert_called_once_with(11)
    deps.generate.assert_called_once()
    assert "Skipping seed 10" in capsys.readouterr().out


def test_porosity_distribution_overwrites_existing_csv(
    app, prediction_dependencies, prediction_options
):
    prediction_dependencies.model.out_channels = 2
    output = prediction_options["output"]
    output.parent.mkdir()
    output.write_text("seed,porosity\n42,0.9\n")
    app.porosity_distribution(**prediction_options, count=1, overwrite=True)
    assert output.read_text() == "seed,porosity\n42,0.25\n"
    prediction_dependencies.generate.assert_called_once()


def test_porosity_distribution_rejects_regression_model(
    app, prediction_dependencies, prediction_options
):
    with pytest.raises(AssertionError, match="Model must be a diffusion model"):
        app.porosity_distribution(**prediction_options)
    prediction_dependencies.generate.assert_not_called()
    assert not prediction_options["output"].exists()


def test_train_delegates_to_mocked_training_backend(app, monkeypatch, tmp_path):
    model = Mock()
    training_loader, validation_loader = object(), object()
    monkeypatch.setattr(app, "model", Mock(return_value=model))
    monkeypatch.setattr(
        app, "dataloaders", Mock(return_value=(training_loader, validation_loader))
    )
    train = Mock()
    monkeypatch.setattr(widitapp.training, "train", train)
    app.train(
        epochs=2,
        learning_rate=0.01,
        results_dir=tmp_path,
        use_diffusion=False,
        dim=2,
        deeprock=tmp_path,
        scale=2,
        run_name="test",
    )
    app.model.assert_called_once_with(
        use_diffusion=False,
        dim=2,
        deeprock=tmp_path,
        scale=2,
        augment=True,
        porosity_loss_weight=0.0,
        porosity_temperature=0.05,
    )
    app.dataloaders.assert_called_once_with(
        dim=2,
        deeprock=tmp_path,
        scale=2,
        augment=True,
        porosity_loss_weight=0.0,
        porosity_temperature=0.05,
    )
    train.assert_called_once_with(
        model=model,
        training_dataloader=training_loader,
        validation_dataloader=validation_loader,
        results_dir=tmp_path,
        use_diffusion=False,
        learning_rate=0.01,
        epochs=2,
        log_every=100,
        run_name="test",
        wandb_logging=False,
        wandb_project="Supercat",
        loss_fn=ANY,
        metrics=ANY,
    )


@pytest.fixture
def tile_options(monkeypatch):
    # Silence progress rendering while exercising real tile generation and blending.
    monkeypatch.setattr(apps, "Progress", MagicMock())
    return dict(
        size_i=4,
        size_j=4,
        size_k=4,
        overlap_i=1,
        overlap_j=1,
        overlap_k=1,
        device="cpu",
        single_crop=False,
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_generate_prediction_single_tile_preserves_identity(app, tile_options, dim):
    image = torch.arange(4**dim, dtype=torch.float32).reshape((1,) + (4,) * dim)
    model = Mock(side_effect=lambda crop: crop)
    result = app.generate_prediction(
        input_image=image, model=model, spatial_dims=dim, **tile_options
    )
    torch.testing.assert_close(result, image[0])
    model.assert_called_once()
    assert model.call_args.args[0].shape == (1, 1) + (4,) * dim


@pytest.mark.parametrize("dim", [2, 3])
def test_generate_prediction_skips_nan_tiles(app, tile_options, dim):
    image = torch.full((1,) + (4,) * dim, float("nan"))
    model = Mock()
    result = app.generate_prediction(
        input_image=image, model=model, spatial_dims=dim, **tile_options
    )
    assert result.shape == (4,) * dim
    assert torch.isnan(result).all()
    model.assert_not_called()


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="Output tile slices include the channel axis (apps.py generate_prediction)",
)
@pytest.mark.parametrize("dim", [2, 3])
def test_generate_prediction_multiple_tiles_preserves_identity(app, tile_options, dim):
    image = torch.arange(7**dim, dtype=torch.float32).reshape((1,) + (7,) * dim)
    result = app.generate_prediction(
        input_image=image, model=torch.nn.Identity(), spatial_dims=dim, **tile_options
    )
    torch.testing.assert_close(result, image[0])


@pytest.mark.parametrize("dim", [2, 3])
def test_generate_prediction_single_crop_with_one_tile(app, tile_options, dim):
    tile_options["single_crop"] = True
    image = torch.ones((1,) + (4,) * dim)
    model = Mock(side_effect=lambda crop: crop)
    result = app.generate_prediction(
        input_image=image, model=model, spatial_dims=dim, **tile_options
    )
    torch.testing.assert_close(result, image[0])
    model.assert_called_once()


@pytest.mark.parametrize("dim", [2, 3])
def test_generate_prediction_empty_image(app, tile_options, dim):
    image = torch.empty((1,) + (0,) * dim)
    model = Mock()
    result = app.generate_prediction(
        input_image=image, model=model, spatial_dims=dim, **tile_options
    )
    assert result.shape == (0,) * dim
    model.assert_not_called()


def test_porosity_distribution_forwards_axis_sizes(
    app, prediction_dependencies, prediction_options
):
    deps = prediction_dependencies
    deps.model.out_channels = 2
    deps.read.return_value = torch.zeros(1, 6, 5, 4)
    app.porosity_distribution(
        **prediction_options, size_i=4, size_j=5, size_k=6, count=1
    )
    deps.read.assert_called_once_with(prediction_options["input"], size=(6, 5, 4))
    forwarded = deps.generate.call_args.kwargs
    assert [forwarded[f"size_{axis}"] for axis in "ijk"] == [4, 5, 6]
