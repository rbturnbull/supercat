import numpy as np
import pytest
import torch

from supercat.metrics import calc_porosity


def test_calc_porosity_splits_bimodal_image_evenly():
    data = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0], dtype=np.float32)
    assert calc_porosity(data) == pytest.approx(0.5)


def test_calc_porosity_counts_void_fraction_of_volume():
    volume = np.ones((4, 4, 5), dtype=np.float32)
    volume[0] = 0.0
    assert calc_porosity(volume) == pytest.approx(0.25)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_calc_porosity_accepts_tensors(dtype):
    values = [0.0, 0.1, 0.2, 0.8, 0.9, 1.0]
    tensor = torch.tensor(values, dtype=dtype)
    expected = calc_porosity(np.array(values, dtype=np.float32))
    assert calc_porosity(tensor) == pytest.approx(expected)
    torch.testing.assert_close(tensor, torch.tensor(values, dtype=dtype))


def test_calc_porosity_of_uniform_data_is_zero():
    assert calc_porosity(np.zeros((3, 3), dtype=np.float32)) == 0.0


@pytest.mark.parametrize("spatial_shape", [(2, 2), (1, 2, 2)])
@pytest.mark.parametrize("hard_mask", [False, True])
def test_porosity_loss_identical_images_have_zero_loss(spatial_shape, hard_mask):
    from supercat.metrics import PorosityLoss

    target = torch.tensor([-1.0, -1.0, 1.0, 1.0]).reshape((1, 1) + spatial_shape)
    prediction = target.clone().requires_grad_()
    loss = PorosityLoss(hard_mask=hard_mask)(prediction, target)
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_porosity_loss_matches_relative_error_formula():
    from supercat.metrics import PorosityLoss

    target = torch.tensor([[[[-1.0, -1.0], [1.0, 1.0]]]])
    prediction = torch.tensor([[[[-1.0, 1.0], [1.0, 1.0]]]], requires_grad=True)
    # Target porosity = 1/2; predicted porosity = 1/4: ratio 0.5, error -0.5.
    loss = PorosityLoss(hard_mask=True)(prediction, target)
    assert loss.item() == pytest.approx(0.495)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert (
        prediction.grad[0, 0, 0, 0] > 0
    )  # Gradient descent increases pore membership.


def test_porosity_loss_uses_ground_truth_threshold_for_prediction():
    from supercat.metrics import PorosityLoss

    target = torch.tensor([[[[-1.0, -1.0], [1.0, 1.0]]]])
    prediction = target + 0.5
    # Independent Otsu thresholds would hide this offset and produce zero error.
    assert PorosityLoss(hard_mask=True)(prediction, target).item() == pytest.approx(
        0.995
    )


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_porosity_loss_reduces_per_sample_errors(reduction):
    from supercat.metrics import PorosityLoss

    target = torch.tensor([[[[-1.0, -1.0], [1.0, 1.0]]]]).repeat(2, 1, 1, 1)
    prediction = target.clone()
    prediction[1, 0, 0, 1] = 1
    actual = PorosityLoss(hard_mask=True, reduction=reduction)(prediction, target)
    expected = torch.tensor([0.0, 0.495])
    if reduction != "none":
        expected = getattr(expected, reduction)()
    torch.testing.assert_close(actual, expected)


def test_porosity_loss_soft_surrogate_passes_gradcheck_and_detaches_target():
    from supercat.metrics import PorosityLoss

    prediction = torch.tensor(
        [[[[-0.9, -0.4], [0.1, 0.8]]]], dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor(
        [[[[-1.0, -0.7], [0.5, 1.0]]]], dtype=torch.float64, requires_grad=True
    )
    criterion = PorosityLoss(temperature=0.2)
    assert torch.autograd.gradcheck(lambda x: criterion(x, target), (prediction,))
    criterion(prediction, target).backward()
    assert target.grad is None
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) > 0


@pytest.mark.parametrize("hard_mask", [False, True])
def test_porosity_loss_is_finite_for_constant_targets(hard_mask):
    from supercat.metrics import PorosityLoss

    target = torch.ones(1, 1, 2, 2)
    criterion = PorosityLoss(hard_mask=hard_mask)
    assert criterion(target.clone(), target).item() == 0
    prediction = torch.zeros_like(target, requires_grad=True)
    loss = criterion(prediction, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_porosity_loss_accumulates_low_precision_inputs_in_float32(dtype):
    from supercat.metrics import PorosityLoss

    target = torch.tensor([[[[-1.0, -1.0], [1.0, 1.0]]]], dtype=dtype)
    prediction = (target + 0.1).requires_grad_()
    loss = PorosityLoss()(prediction, target)
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize(
    "options",
    [
        {"temperature": 0},
        {"temperature": float("nan")},
        {"eps": 0},
        {"eps": 2},
        {"beta": -1},
        {"beta": float("inf")},
        {"reduction": "invalid"},
    ],
)
def test_porosity_loss_rejects_invalid_configuration(options):
    from supercat.metrics import PorosityLoss

    with pytest.raises(ValueError):
        PorosityLoss(**options)


@pytest.mark.parametrize("shape", [(1, 2, 2), (1, 2, 2, 2), (0, 1, 2, 2)])
def test_porosity_loss_rejects_invalid_input_shape(shape):
    from supercat.metrics import PorosityLoss

    with pytest.raises(ValueError):
        PorosityLoss()(torch.zeros(shape), torch.zeros(shape))


def test_porosity_loss_rejects_mismatched_shapes():
    from supercat.metrics import PorosityLoss

    with pytest.raises(ValueError, match="matching shapes"):
        PorosityLoss()(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 3, 3))


def test_porosity_loss_rejects_integer_inputs():
    from supercat.metrics import PorosityLoss

    target = torch.zeros(1, 1, 2, 2, dtype=torch.int64)
    with pytest.raises(TypeError, match="floating-point"):
        PorosityLoss()(target, target)


@pytest.mark.parametrize("invalid_target", [False, True])
def test_porosity_loss_rejects_nonfinite_values(invalid_target):
    from supercat.metrics import PorosityLoss

    prediction, target = torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)
    (target if invalid_target else prediction)[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite values"):
        PorosityLoss()(prediction, target)


def test_porosity_loss_combines_with_pixel_smooth_l1():
    from supercat.metrics import PorosityLoss

    target = torch.tensor([[[[-1.0, -0.5], [0.5, 1.0]]]])
    prediction = torch.zeros_like(target, requires_grad=True)
    loss = torch.nn.functional.smooth_l1_loss(
        prediction, target
    ) + 0.001 * PorosityLoss()(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) > 0


def test_porosity_loss_rejects_mismatched_devices_without_running_otsu():
    from supercat.metrics import PorosityLoss

    prediction = torch.zeros(1, 1, 2, 2)
    target = torch.empty(1, 1, 2, 2, device="meta")
    with pytest.raises(ValueError, match="same device"):
        PorosityLoss()(prediction, target)


@pytest.mark.parametrize("hard_mask", [False, True])
def test_precomputed_porosity_matches_image_loss_without_calling_otsu(
    monkeypatch, hard_mask
):
    from supercat import metrics
    from unittest.mock import Mock

    target = torch.tensor([[[[-1.0, -0.5], [0.5, 1.0]]], [[[-0.8, 0.1], [0.2, 0.9]]]])
    prediction = (target + 0.1).requires_grad_()
    criterion = metrics.PorosityLoss(temperature=0.2, hard_mask=hard_mask)
    expected = criterion(prediction, target)
    references = [
        metrics.porosity_reference(sample, temperature=0.2, hard_mask=hard_mask)
        for sample in target
    ]
    thresholds, porosities = [
        torch.stack(values).requires_grad_() for values in zip(*references)
    ]
    otsu = Mock(side_effect=AssertionError("Otsu must not run in the loss"))
    monkeypatch.setattr(metrics.filters, "threshold_otsu", otsu)
    actual = criterion(prediction, thresholds, porosities)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(prediction.grad).all()
    assert thresholds.grad is None and porosities.grad is None
    otsu.assert_not_called()


def test_precomputed_porosity_uses_one_threshold_per_sample():
    from supercat.metrics import PorosityLoss

    prediction = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]).repeat(2, 1, 1, 1)
    thresholds = torch.tensor([0.5, 2.5])
    porosities = torch.tensor([0.25, 0.75])
    torch.testing.assert_close(
        PorosityLoss(hard_mask=True, reduction="none")(
            prediction, thresholds, porosities
        ),
        torch.zeros(2),
    )


@pytest.mark.parametrize(
    "threshold,porosity,error",
    [
        (torch.tensor([0.0, 1.0]), torch.tensor([0.5]), ValueError),
        (torch.tensor([0.0]), torch.tensor([0.5, 0.5]), ValueError),
        (torch.tensor([0]), torch.tensor([0.5]), TypeError),
        (torch.tensor([0.0]), torch.tensor([-0.1]), ValueError),
        (torch.tensor([0.0]), torch.tensor([1.1]), ValueError),
        (torch.tensor([float("nan")]), torch.tensor([0.5]), ValueError),
        (torch.tensor([0.0]), torch.tensor([float("inf")]), ValueError),
    ],
)
def test_precomputed_porosity_validates_metadata(threshold, porosity, error):
    from supercat.metrics import PorosityLoss

    with pytest.raises(error):
        PorosityLoss()(torch.zeros(1, 1, 2, 2), threshold, porosity)


def test_porosity_reference_returns_detached_normalized_scalars():
    from supercat.metrics import porosity_reference

    target = torch.tensor([[[-1.0, -1.0], [1.0, 1.0]]], requires_grad=True)
    threshold, porosity = porosity_reference(target, hard_mask=True)
    assert -1 < threshold < 1
    assert porosity.item() == 0.5
    assert threshold.ndim == porosity.ndim == 0
    assert not threshold.requires_grad and not porosity.requires_grad


@pytest.mark.parametrize("temperature", [0, float("nan")])
def test_porosity_reference_rejects_invalid_temperature(temperature):
    from supercat.metrics import porosity_reference

    with pytest.raises(ValueError, match="temperature"):
        porosity_reference(torch.zeros(1, 2, 2), temperature=temperature)


def test_porosity_reference_rejects_integer_hr():
    from supercat.metrics import porosity_reference

    with pytest.raises(TypeError, match="floating-point"):
        porosity_reference(torch.zeros(1, 2, 2, dtype=torch.int64))


def test_legacy_porosity_loss_rejects_integer_hr_with_float_prediction():
    from supercat.metrics import PorosityLoss

    with pytest.raises(TypeError, match="target must be a floating-point"):
        PorosityLoss()(
            torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2, dtype=torch.int64)
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_precomputed_porosity_accumulates_half_precision_references_safely(dtype):
    from supercat.metrics import PorosityLoss

    prediction = torch.zeros(1, 1, 2, 2, dtype=dtype, requires_grad=True)
    threshold = torch.tensor([0.1], dtype=dtype)
    porosity = torch.tensor([0.5], dtype=dtype)
    loss = PorosityLoss()(prediction, threshold, porosity)
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
