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


def test_porosity_loss_is_invariant_to_a_uniform_intensity_offset():
    from supercat.metrics import PorosityLoss

    target = torch.tensor([[[[-1.0, -1.0], [1.0, 1.0]]]])
    prediction = target + 0.5
    # Each image supplies its own Otsu threshold, exactly as calc_porosity does
    # when the saved prediction is measured, so a shift moves the threshold with
    # the data and leaves the porosity alone. The pixel loss owns intensity.
    assert PorosityLoss(hard_mask=True)(prediction, target).item() == 0


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


def test_porosity_loss_gradient_holds_the_otsu_threshold_fixed():
    from supercat.metrics import PorosityLoss, otsu_threshold

    prediction = torch.tensor(
        [[[[-0.9, -0.4], [0.1, 0.8]]]], dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor(
        [[[[-1.0, -0.7], [0.5, 1.0]]]], dtype=torch.float64, requires_grad=True
    )
    criterion = PorosityLoss(temperature=0.2)

    # Otsu's argmax is not differentiable, so the threshold is detached and the
    # gradient is the one obtained by holding it fixed. Pinning it makes the mask
    # path smooth again, which is what gradcheck can verify.
    threshold = otsu_threshold(prediction).reshape(1, 1, 1, 1)

    def fixed_threshold(x):
        mask = torch.sigmoid((threshold - x) / 0.2)
        porosity = mask.flatten(1).mean(1)
        reference = criterion.porosity(target.detach())
        error = (porosity - reference) / reference.clamp_min(criterion.eps)
        return torch.nn.functional.smooth_l1_loss(
            error, torch.zeros_like(error), beta=criterion.beta
        )

    assert torch.autograd.gradcheck(fixed_threshold, (prediction,))
    torch.testing.assert_close(
        torch.autograd.grad(criterion(prediction, target), prediction)[0],
        torch.autograd.grad(fixed_threshold(prediction), prediction)[0],
    )
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


def test_legacy_porosity_loss_rejects_integer_hr_with_float_prediction():
    from supercat.metrics import PorosityLoss

    with pytest.raises(TypeError, match="target must be a floating-point"):
        PorosityLoss()(
            torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2, dtype=torch.int64)
        )


def otsu_reference(sample):
    """skimage's threshold, the implementation this one replaces."""
    from skimage import filters

    return filters.threshold_otsu(sample.detach().numpy().reshape(-1))


def bin_width(sample, nbins=256):
    return (sample.max() - sample.min()).item() / nbins


@pytest.mark.parametrize(
    "sample",
    [
        torch.linspace(-1, 1, 64).reshape(8, 8),
        torch.cat([torch.full((32,), -0.8), torch.full((32,), 0.7)]).reshape(8, 8),
        torch.tensor([0.0, 0.1, 0.2, 0.8, 0.9, 1.0]),
    ],
    ids=["ramp", "bimodal", "sparse"],
)
def test_otsu_threshold_matches_skimage_within_one_bin(sample):
    from supercat.metrics import otsu_threshold

    actual = otsu_threshold(sample[None]).item()
    assert abs(actual - otsu_reference(sample)) <= bin_width(sample)


def test_otsu_threshold_matches_skimage_on_a_real_micro_ct_slice():
    from supercat.metrics import otsu_threshold

    # A smooth analytic stand-in for a rock slice: two phases plus a blurred rim.
    grid = torch.linspace(-1, 1, 128)
    radius = (grid[:, None] ** 2 + grid[None, :] ** 2).sqrt()
    slice_2d = torch.tanh((radius - 0.6) * 8) * 0.9
    actual = otsu_threshold(slice_2d[None]).item()
    assert abs(actual - otsu_reference(slice_2d)) <= bin_width(slice_2d)


def test_otsu_threshold_is_batched_and_independent_per_sample():
    from supercat.metrics import otsu_threshold

    samples = torch.stack(
        [torch.linspace(-1, 1, 64), torch.linspace(4, 9, 64)]
    ).reshape(2, 8, 8)
    batched = otsu_threshold(samples)
    assert batched.shape == (2,)
    for index, sample in enumerate(samples):
        torch.testing.assert_close(batched[index], otsu_threshold(sample[None])[0])
    assert batched[1] > batched[0]  # Thresholds follow each sample's own range.


def test_otsu_threshold_keeps_the_input_device_and_never_detaches_a_graph():
    from supercat.metrics import otsu_threshold

    sample = torch.linspace(-1, 1, 64).reshape(1, 8, 8).requires_grad_()
    threshold = otsu_threshold(sample)
    assert threshold.device == sample.device
    assert not threshold.requires_grad  # argmax over bins is not differentiable


def test_otsu_threshold_of_a_constant_sample_leaves_nothing_below_it():
    from supercat.metrics import calc_porosity, otsu_threshold

    constant = torch.full((1, 4, 4), 0.3)
    threshold = otsu_threshold(constant)
    torch.testing.assert_close(threshold, torch.tensor([0.3]))
    assert otsu_reference(constant[0]) == pytest.approx(0.3)
    assert (constant < threshold).sum() == 0
    assert calc_porosity(constant[0]) == 0.0


def test_otsu_threshold_handles_near_constant_samples_that_skimage_rejects():
    from supercat.metrics import calc_porosity, otsu_threshold

    torch.manual_seed(0)
    near_constant = torch.full((1, 16, 16), 0.3) + 1e-6 * torch.randn(1, 16, 16)
    # skimage cannot build 256 finite-sized bins across so narrow a range.
    with pytest.raises(ValueError, match="bins"):
        otsu_reference(near_constant[0])
    threshold = otsu_threshold(near_constant)
    assert threshold.isfinite().all()
    assert near_constant.min() <= threshold.item() <= near_constant.max()
    assert 0.0 <= calc_porosity(near_constant[0]) <= 1.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.uint8])
def test_otsu_threshold_promotes_narrow_dtypes(dtype):
    from supercat.metrics import otsu_threshold

    values = torch.tensor([0, 10, 20, 200, 210, 220])
    sample = values.to(dtype).reshape(1, 6)
    assert otsu_threshold(sample).isfinite().all()


def test_otsu_threshold_rejects_degenerate_configuration():
    from supercat.metrics import otsu_threshold

    with pytest.raises(ValueError, match="nbins"):
        otsu_threshold(torch.zeros(1, 4), nbins=1)
    with pytest.raises(ValueError, match="non-empty"):
        otsu_threshold(torch.zeros(0, 4))


def test_calc_porosity_uses_the_shared_threshold():
    from supercat.metrics import calc_porosity, otsu_threshold

    volume = torch.linspace(-1, 1, 125).reshape(5, 5, 5)
    threshold = otsu_threshold(volume.reshape(1, -1))
    assert calc_porosity(volume) == pytest.approx(
        (volume < threshold).double().mean().item()
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.int32])
def test_calc_porosity_accepts_integer_images(dtype):
    from supercat.metrics import calc_porosity

    values = np.array([0, 10, 20, 200, 210, 220], dtype=dtype)
    assert calc_porosity(values) == pytest.approx(calc_porosity(values.astype(np.float32)))
