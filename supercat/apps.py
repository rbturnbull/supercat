from pathlib import Path
from widitapp import WiDiTApp
import cluey
from cluey import main, method, tool
from contextlib import nullcontext
from rich.progress import Progress

from .data import build_datasets3D, build_datasets2D


class Supercat(WiDiTApp):

    @cluey.method("super")
    def loss(
        self,
        porosity_loss_weight: float = cluey.Option(
            0.0,
            help="Weight of the auxiliary porosity loss; zero preserves the parent objective",
        ),
        porosity_temperature: float = cluey.Option(
            0.05, help="Sigmoid temperature for porosity; intensity units"
        ),
        **kwargs,
    ):
        """Add optional porosity loss to the inherited image criterion."""
        from .training import add_porosity_loss

        parent = super().loss(**kwargs)
        return add_porosity_loss(parent, porosity_loss_weight, porosity_temperature)

    @cluey.method("super")
    def metrics(self, **kwargs):
        """Report porosity loss alongside inherited validation metrics."""
        from .metrics import PorosityLoss

        metrics = super().metrics(**kwargs)
        metrics["porosity_loss"] = PorosityLoss(hard_mask=True)
        return metrics

    @method
    def datasets(
        self,
        dim: int = cluey.Option(3, help="Number of spatial dimensions (2 or 3)"),
        deeprock: Path = cluey.Option(
            None, help="Path to the DeepRock dataset directory"
        ),
        scale: int = cluey.Option(4, help="Scale factor for downsampling the images"),
        augment: bool = cluey.Option(
            True, help="Apply data augmentation to the training images"
        ),
        include_porosity: bool = cluey.Option(
            False,
            help="Include HR threshold and porosity in dataset samples (requires a compatible training loop)",
        ),
        porosity_temperature: float = cluey.Option(
            0.05, help="Sigmoid temperature for HR porosity; must match PorosityLoss"
        ),
        porosity_csv: Path = cluey.Option(
            None,
            help="Load or save DeepRock HR porosity references in this CSV; enables porosity metadata",
        ),
        **kwargs,
    ) -> tuple:
        """Build training and validation datasets for 2D or 3D super-resolution."""
        build_function = build_datasets2D if dim == 2 else build_datasets3D
        metadata_options = {}
        if include_porosity or porosity_csv is not None:
            metadata_options = dict(
                include_porosity=True,
                porosity_temperature=porosity_temperature,
            )
        if porosity_csv is not None:
            metadata_options["porosity_csv"] = porosity_csv
        return build_function(
            deeprock=deeprock, scale=scale, train_augment=augment, **metadata_options
        )

    @main
    def predict(
        self,
        input: Path = cluey.Option(None, help="Path to the input image"),
        output: Path = cluey.Option(None, help="Path to save the predicted image"),
        size: int = cluey.Option(
            100, help="Default prediction tile size for all spatial dimensions"
        ),
        size_i: int = cluey.Option(
            0, help="Prediction tile size along the i dimension (0 uses size)"
        ),
        size_j: int = cluey.Option(
            0, help="Prediction tile size along the j dimension (0 uses size)"
        ),
        size_k: int = cluey.Option(
            0, help="Prediction tile size along the k dimension (0 uses size)"
        ),
        overlap: int = cluey.Option(
            10, help="Default overlap between prediction tiles in pixels or voxels"
        ),
        overlap_i: int = cluey.Option(
            0, help="Tile overlap along the i dimension (0 uses overlap)"
        ),
        overlap_j: int = cluey.Option(
            0, help="Tile overlap along the j dimension (0 uses overlap)"
        ),
        overlap_k: int = cluey.Option(
            0, help="Tile overlap along the k dimension (0 uses overlap)"
        ),
        checkpoint: Path = cluey.Option(None, help="Path to the model checkpoint"),
        num_sampling_steps: int = cluey.Option(
            250, help="Number of diffusion sampling steps per prediction"
        ),
        seed: int = cluey.Option(42, help="Random seed for diffusion sampling"),
        single_crop: bool = cluey.Option(False, help="Predict only the center tile"),
        **kwargs,
    ):
        """Generate and save a super-resolution prediction for an input image."""
        import torch
        from widit import load_model

        from .data import read_image_as_tensor
        from .models import DiffusionPredictionModel
        from .utils import write_image

        torch.set_grad_enabled(False)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        assert output is not None, "Must provide output path"
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        assert checkpoint is not None, "Must provide path to model checkpoint"

        size_i = size_i or size
        size_j = size_j or size
        size_k = size_k or size

        overlap_i = overlap_i or overlap
        overlap_j = overlap_j or overlap
        overlap_k = overlap_k or overlap

        assert input is not None, "Must provide input path"
        input_image = read_image_as_tensor(input, size=(size_k, size_j, size_i))
        spatial_dims = input_image.ndim - 1
        assert spatial_dims in (
            2,
            3,
        ), f"Input image must have 3 or 4 dimensions (C, [D], H, W), got {input_image.shape}"

        assert (
            input_image.shape[-1] == size_i
        ), f"Input image size in i dimension ({input_image.shape[-1]}) does not match specified size_i ({size_i})"
        assert (
            input_image.shape[-2] == size_j
        ), f"Input image size in j dimension ({input_image.shape[-2]}) does not match specified size_j ({size_j})"
        if spatial_dims == 3:
            assert (
                input_image.shape[-3] == size_k
            ), f"Input image size in k dimension ({input_image.shape[-3]}) does not match specified size_k ({size_k})"

        # Load checkpoint
        model = load_model(checkpoint)

        diffusion = model.out_channels == 2
        if diffusion:
            model = DiffusionPredictionModel(model, num_sampling_steps)
            torch.manual_seed(seed)

        model.to(device=device)
        model.eval()

        prediction = self.generate_prediction(
            input_image=input_image,
            model=model,
            size_i=size_i,
            size_j=size_j,
            size_k=size_k,
            overlap_i=overlap_i,
            overlap_j=overlap_j,
            overlap_k=overlap_k,
            spatial_dims=spatial_dims,
            device=device,
            single_crop=single_crop,
        )
        write_image(prediction, output_path)

    def generate_prediction(
        self,
        input_image: "torch.Tensor",
        model: "torch.nn.Module",
        size_i: int,
        size_j: int,
        size_k: int,
        overlap_i: int,
        overlap_j: int,
        overlap_k: int,
        spatial_dims: int,
        device: str,
        single_crop: bool,
    ) -> "torch.Tensor":
        import torch
        import math

        from .utils import generate_overlapping_intervals, distance_to_boundary

        episilon = 0.01  # small number so that we do not have a zero weight
        weight = episilon + distance_to_boundary(
            size_i=size_i, size_j=size_j, size_k=size_k if spatial_dims == 3 else 1
        )
        if spatial_dims == 2:
            weight = weight[:, :, 0]

        dtype = torch.float32

        prediction = torch.zeros_like(input_image, dtype=dtype)
        summed_weights = torch.zeros_like(input_image, dtype=dtype)
        input_image = input_image.to(dtype=dtype).unsqueeze(0)  # add batch dimension

        intervals_i = generate_overlapping_intervals(
            prediction.shape[-spatial_dims], size_i, overlap_i
        )
        intervals_j = generate_overlapping_intervals(
            prediction.shape[1 - spatial_dims], size_j, overlap_j
        )
        if spatial_dims == 3:
            intervals_k = generate_overlapping_intervals(
                prediction.shape[2 - spatial_dims], size_k, overlap_k
            )
        else:
            intervals_k = [(0, None)]

        if single_crop:
            # choose the center crop only
            intervals_i = [intervals_i[len(intervals_i) // 2]]
            intervals_j = [intervals_j[len(intervals_j) // 2]]

            if spatial_dims == 3:
                intervals_k = [intervals_k[len(intervals_k) // 2]]

        total_tiles = len(intervals_i) * len(intervals_j) * len(intervals_k)
        progress_context = Progress() if total_tiles else nullcontext()

        with torch.no_grad():
            with progress_context as progress_bar:
                task_id = None
                if total_tiles:
                    task_id = progress_bar.add_task(
                        "Predicting tiles", total=total_tiles
                    )

                for start_i, end_i in intervals_i:
                    for start_j, end_j in intervals_j:
                        for start_k, end_k in intervals_k:
                            spatial_ranges = [
                                slice(start_i, end_i),
                                slice(start_j, end_j),
                            ]
                            if spatial_dims == 3:
                                spatial_ranges.append(slice(start_k, end_k))

                            spatial_ranges = tuple(spatial_ranges)

                            cropped = input_image[
                                (slice(None), slice(None), *spatial_ranges)
                            ]
                            if torch.isnan(cropped).all():
                                continue

                            cropped = cropped.to(device=device)

                            result = model(cropped)
                            prediction[spatial_ranges] += (
                                result[0, 0].to("cpu") * weight
                            )
                            summed_weights[spatial_ranges] += weight

                            if task_id is not None:
                                progress_bar.advance(task_id)

        non_zero_voxels = summed_weights > 0
        prediction[non_zero_voxels] /= summed_weights[non_zero_voxels]
        prediction[~non_zero_voxels] = math.nan

        return prediction.squeeze(0)  # remove batch dimension

    @tool
    def porosity(
        self,
        input: Path = cluey.Option(None, help="Path to the input image"),
    ):
        """Calculate and print the porosity of an input image."""
        from .data import read_image_as_tensor
        from .metrics import calc_porosity

        input_image = read_image_as_tensor(input)
        porosity = calc_porosity(input_image)
        print(f"Porosity: {porosity}")
        return porosity

    @tool
    def porosity_distribution(
        self,
        input: Path = cluey.Option(None, help="Path to the input image"),
        output: Path = cluey.Option(
            None, help="Path to the output CSV containing seeds and porosities"
        ),
        size: int = cluey.Option(
            100, help="Default prediction tile size for all spatial dimensions"
        ),
        size_i: int = cluey.Option(
            0, help="Prediction tile size along the i dimension (0 uses size)"
        ),
        size_j: int = cluey.Option(
            0, help="Prediction tile size along the j dimension (0 uses size)"
        ),
        size_k: int = cluey.Option(
            0, help="Prediction tile size along the k dimension (0 uses size)"
        ),
        overlap: int = cluey.Option(
            10, help="Default overlap between prediction tiles in pixels or voxels"
        ),
        overlap_i: int = cluey.Option(
            0, help="Tile overlap along the i dimension (0 uses overlap)"
        ),
        overlap_j: int = cluey.Option(
            0, help="Tile overlap along the j dimension (0 uses overlap)"
        ),
        overlap_k: int = cluey.Option(
            0, help="Tile overlap along the k dimension (0 uses overlap)"
        ),
        checkpoint: Path = cluey.Option(None, help="Path to the model checkpoint"),
        num_sampling_steps: int = cluey.Option(
            250, help="Number of diffusion sampling steps per prediction"
        ),
        seed: int = cluey.Option(
            42, help="Starting random seed for consecutive diffusion samples"
        ),
        single_crop: bool = cluey.Option(False, help="Predict only the center tile"),
        count: int = cluey.Option(
            50,
            help="Number of consecutive seeds to sample, skipping seeds already saved",
        ),
        overwrite: bool = cluey.Option(
            False, help="Replace the existing output CSV instead of appending results"
        ),
        **kwargs,
    ):
        """Sample diffusion predictions and save their seeds and porosities to a CSV file."""
        import torch
        from widit import load_model

        from .data import read_image_as_tensor
        from .models import DiffusionPredictionModel
        from .metrics import calc_porosity

        torch.set_grad_enabled(False)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        size_i = size_i or size
        size_j = size_j or size
        size_k = size_k or size

        assert input is not None, "Must provide input path"
        input_image = read_image_as_tensor(input, size=(size_k, size_j, size_i))
        spatial_dims = input_image.ndim - 1
        assert spatial_dims in (
            2,
            3,
        ), f"Input image must have 3 or 4 dimensions (C, [D], H, W), got {input_image.shape}"

        assert output is not None, "Must provide output path"
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        assert checkpoint is not None, "Must provide path to model checkpoint"

        overlap_i = overlap_i or overlap
        overlap_j = overlap_j or overlap
        overlap_k = overlap_k or overlap

        assert input_image.shape[-1] == size_i, f"Input image size in i dimension ({input_image.shape[-1]}) does not match specified size_i ({size_i})"
        assert input_image.shape[-2] == size_j, f"Input image size in j dimension ({input_image.shape[-2]}) does not match specified size_j ({size_j})"
        if spatial_dims == 3:
            assert (
                input_image.shape[-3] == size_k
            ), f"Input image size in k dimension ({input_image.shape[-3]}) does not match specified size_k ({size_k})"

        # Load checkpoint
        model = load_model(checkpoint)

        diffusion = model.out_channels == 2
        assert (
            diffusion
        ), "Model must be a diffusion model with 2 output channels for porosity distribution tool"
        if diffusion:
            model = DiffusionPredictionModel(model, num_sampling_steps)

        model.to(device=device)
        model.eval()

        if overwrite and output_path.exists():
            output_path.unlink()

        existing_seeds = set()
        write_header = True
        if output_path.exists():
            write_header = False
            with open(output_path) as existing_file:
                for line in existing_file:
                    line = line.strip()
                    if not line or line == "seed,porosity":
                        continue
                    seed_value, _, _ = line.partition(",")
                    try:
                        existing_seeds.add(int(seed_value))
                    except ValueError:
                        continue

        with open(output_path, "a") as f:
            if write_header:
                f.write("seed,porosity\n")
                f.flush()

            for index in range(count):
                current_seed = seed + index
                if current_seed in existing_seeds:
                    print(
                        f"Skipping seed {current_seed}; already present in {output_path}"
                    )
                    continue

                torch.manual_seed(current_seed)
                prediction = self.generate_prediction(
                    input_image=input_image,
                    model=model,
                    size_i=size_i,
                    size_j=size_j,
                    size_k=size_k,
                    overlap_i=overlap_i,
                    overlap_j=overlap_j,
                    overlap_k=overlap_k,
                    spatial_dims=spatial_dims,
                    device=device,
                    single_crop=single_crop,
                )
                porosity = calc_porosity(prediction)
                print(f"Seed: {current_seed}, Porosity: {porosity}")
                f.write(f"{current_seed},{porosity}\n")
                f.flush()
