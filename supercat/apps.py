from pathlib import Path
from widitapp import WiDiTApp
from cluey import main, method
from contextlib import nullcontext
from rich.progress import Progress

from .data import build_datasets3D, build_datasets2D

class Supercat(WiDiTApp):
    @method
    def datasets(
        self,
        dim:int=3,
        deeprock:Path=None,
        scale:int=4,
        augment:bool=True,
        **kwargs,
    ) -> tuple:
        """ Returns training and validation datasets """
        build_function = build_datasets2D if dim == 2 else build_datasets3D
        return build_function(deeprock=deeprock, scale=scale, train_augment=augment)

    @main
    def predict(
        self,
        input:Path =None,
        output:Path =None,
        diffusion:bool=False,
        size:int = 100,
        size_i: int = 0,
        size_j: int = 0,
        size_k: int = 0,
        overlap:int=10,
        overlap_i:int=0,
        overlap_j:int=0,
        overlap_k:int=0,
        fusion:bool=False,
        checkpoint:Path=None,
        num_sampling_steps: int = 250,
        fusion_steps: int = 6,
        seed: int = 42,
        single_crop: bool = False,
        **kwargs,
    ):
        """ Makes predictions """
        import torch
        import math

        from .data import read_image
        from .models import DiffusionPredictionModel
        from .utils import generate_overlapping_intervals, distance_to_boundary, write_volume

        torch.set_grad_enabled(False)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        assert input is not None, "Must provide input path"
        input_image = torch.as_tensor(read_image(input))

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

        # Load checkpoint
        from widit import load_model

        model = load_model(checkpoint)

        if diffusion:
            model = DiffusionPredictionModel(model, num_sampling_steps)
            torch.manual_seed(seed)

        model.to(device=device)
        model.eval()

        episilon = 0.01 # small number so that we do not have a zero weight
        weight = episilon + distance_to_boundary(size_i=size_i, size_j=size_j, size_k=size_k)

        dtype = torch.float32

        prediction = torch.zeros_like(input_image, dtype=dtype)
        summed_weights = torch.zeros_like(input_image, dtype=dtype)
        input_image = input_image.to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        input_image[torch.isnan(input_image)] = -1.0

        intervals_i = generate_overlapping_intervals(prediction.shape[0], size_i, overlap_i)
        intervals_j = generate_overlapping_intervals(prediction.shape[1], size_j, overlap_j)
        intervals_k = generate_overlapping_intervals(prediction.shape[2], size_k, overlap_k)

        if single_crop:
            # choose the center crop only
            intervals_i = [intervals_i[len(intervals_i)//2]]
            intervals_j = [intervals_j[len(intervals_j)//2]]
            intervals_k = [intervals_k[len(intervals_k)//2]]

        total_tiles = len(intervals_i) * len(intervals_j) * len(intervals_k)
        progress_context = Progress() if total_tiles else nullcontext()

        with torch.no_grad():
            with progress_context as progress_bar:
                task_id = None
                if total_tiles:
                    task_id = progress_bar.add_task("Predicting tiles", total=total_tiles)

                for start_i, end_i in intervals_i:
                    for start_j, end_j in intervals_j:
                        for start_k, end_k in intervals_k:
                            cropped = input_image[:,:, start_i:end_i, start_j:end_j, start_k:end_k]
                            if torch.isnan(cropped).all():
                                 continue
                            
                            cropped = cropped.to(device=device)

                            result = model(cropped)
                            prediction[start_i:end_i, start_j:end_j, start_k:end_k] += result.squeeze(dim=0).squeeze(dim=0).to("cpu") * weight
                            summed_weights[start_i:end_i, start_j:end_j, start_k:end_k] += weight

                            if task_id is not None:
                                progress_bar.advance(task_id)

        non_zero_voxels = summed_weights > 0
        prediction[non_zero_voxels] /= summed_weights[non_zero_voxels]
        prediction[~non_zero_voxels] = math.nan

        write_volume(prediction, output_path)
