from pathlib import Path
import csv
import os
import tempfile
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import rescale, resize
import hdf5storage
from PIL import Image
from rich.progress import Progress


def read_image(path: str | Path, size: tuple[int, int, int] | None = None):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in [".png", ".jpg", ".jpeg"]:
        img = Image.open(path).convert("L")
        if size is not None:
            size = size[:2]
            img = img.resize(size[::-1], Image.BICUBIC)
    elif suffix in [".mat"]:
        img = read_mat(path)
        assert img.ndim == 3, f"Expected 3D array from {path}, got shape {img.shape}"
        if size is not None:
            img = resize(
                img,
                output_shape=size,
                order=3,  # cubic
                mode="reflect",
                anti_aliasing=False,  # no AA on upscaling
                preserve_range=True,
            )
    else:
        raise ValueError(f"Unsupported image format: {suffix}")

    array = np.array(img, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def transform_scale(data):
    return 2.0 * data / 255.0 - 1


def read_image_as_tensor(path, **kwargs) -> torch.Tensor:
    image = read_image(path, **kwargs)
    image = transform_scale(image)

    image = torch.from_numpy(image.copy())  # ensure contiguous

    return image


TRANSFORMATIONS_3D = [
    lambda tensor: tensor,  # Identity
    lambda tensor: tensor.flip(-2).transpose(-1, -2),  # 90° rotation
    lambda tensor: tensor.flip(-2).flip(-1),  # 180° rotation
    lambda tensor: tensor.flip(-2)
    .transpose(-1, -2)
    .flip(-1),  # TR-BL diagonal reflection
    lambda tensor: tensor.flip(-1),  # Vertical reflection
    lambda tensor: tensor.flip(-2),  # Horizontal reflection
    lambda tensor: tensor.flip(-1).transpose(-1, -2),  # TL-BR diagonal reflection
    lambda tensor: tensor.flip(-1)
    .flip(-2)
    .transpose(-1, -2),  # TR-BL diagonal reflection with both flips
    lambda tensor: tensor.flip(-3),
    lambda tensor: tensor.flip(-1).flip(-3),
    lambda tensor: tensor.flip(-2).flip(-3),
    lambda tensor: tensor.flip(-1).flip(-2).flip(-3),
    lambda tensor: tensor.transpose(-1, -2).flip(-3),
    lambda tensor: tensor.transpose(-1, -2).flip(-1).flip(-3),
    lambda tensor: tensor.transpose(-1, -2).flip(-2).flip(-3),
    lambda tensor: tensor.transpose(-1, -2).flip(-1).flip(-2).flip(-3),
    lambda tensor: tensor.transpose(-1, -3),
    lambda tensor: tensor.transpose(-1, -3).flip(-1),
    lambda tensor: tensor.transpose(-1, -3).flip(-2),
    lambda tensor: tensor.transpose(-1, -3).flip(-3),
    lambda tensor: tensor.transpose(-1, -3).flip(-1).flip(-2),
    lambda tensor: tensor.transpose(-1, -3).flip(-1).flip(-3),
    lambda tensor: tensor.transpose(-1, -3).flip(-2).flip(-3),
    lambda tensor: tensor.transpose(-1, -3).flip(-1).flip(-2).flip(-3),
    lambda tensor: tensor.transpose(-2, -3),
    lambda tensor: tensor.transpose(-2, -3).flip(-1),
    lambda tensor: tensor.transpose(-2, -3).flip(-2),
    lambda tensor: tensor.transpose(-2, -3).flip(-3),
    lambda tensor: tensor.transpose(-2, -3).flip(-1).flip(-2),
    lambda tensor: tensor.transpose(-2, -3).flip(-1).flip(-3),
    lambda tensor: tensor.transpose(-2, -3).flip(-2).flip(-3),
    lambda tensor: tensor.transpose(-2, -3).flip(-1).flip(-2).flip(-3),
    lambda tensor: tensor.transpose(-2, -3).transpose(-1, -2),
    lambda tensor: tensor.transpose(-2, -3).transpose(-1, -2).flip(-1),
    lambda tensor: tensor.transpose(-2, -3).transpose(-1, -2).flip(-2),
    lambda tensor: tensor.transpose(-2, -3).transpose(-1, -2).flip(-3),
    lambda tensor: tensor.transpose(-2, -3).transpose(-1, -2).flip(-1).flip(-2),
    lambda tensor: tensor.transpose(-2, -3).transpose(-1, -2).flip(-1).flip(-3),
    lambda tensor: tensor.transpose(-2, -3).transpose(-1, -2).flip(-2).flip(-3),
    lambda tensor: tensor.transpose(-2, -3)
    .transpose(-1, -2)
    .flip(-1)
    .flip(-2)
    .flip(-3),
    lambda tensor: tensor.transpose(-1, -2).transpose(-2, -3),
    lambda tensor: tensor.transpose(-1, -2).transpose(-2, -3).flip(-1),
    lambda tensor: tensor.transpose(-1, -2).transpose(-2, -3).flip(-2),
    lambda tensor: tensor.transpose(-1, -2).transpose(-2, -3).flip(-3),
    lambda tensor: tensor.transpose(-1, -2).transpose(-2, -3).flip(-1).flip(-2),
    lambda tensor: tensor.transpose(-1, -2).transpose(-2, -3).flip(-1).flip(-3),
    lambda tensor: tensor.transpose(-1, -2).transpose(-2, -3).flip(-2).flip(-3),
    lambda tensor: tensor.transpose(-1, -2)
    .transpose(-2, -3)
    .flip(-1)
    .flip(-2)
    .flip(-3),
]
TRANSFORMATIONS_2D = TRANSFORMATIONS_3D[:8]

# def flip_and_rotate(
#     batch:tuple[torch.Tensor,torch.Tensor,torch.Tensor],
#     sym_id:int|None=None
# ) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
#     """
#     Applies a random or specified flip and/or rotation transformation to a batch of 2D or 3D tensors.
#     This function transforms the input, target, and residual tensors by flipping, rotating, and
#     transposing them in a consistent manner.

#     Args:
#         batch (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing the input, target,
#             and residual tensors to be transformed.
#         sym_id (int or None, optional): The transformation identifier. If None, a random transformation
#             is chosen. For 2D tensors, sym_id ranges from 0 to 7; for 3D tensors, it ranges from 0 to 47.

#     Returns:
#         tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the transformed input,
#             target, and residual tensors after applying the flip and/or rotation.

#     Raises:
#         AssertionError: If the sym_id is out of the valid range for the tensor dimensions.
#     """
#     input, target, residual = batch

#     # Determine if 2D (4D) or 3D (5D)
#     is_2d = (input.ndim == 4)
#     number_of_transforms = 8 if is_2d else len(TRANSFORMATIONS)

#     if sym_id is None:
#         # Randomly choose one of the transformations
#         sym_id = random.randint(0, number_of_transforms - 1)

#     # Shortcut for no transformation
#     if sym_id == 0:
#         return batch

#     assert 0 <= sym_id < number_of_transforms
#     transformation = TRANSFORMATIONS[sym_id]

#     input = transformation(input)
#     target = transformation(target)
#     residual = transformation(residual)

#     return input, target, residual


def downscale_tricubic_rescale(vol: np.ndarray, factor: int = 4) -> np.ndarray:
    """
    Downscale a 3D volume by `factor` using tricubic interpolation + anti-aliasing.
    """
    low = rescale(
        vol,
        scale=1.0 / factor,
        order=3,  # cubic
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
        channel_axis=None,  # ensures (D,H,W) or (C,D,H,W) handled correctly if needed
    )
    return low.astype(vol.dtype, copy=False)


def upscale_tricubic_rescale(vol: np.ndarray, factor: int = 4) -> np.ndarray:
    """
    Upscale a 3D volume by `factor` using tricubic interpolation.
    """
    up = rescale(
        vol,
        scale=factor,
        order=3,  # cubic
        mode="reflect",
        anti_aliasing=False,  # no AA on upscaling
        preserve_range=True,
        channel_axis=None,
    )
    return up.astype(vol.dtype, copy=False)


def read_mat(path: Path):
    DEEPROCK_HDF5_KEY = "temp"
    try:
        data_dict = hdf5storage.loadmat(str(path))
    except Exception as err:
        raise IOError(f"Error reading 3D file '{path}':\n{err}")
    if DEEPROCK_HDF5_KEY not in data_dict:
        keys_found = ",".join(data_dict.keys())
        raise Exception(
            f"expected key {DEEPROCK_HDF5_KEY} not found in '{path}'.\nCheck the following keys: {keys_found}"
        )

    return data_dict[DEEPROCK_HDF5_KEY]


class Deeprock3D(Dataset):
    def __init__(
        self,
        deeprock: Path,
        scale: int = 4,
        channel_first: bool = True,
        partition: str = "train",
        augment: bool = False,
    ):
        self.deeprock = Path(deeprock)
        self.scale = int(scale)
        self.channel_first = channel_first
        categories = ["sandstone", "carbonate", "coal", "sand"]
        self.augment = augment

        hr_items: list[Path] = []
        for cat in categories:
            hr_dir = self.deeprock / f"{cat}3D" / f"{cat}3D_{partition}_HR"
            hr_items.extend(sorted(hr_dir.glob("*.mat")))
        self.hr_items = hr_items

        if len(self.hr_items) == 0:
            raise FileNotFoundError(
                f"No .mat files found under {self.deeprock}/*3D/*3D_{partition}_HR"
            )

    def _hr_to_lr_path(self, hr_path: Path) -> Path:
        """
        Map .../<cat>3D/<cat>3D_train_HR/<file>.mat  -->
            .../<cat>3D/<cat>3D_train_TRI_unknown_X{scale}/<file>.mat
        """
        rel = hr_path.relative_to(self.deeprock)
        parts = list(rel.parts)
        # parts[-2] expected: "<cat>3D_train_HR"
        parent_name = parts[-2]
        if not parent_name.endswith("_HR"):
            raise ValueError(f"Unexpected parent '{parent_name}' for {hr_path}")
        parts[-2] = parent_name.replace("_HR", f"_LR_default_X{self.scale}").replace(
            ".mat", f"x{self.scale}.mat"
        )
        parts[-1] = parts[-1].replace(".mat", f"x{self.scale}.mat")
        return self.deeprock.joinpath(*parts)

    def __len__(self) -> int:
        return len(self.hr_items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hr_path = self.hr_items[idx]

        lr_path = self._hr_to_lr_path(hr_path)

        # if not lr_path.exists():
        #     raise FileNotFoundError(f"LR file missing for {hr_path.name}: {lr_path}")

        hr = transform_scale(read_mat(hr_path))
        lr_orig = transform_scale(read_mat(lr_path))
        # lr_orig = downscale_tricubic_rescale(lr_orig, factor=self.scale)
        lr = upscale_tricubic_rescale(lr_orig, factor=self.scale)

        # lr = transform_scale(read_mat(lr_path))

        # Optional: enforce channel-first tensors
        # If (D,H,W), add channel dim; if already (C,D,H,W), leave it.
        if hr.ndim == 3:
            hr = hr[None, ...]  # (1,D,H,W)
        if lr.ndim == 3:
            lr = lr[None, ...]
        if not self.channel_first and hr.ndim == 4:
            # Convert to (D,H,W,C)
            hr = np.moveaxis(hr, 0, -1)
            lr = np.moveaxis(lr, 0, -1)

        hr_t = torch.from_numpy(hr.copy())  # ensure contiguous
        lr_t = torch.from_numpy(lr.copy())

        if self.augment:
            transformation = TRANSFORMATIONS_3D[
                np.random.randint(0, len(TRANSFORMATIONS_3D))
            ]
            hr_t = transformation(hr_t)
            lr_t = transformation(lr_t)

        assert (
            lr_t.shape == hr_t.shape
        ), f"LR shape {lr_t.shape} != HR shape {hr_t.shape} for {hr_path.name} - {lr_path.name}, original LR shape {lr_orig.shape}. scale={self.scale}"
        assert (
            lr_t.max() < 1.01
        ), f"Low resolution {lr_path} gives range {lr_t.min()}-{lr_t.max()}"
        assert (
            lr_t.min() > -1.01
        ), f"Low resolution {lr_path} gives range {lr_t.min()}-{lr_t.max()}"
        assert (
            hr_t.max() < 1.01
        ), f"High resolution {hr_path} gives range {hr_t.min()}-{hr_t.max()}"
        assert (
            hr_t.min() > -1.01
        ), f"High resolution {hr_path} gives range {hr_t.min()}-{hr_t.max()}"

        return lr_t, hr_t


class Deeprock2D(Dataset):
    def __init__(
        self,
        deeprock: Path,
        scale: int = 4,
        channel_first: bool = True,
        partition: str = "train",
        augment: bool = False,
    ):
        self.deeprock = Path(deeprock)
        self.scale = int(scale)
        self.channel_first = channel_first
        categories = ["sandstone", "carbonate", "coal", "sand"]
        self.augment = augment

        hr_items: list[Path] = []
        for cat in categories:
            hr_dir = self.deeprock / f"{cat}2D" / f"{cat}2D_{partition}_HR"
            hr_items.extend(sorted(hr_dir.glob("*.png")))
        self.hr_items = hr_items

        if len(self.hr_items) == 0:
            raise FileNotFoundError(
                f"No .png files found under {self.deeprock}/*2D/*2D_{partition}_HR"
            )

    def _hr_to_lr_path(self, hr_path: Path) -> Path:
        """
        Map .../<cat>2D/<cat>2D_train_HR/<file>.png  -->
            .../<cat>2D/<cat>2D_train_BI_unknown_X{scale}/<file>.png
        """
        rel = hr_path.relative_to(self.deeprock)
        parts = list(rel.parts)
        # parts[-2] expected: "<cat>2D_train_HR"
        parent_name = parts[-2]
        if not parent_name.endswith("_HR"):
            raise ValueError(f"Unexpected parent '{parent_name}' for {hr_path}")
        parts[-2] = parent_name.replace("_HR", f"_BI_unknown_X{self.scale}")
        return self.deeprock.joinpath(*parts)

    def __len__(self) -> int:
        return len(self.hr_items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hr_path = self.hr_items[idx]
        lr_path = self._hr_to_lr_path(hr_path)

        if not lr_path.exists():
            raise FileNotFoundError(f"LR file missing for {hr_path.name}: {lr_path}")

        hr_t = read_image_as_tensor(hr_path)
        lr_t = read_image_as_tensor(lr_path)

        assert hr_t.shape[0] == 1

        if self.augment:
            transformation = TRANSFORMATIONS_2D[
                np.random.randint(0, len(TRANSFORMATIONS_2D))
            ]
            hr_t = transformation(hr_t)
            lr_t = transformation(lr_t)

        return lr_t, hr_t



def reject_metadata_batches(include_porosity, porosity_csv=None, allow=False):
    """Refuse porosity metadata for apps whose datasets feed WiDiTApp's trainer.

    PorosityDataset appends the HR threshold and porosity to each sample, so
    batches carry four items. The trainer accepts only (x, target) or
    (x, target, timestep) and raises deep inside training, after the model and
    any W&B run are already up. Failing here keeps that mistake cheap.

    build_datasets2D, build_datasets3D and PorosityDataset accept the metadata
    unconditionally; ``allow`` opts an app back in for a custom loop that reads
    four-item batches from its dataloaders.
    """
    if allow or (not include_porosity and porosity_csv is None):
        return
    requested = "include_porosity" if include_porosity else "porosity_csv"
    raise ValueError(
        f"{requested} adds HR porosity metadata to every sample, producing four-item "
        "batches that the built-in trainer cannot consume. Drop the option to train, "
        "pass allow_metadata_batches=True for a custom loop that reads them, or build "
        "the datasets directly with supercat.data.build_datasets2D, build_datasets3D "
        "or PorosityDataset."
    )


def build_datasets3D(
    deeprock: Path,
    scale: int = 4,
    train_augment: bool = True,
    include_porosity: bool = False,
    porosity_temperature: float = 0.05,
    porosity_hard_mask: bool = False,
    porosity_csv: Path | None = None,
):
    training_dataset = Deeprock3D(
        deeprock=deeprock, scale=scale, partition="train", augment=train_augment
    )
    validation_dataset = Deeprock3D(
        deeprock=deeprock, scale=scale, partition="valid", augment=False
    )
    if include_porosity or porosity_csv is not None:
        return _deeprock_porosity_datasets(
            (training_dataset, validation_dataset),
            porosity_temperature,
            porosity_hard_mask,
            porosity_csv,
        )
    return training_dataset, validation_dataset


def build_datasets2D(
    deeprock: Path,
    scale: int = 4,
    train_augment: bool = True,
    include_porosity: bool = False,
    porosity_temperature: float = 0.05,
    porosity_hard_mask: bool = False,
    porosity_csv: Path | None = None,
):
    training_dataset = Deeprock2D(
        deeprock=deeprock, scale=scale, partition="train", augment=train_augment
    )
    validation_dataset = Deeprock2D(
        deeprock=deeprock, scale=scale, partition="valid", augment=False
    )
    if include_porosity or porosity_csv is not None:
        return _deeprock_porosity_datasets(
            (training_dataset, validation_dataset),
            porosity_temperature,
            porosity_hard_mask,
            porosity_csv,
        )
    return training_dataset, validation_dataset


class PorosityDataset(Dataset):
    """Append HR threshold and porosity to (LR, HR) samples for default collation.

    Returns (lr, hr, hr_threshold, hr_porosity), with scalar CPU reference tensors.
    Use the same temperature and hard_mask settings in PorosityLoss. References
    use the HR sample's intensity scale, normally [-1, 1].

    precompute=True calculates and stores only the two scalars per sample at
    construction. Use it only when target histograms are invariant across reads
    (e.g. DeepRock flips/rotations). Leave it False for random crops, padding or
    other transforms that change porosity; references are then computed for the
    actual returned HR crop in the dataset worker, before GPU training.
    """

    def __init__(
        self,
        dataset: Dataset,
        temperature: float = 0.05,
        hard_mask: bool = False,
        precompute: bool = False,
    ):
        from .metrics import PorosityLoss, porosity_reference

        # Validate the shared mask configuration even for an empty dataset.
        PorosityLoss(temperature=temperature, hard_mask=hard_mask)
        self.dataset = dataset
        self.temperature = temperature
        self.hard_mask = hard_mask
        self.references = None
        if precompute:
            self.references = [
                porosity_reference(dataset[index][1], temperature, hard_mask)
                for index in _with_progress(range(len(dataset)))
            ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        from .metrics import porosity_reference

        lr, hr = self.dataset[index]
        threshold, porosity = (
            self.references[index]
            if self.references is not None
            else porosity_reference(hr, self.temperature, self.hard_mask)
        )
        return lr, hr, threshold.clone(), porosity.clone()



def _with_progress(items, description="Calculating Otsu thresholds and porosity"):
    """Yield items behind a rich progress bar, staying silent when there is no work."""
    total = len(items)
    if not total:
        yield from items
        return
    with Progress() as progress_bar:
        task_id = progress_bar.add_task(description, total=total)
        for item in items:
            yield item
            progress_bar.advance(task_id)


def _deeprock_porosity_datasets(datasets, temperature, hard_mask, csv_path):
    """Wrap both splits using an optional shared CSV of normalized HR references."""
    from .metrics import porosity_reference

    wrapped = tuple(
        PorosityDataset(dataset, temperature, hard_mask) for dataset in datasets
    )
    if csv_path is None:
        return wrapped

    csv_path = Path(csv_path)
    fields = ["hr_path", "threshold", "porosity", "temperature", "hard_mask", "dtype"]
    references = {}
    if csv_path.exists():
        with csv_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != fields:
                raise ValueError(
                    f"Invalid porosity CSV columns in {csv_path}; expected {fields}"
                )
            for row in reader:
                key = row["hr_path"]
                if key in references:
                    raise ValueError(f"Duplicate HR path in porosity CSV: {key}")
                if float(row["temperature"]) != temperature or row["hard_mask"] != str(
                    hard_mask
                ):
                    raise ValueError(
                        f"Porosity CSV mask settings do not match for {key}"
                    )
                if row["dtype"] not in {"float32", "float64"}:
                    raise ValueError(f"Invalid porosity CSV dtype for {key}")
                threshold, porosity = float(row["threshold"]), float(row["porosity"])
                if (
                    not np.isfinite(threshold)
                    or not np.isfinite(porosity)
                    or not 0 <= porosity <= 1
                ):
                    raise ValueError(f"Invalid porosity CSV reference values for {key}")
                dtype = getattr(torch, row["dtype"])
                references[key] = (
                    torch.tensor(threshold, dtype=dtype),
                    torch.tensor(porosity, dtype=dtype),
                )
        for wrapper in wrapped:
            keys = [
                path.relative_to(wrapper.dataset.deeprock).as_posix()
                for path in wrapper.dataset.hr_items
            ]
            missing = [key for key in keys if key not in references]
            if missing:
                raise ValueError(
                    f"Porosity CSV {csv_path} is missing HR paths: {missing}"
                )
            wrapper.references = [references[key] for key in keys]
        return wrapped

    # Read only HR images; LR interpolation and random augmentation are unnecessary.
    rows = []
    for wrapper in wrapped:
        wrapper.references = []
    items = [(wrapper, path) for wrapper in wrapped for path in wrapper.dataset.hr_items]
    for wrapper, path in _with_progress(items):
        hr = (
            torch.from_numpy(transform_scale(read_mat(path)).copy())
            if isinstance(wrapper.dataset, Deeprock3D)
            else read_image_as_tensor(path)
        )
        threshold, porosity = porosity_reference(hr, temperature, hard_mask)
        wrapper.references.append((threshold, porosity))
        rows.append(
            dict(
                hr_path=path.relative_to(wrapper.dataset.deeprock).as_posix(),
                threshold=threshold.item(),
                porosity=porosity.item(),
                temperature=temperature,
                hard_mask=hard_mask,
                dtype=str(threshold.dtype).removeprefix("torch."),
            )
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete CSV, so an interrupted computation cannot leave a partial cache.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", dir=csv_path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, csv_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return wrapped
