.. image:: https://raw.githubusercontent.com/rbturnbull/supercat/main/docs/images/Supercat-Banner.svg

.. start-badges

|pypi badge| |coverage badge| |docs badge| |black badge| |git3moji badge| |torchapp badge|

.. |pypi badge| image:: https://img.shields.io/pypi/v/supercat-sr.svg?color=blue
    :target: https://pypi.org/project/supercat-sr/

.. |testing badge| image:: https://github.com/rbturnbull/supercat/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/rbturnbull/supercat/actions

.. |docs badge| image:: https://github.com/rbturnbull/supercat/actions/workflows/docs.yml/badge.svg
    :target: https://rbturnbull.github.io/supercat
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. |coverage badge| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rbturnbull/009c4d68ae1d50b3af29a078bc346856/raw/coverage-badge.json
    :target: https://rbturnbull.github.io/supercat/coverage/

.. |git3moji badge| image:: https://img.shields.io/badge/git3moji-%E2%9A%A1%EF%B8%8F%F0%9F%90%9B%F0%9F%93%BA%F0%9F%91%AE%F0%9F%94%A4-fffad8.svg
    :target: https://robinpokorny.github.io/git3moji/

.. |torchapp badge| image:: https://img.shields.io/badge/MLOpps-torchapp-B1230A.svg
    :target: https://rbturnbull.github.io/torchapp/

.. end-badges

.. start-quickstart

Supercat provides deep learning tools for super-resolution of 2D CT images and
3D rock volumes. It supports diffusion and regression models, training on
DeepRock datasets, and porosity analysis of images and generated predictions.

Installation
==================================

Supercat requires Python 3.10, 3.11, or 3.12. Install from PyPI::

    pip install supercat-sr

The distribution is named ``supercat-sr`` because the name ``supercat`` was
already taken on PyPI. The module, the commands and the imports are all still
``supercat``.

To work from a local checkout instead::

    git clone https://github.com/rbturnbull/supercat.git
    cd supercat
    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install -e .

Check the available commands::

    supercat --help
    supercat-tools --help
    supercat-tools train --help

Usage
==================================

The examples below use placeholder dataset, image, and checkpoint paths. Replace
these with your own files. Prediction requires a trained model checkpoint that
matches the input dimensionality.

Generate a prediction
----------------------------------

Generate a 3D prediction from a MATLAB volume::

    supercat \
        --input /path/to/volume.mat \
        --output results/prediction.tif \
        --checkpoint /path/to/checkpoint.pt \
        --size 100

``supercat-tools predict`` exposes the same prediction command. Prediction uses
CUDA when available and otherwise runs on the CPU. Output directories are
created automatically.

Input and output formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* 2D inputs: ``.png``, ``.jpg``, or ``.jpeg``, converted to grayscale.
* 3D inputs: ``.mat`` files containing a three-dimensional array named ``temp``.
* Image intensities are expected on a 0–255 scale and normalized to [-1, 1].
* Use ``.tif`` for image or volume output, or ``.pt`` to save the prediction as a
  PyTorch tensor. TIFF output is converted to unsigned 8-bit intensities; tensor
  output retains the model's values.

Prediction options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 14 54

   * - Option
     - Default
     - Purpose
   * - ``--input``
     - Required
     - Input image or volume path.
   * - ``--output``
     - Required
     - Destination for the prediction.
   * - ``--checkpoint``
     - Required
     - Trained model checkpoint.
   * - ``--size``
     - ``100``
     - Default prediction tile size for all spatial dimensions.
   * - ``--size-i``, ``--size-j``, ``--size-k``
     - ``0``
     - Override individual dimensions; zero uses ``--size``.
   * - ``--overlap``
     - ``10``
     - Default tile overlap in pixels or voxels.
   * - ``--overlap-i``, ``--overlap-j``, ``--overlap-k``
     - ``0``
     - Override individual overlaps; zero uses ``--overlap``.
   * - ``--num-sampling-steps``
     - ``250``
     - Diffusion sampling steps per prediction.
   * - ``--seed``
     - ``42``
     - Random seed for diffusion sampling.
   * - ``--single-crop``
     - Disabled
     - Predict only the center tile.

The current prediction command resizes the input to the requested spatial size
before running the model. For example, ``--size 100`` resizes a 3D volume to
100 × 100 × 100 voxels. Choose a size compatible with your model and intended
output resolution. The training option ``--scale`` is not a prediction option.
Diffusion sampling options apply when a diffusion checkpoint is loaded.

Train a model
----------------------------------

Train a 3D diffusion model using paired DeepRock images::

    supercat-tools train \
        --deeprock /path/to/deeprock \
        --dim 3 \
        --scale 4 \
        --use-diffusion \
        --epochs 40 \
        --batch-size 1 \
        --learning-rate 0.0001 \
        --results-dir results/diffusion3d

For 2D training, use ``--dim 2`` with the corresponding 2D dataset. To train a
regression model, pass ``--no-use-diffusion``. Training augmentation is enabled
by default; disable it with ``--no-augment``.

Dataset layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass the directory containing the rock category folders to ``--deeprock``.
The loaders search for ``sandstone``, ``carbonate``, ``coal``, and ``sand``;
provide training and validation pairs for the categories you use.

For 3D data at scale 4, the layout for one category is::

    deeprock/
        sandstone3D/
            sandstone3D_train_HR/
                sample.mat
            sandstone3D_train_LR_default_X4/
                samplex4.mat
            sandstone3D_valid_HR/
                sample.mat
            sandstone3D_valid_LR_default_X4/
                samplex4.mat

Each MATLAB file must contain an array named ``temp``. The 3D loader upsamples
the low-resolution volume by ``--scale`` using cubic interpolation before
passing it to the model; the resulting shape must match the high-resolution
volume.

For 2D data at scale 4, the layout is::

    deeprock/
        sandstone2D/
            sandstone2D_train_HR/
                sample.png
            sandstone2D_train_BI_unknown_X4/
                sample.png
            sandstone2D_valid_HR/
                sample.png
            sandstone2D_valid_BI_unknown_X4/
                sample.png

The 2D loader uses the same filename for each pair and reads the images at their
stored sizes. Prepare the low-resolution inputs at the spatial size expected by
your model. For other scales, replace ``X4`` in directory names and ``x4`` in 3D
filenames with the selected scale.

Training options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 14 54

   * - Option
     - Default
     - Purpose
   * - ``--dim``
     - ``3``
     - Spatial dimensionality: 2 or 3.
   * - ``--scale``
     - ``4``
     - Select the dataset's downsampling scale.
   * - ``--epochs``
     - ``40``
     - Number of training epochs.
   * - ``--learning-rate``
     - ``0.0001``
     - Optimizer learning rate.
   * - ``--batch-size``
     - ``1``
     - Number of samples per batch.
   * - ``--num-workers``
     - ``4``
     - Number of data-loading workers.
   * - ``--results-dir``
     - ``./results``
     - Training output directory.
   * - ``--use-diffusion`` / ``--no-use-diffusion``
     - Enabled
     - Select diffusion or regression training.
   * - ``--augment`` / ``--no-augment``
     - Enabled
     - Apply training data augmentation.
   * - ``--wandb`` / ``--no-wandb``
     - Disabled
     - Enable Weights & Biases logging.

Use ``supercat-tools train --help`` for the full option list, including model
presets, U-Net selection, transformer dimensions, and checkpoint options.

Pretrain on images or videos
----------------------------------

Pretraining creates low-resolution/high-resolution pairs from ordinary images
or videos. Each sample is converted to grayscale, downsampled by ``--scale``,
and interpolated back to its original spatial size to form the model input.
The original sample provides the training target.

Provide separate training and validation directories. Files are discovered
recursively, so no DeepRock category layout or precomputed image pairs are
needed.

Image pretraining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train a 2D model from directories of PNG or JPEG images::

    supercat-pretrain-image train \
        --training /path/to/images/train \
        --validation /path/to/images/valid \
        --dim 2 \
        --scale 4 \
        --min-size 224 \
        --max-size 224 \
        --epochs 40 \
        --results-dir results/pretrain-image

The image dataset uses bicubic interpolation to create the degraded input.
Setting ``--min-size`` and ``--max-size`` to the same value produces samples
with a consistent spatial size; use an even size compatible with your model.

Video pretraining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train a 3D model using clips from video files::

    supercat-pretrain-movie train \
        --training /path/to/videos/train \
        --validation /path/to/videos/valid \
        --dim 3 \
        --scale 4 \
        --min-size 100 \
        --max-size 100 \
        --max-training-items 1000 \
        --max-validation-items 100 \
        --epochs 40 \
        --results-dir results/pretrain-movie

The video dataset discovers ``.mp4``, ``.avi``, ``.mov``, and ``.mkv`` files.
It treats the frame axis as the third spatial dimension and uses trilinear
interpolation to create the degraded input. A working video decoding backend
is required, such as ``imageio[ffmpeg]`` or an ``ffmpeg`` executable on ``PATH``.

Pretraining options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Option
     - Default
     - Purpose
   * - ``--training``
     - Required
     - Directory of training images or videos.
   * - ``--validation``
     - Required
     - Directory of validation images or videos.
   * - ``--scale``
     - ``4``
     - Downsampling factor used to create degraded inputs.
   * - ``--augment`` / ``--no-augment``
     - Enabled
     - Apply random cropping and augmentation to training samples.
   * - ``--min-size``
     - ``16``
     - Pad dimensions smaller than this size; zero disables padding.
   * - ``--max-size``
     - Images: ``224``; videos: ``100``
     - Crop dimensions larger than this size; zero disables cropping.
   * - ``--max-training-items``
     - All videos
     - Limit the number of training videos; video pretraining only.
   * - ``--max-validation-items``
     - All videos
     - Limit the number of validation videos; video pretraining only.

For images, size limits apply to height and width. For videos, they also apply
to the frame count. Validation uses centered crops without augmentation.
The item limits count video files, not individual frames.

Both commands also accept the model and training options described above,
including ``--no-use-diffusion`` for regression training. Inspect the complete
options with::

    supercat-pretrain-image train --help
    supercat-pretrain-movie train --help

Train with an auxiliary porosity loss
---------------------------------------

Supercat requires WiDiTApp 0.1.14 or newer for its ``loss`` and ``metrics`` hooks.
The DeepRock, image pretraining, and movie pretraining apps reuse
``supercat.metrics.PorosityLoss`` for an optional training objective and the
``porosity_loss`` validation metric.

For supervised training with pixel Smooth L1 and an auxiliary porosity term::

    supercat-tools train \
        --deeprock /path/to/deeprock \
        --dim 3 \
        --no-use-diffusion \
        --loss-fn smoothl1 \
        --porosity-loss-weight 0.001 \
        --porosity-temperature 0.05

The weight above is an example to tune on validation data. ``--porosity-loss-weight``
defaults to zero: the parent loss is returned unchanged and no auxiliary loss
is constructed or evaluated for the objective. With a positive weight,
supervised training uses the parent criterion plus the weighted porosity loss.
The inherited default supervised criterion remains MSE; select
``--loss-fn smoothl1`` for pixel Smooth L1.

For diffusion training::

    supercat-tools train \
        --deeprock /path/to/deeprock \
        --dim 3 \
        --use-diffusion \
        --porosity-loss-weight 0.001 \
        --porosity-temperature 0.05

WiDiTApp adds its standard diffusion objective automatically. Supercat supplies
only the weighted porosity term when the parent diffusion image criterion is
``None``. This term receives the unclipped clean-image estimate, not a fully
sampled image. The same hook options are available on
``supercat-pretrain-image train`` and ``supercat-pretrain-movie train``.

Python callers use the same options::

    from supercat.apps import Supercat

    app = Supercat()
    criterion = app.loss(
        use_diffusion=False, loss_fn="smoothl1", porosity_loss_weight=0.001,
        porosity_temperature=0.05,
    )
    loss = criterion(prediction, target)
    metrics = app.metrics(use_diffusion=False)

All returned losses and metrics are scalar batch means. The training objective
always uses soft sigmoid masks, with gradients controlled by
``--porosity-temperature``. The validation metric always uses hard binary Otsu
masks and its value is independent of temperature. There is no app option to
switch mask types. Both use the HR Otsu threshold for prediction and target;
the metric reports the percentage-ratio loss, not raw predicted porosity.

Validation always includes ``val/porosity_loss``, even at zero auxiliary weight,
and preserves the parent's metrics (including supervised ``val/mse`` and
``val/smoothl1``). Metrics are computed independently for each batch without
gradients. WiDiTApp handles console and W&B logging; the metric does not change
the objective or checkpoint selection. The auxiliary loss does change the
objective when its weight is positive.

These hooks receive only prediction and target images, so they use the existing
two-argument ``PorosityLoss`` interface and compute HR references from each target
batch. Keep ``include_porosity=False`` and omit ``porosity_csv`` for built-in
training; the apps raise a clear error if you ask for metadata without also
passing ``allow_metadata_batches=True``. Precomputed references and four-item
batches below are for custom loops that explicitly pass the metadata to the
loss.

Use precomputed HR porosity in a loss
---------------------------------------

Enable ``include_porosity=True`` when building datasets or dataloaders to receive
four values per batch: ``(lr, hr, hr_threshold, hr_porosity)``. Thresholds use the
same normalized intensity scale as HR images, and porosities are fractions in
[0, 1]. Each reference tensor has shape ``(batch_size,)`` after collation.

For example, in a custom regression training loop::

    import torch
    from supercat.apps import Supercat
    from supercat.metrics import PorosityLoss

    loader, validation_loader = Supercat().dataloaders(
        deeprock="/path/to/deeprock",
        dim=3,
        scale=4,
        batch_size=1,
        num_workers=0,
        include_porosity=True,
        porosity_temperature=0.05,
        allow_metadata_batches=True,
    )
    porosity_loss = PorosityLoss(temperature=0.05, hard_mask=False)
    pixel_loss = torch.nn.SmoothL1Loss()

    # Supply your model, device, optimizer, and tuned porosity_weight.
    for lr, hr, hr_threshold, hr_porosity in loader:
        lr, hr = lr.to(device), hr.to(device)
        prediction = model(lr)
        loss = pixel_loss(prediction, hr) + porosity_weight * porosity_loss(
            prediction, hr_threshold, hr_porosity
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

The loss moves the small reference tensors to the prediction's device and
never runs Otsu on this three-argument path. App datasets always supply soft HR
porosity for training; match ``porosity_temperature`` to the loss's
``temperature``. Hard-mask validation needs a separate binary HR porosity, not
this cached soft value. Built-in validation computes that binary reference from
the target image. For custom loops, the lower-level ``porosity_reference`` and
``PorosityDataset`` utilities still support ``hard_mask=True`` to build separate
metric references. Existing hard-mask CSV caches cannot be used as soft training
references; choose a new cache path or regenerate them.
The loss compares the predicted-to-HR porosity ratio, expressed as a percentage,
against 100, with stabilization near zero HR porosity. Tune its weight alongside
the pixel loss because these terms have different scales.

DeepRock dataset builders accept an optional ``porosity_csv`` path (also exposed
as ``--porosity-csv``). Supplying this path enables porosity metadata:

* If the CSV exists, its references are loaded without recomputing Otsu or HR
  porosity.
* If it does not exist, references for both training and validation are computed
  from HR files and saved for subsequent runs. Parent directories are created.
* With ``porosity_csv=None`` (the default), references are computed on access and
  no CSV is written. Use ``include_porosity=True`` to request these extra values.

For example, add ``porosity_csv="/path/to/porosity.csv"`` to the dataloader call
above. The CSV records ``hr_path``, ``threshold``, ``porosity``, ``temperature``,
``hard_mask``, and ``dtype``. Paths are relative to the DeepRock root and include
the split directories. Existing caches must contain every requested HR file and
match the mask settings; invalid caches raise an error without recomputation.
Use a new CSV path or remove the old CSV when HR content or mask settings change.
The cache does not detect edits to image contents. Flips and rotations preserve
cached references.

The image and movie pretraining apps also accept these options, but compute the
references for each returned crop in the dataset worker, so random cropping and
padding cannot leave stale reference values. Each crop supplies its own Otsu
threshold and porosity.

For a dataset you construct directly, wrap it with
``supercat.data.PorosityDataset(dataset, temperature=0.05, hard_mask=False)``.
Set ``precompute=True`` only when each index always has the same HR intensity
histogram. The existing ``PorosityLoss(prediction, hr)`` call still works, but
computes the references on demand.

Metadata is disabled by default to preserve existing training. The installed
``widitapp`` trainer accepts only two- or three-item batches; these four-item
batches require a custom training loop such as the one above, so the app
datasets refuse them unless ``allow_metadata_batches=True`` acknowledges that
your loop reads them. Enabling metadata
does not supply that metadata to the built-in loss/metric hooks. Use
``--porosity-loss-weight`` to enable the built-in auxiliary objective; its validation
metric is registered independently of dataset metadata.

Calculate porosity
----------------------------------

Calculate porosity for a supported input image or volume::

    supercat-tools porosity --input /path/to/volume.mat

The tool uses Otsu's threshold to classify pixels or voxels below the threshold
as void space, then prints the void fraction as a value between 0 and 1. It
accepts the input formats listed above; TIFF and PyTorch tensor outputs must be
converted to a supported input format before using this command.

Sample a porosity distribution
----------------------------------

Generate multiple diffusion predictions and record the porosity of each::

    supercat-tools porosity-distribution \
        --input /path/to/volume.mat \
        --output results/porosity.csv \
        --checkpoint /path/to/diffusion-checkpoint.pt \
        --size 100 \
        --num-sampling-steps 250 \
        --seed 42 \
        --count 50

This tool requires a diffusion checkpoint with two output channels. Like
``supercat``, it resizes the input to the requested spatial dimensions before
prediction: for this example, a 100 × 100 × 100 volume. Use ``--size-i``,
``--size-j``, and ``--size-k`` to override individual dimensions.

The output CSV contains ``seed,porosity`` columns. ``--count`` defaults to 50
and selects consecutive seeds starting at ``--seed``. Seeds already present in
the output file are skipped, so repeating a command resumes an interrupted
run. Pass ``--overwrite`` to replace an existing CSV. Individual prediction
images are not saved by this tool.

.. end-quickstart

Results
==================================

.. start-results

Test results for the Supercat model are presented at
`unimelbmdap.github.io/supercat-results <https://unimelbmdap.github.io/supercat-results/>`_,
which shows super-resolution outputs for two- and three-dimensional micro-CT
images of rocks.

For example, a carbonate slice from the 2D test set upscaled by a factor of
four with the WiDiT diffusion model:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Low-resolution input (125 × 125)
     - Supercat prediction (500 × 500)
   * - .. image:: https://objects.storage.unimelb.edu.au/4320-supercat/DeepRockSR-2D/carbonate2D/carbonate2D_test_LR_unknown_X4/3601x4.png
          :width: 100%
          :alt: Low-resolution carbonate micro-CT slice, sample 3601
     - .. image:: https://objects.storage.unimelb.edu.au/4320-supercat/predictions/2D/X4/WiDiT/Diffusion/carbonate/3601.png
          :width: 100%
          :alt: Supercat WiDiT diffusion super-resolution prediction, carbonate sample 3601

The results use the DeepRock-SR 2D and 3D datasets introduced by Wang,
Armstrong and Mostaghimi:

* Wang, Y. D., Armstrong, R. T. & Mostaghimi, P. (2020). `Boosting Resolution
  and Recovering Texture of 2D and 3D Micro-CT Images with Deep Learning
  <https://doi.org/10.1029/2019WR026052>`_. *Water Resources Research*, 56(1),
  e2019WR026052.
* Wang, Y. D., Armstrong, R. & Mostaghimi, P. (2019). `A Diverse Super
  Resolution Dataset of Digital Rocks (DeepRock-SR): Sandstone, Carbonate, and
  Coal <https://doi.org/10.17612/s3m9-e024>`_. *Digital Rocks Portal*.

.. end-results

Credits
==================================

.. start-credits

* `Robert Turnbull <https://robturnbull.com>`_,
  `Jonathan Garber <https://www.linkedin.com/in/jonathan-garber-78a84923/>`_,
  `Jay Black <https://findanexpert.unimelb.edu.au/profile/639143-jay-black>`_,
  `Wenbin Fei <https://wenbinfei.github.io/>`_,
  `Tingxuan Wang <https://cis.unimelb.edu.au/people/graduate-researchers/artificial-intelligence/tingxuan-wang>`_,
  `Yu Hsien Chiang <https://github.com/yuhsienchiang>`_
* Publication details to follow
* Created using torchapp (https://github.com/rbturnbull/torchapp)
* Logo derived from https://thenounproject.com/icon/cat-113020/

.. end-credits
