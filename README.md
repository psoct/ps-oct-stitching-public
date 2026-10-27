# PS-OCT Stitching Pipeline

An end-to-end post-processing pipeline for automated serial-sectioning
polarization-sensitive optical coherence tomography (PS-OCT) data.

The pipeline performs tile downsampling, flat-field correction, XY stitching,
tissue segmentation, intensity bias-field correction, and Z stitching. The
computed transforms are also applied to retardance and optic-axis orientation
volumes.

## Requirements

- Python 3.9
- Java 8 and Maven (used by PyImageJ and BigStitcher)
- An NVIDIA GPU compatible with CUDA 11.8 for the current SAM configuration
- Sufficient system memory for ImageJ/BigStitcher; adjust
  `general.alloc_memory` in the configuration to match the machine

The default configuration requests a 256 GB Java heap because the original
datasets are large. Reduce this value when working with smaller datasets or on
machines with less memory.

## Installation

The recommended installation uses Mamba or Conda:

```bash
mamba env create -f environment.yml
mamba activate ps-oct-stitching
```

`environment.yml` defines the complete Conda environment, including Java,
CUDA, and PyTorch. `requirements.txt` contains the remaining pip-managed Python
dependencies and is referenced by `environment.yml`.

For a manual installation, create a Python 3.9 environment with Java 8 and
Maven, install the CUDA 11.8 build of PyTorch, and then install the remaining
dependencies:

```bash
mamba create -n ps-oct-stitching -c conda-forge python=3.9 openjdk=8 maven git pip
mamba activate ps-oct-stitching
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## SAM checkpoint

SAM checkpoints are not included. Download one from the official Segment
Anything release:

- [ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- [ViT-L](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- [ViT-B](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

Place the checkpoint in `checkpoints/sam/`. The default pipeline configuration
uses `sam_vit_h_4b8939.pth`. The Segment Anything package is installed from the
official repository at the commit pinned in `requirements.txt`.

## Input data

Input images are NIfTI volumes. For every section identifier, the configured
data directory should contain matching files for the three PS-OCT contrasts:

```text
data/
├── intensity_flat_raw.nii
├── retardance_flat_raw.nii
├── Slice_<section>_*.oct_intensity.nii
├── Slice_<section>_*.oct_retardance.nii
└── Slice_<section>_*.oct_axisorient.nii
```

The repository does not include imaging data. Update
`config/mouse_10x10x10.py` for the dataset being processed, particularly:

- `general.data_dir`, `general.save_root`, and `general.test_slice`
- `general.alloc_memory`
- the flat-field input and output paths under `shading_corr`
- tile counts, overlap, grid direction, and fixed tile under `stitching`
- the SAM prompt box under `segment`
- Z-overlap under `z_stit`; by default, Z-stitching uses the middle section as
  the fixed section, while `z_stit.fixed_id` can specify another zero-based index

The example configuration uses project-relative `data/` and `outputs/`
directories. Its `general.test_slice` list is intentionally empty and must be
filled with the section identifiers for the dataset before running the pipeline.

## Usage

Run the complete pipeline from the repository root:

```bash
python scripts/pipeline_mouse.py --config config/mouse_10x10x10.py
```

PyImageJ downloads the required ImageJ and BigStitcher components on first use,
so the first run requires network access. Intermediate and final volumes are
written below the configured output directory.

## Repository layout

```text
.
├── checkpoints/sam/       # Downloaded SAM checkpoint (not tracked by Git)
├── config/                # Example pipeline configuration
├── scripts/               # End-to-end pipeline entry point
├── utils/                 # Processing and stitching modules
├── environment.yml        # Recommended reproducible environment
├── requirements.txt       # Python dependencies excluding PyTorch
└── README.md
```

## Data and citation

The imaging dataset and its citation information are distributed separately
with the associated publication. Cite that publication when using the dataset
or this processing pipeline.

## Acknowledgements

We thank [Bot Zhao (@ZhaoBotee)](https://github.com/ZhaoBotee) for developing
the original implementation of this pipeline.

## Third-party software

This project uses the following third-party software:

- [Segment Anything](https://github.com/facebookresearch/segment-anything),
  licensed under Apache-2.0. Its source code and model checkpoints are not
  included in this repository.
- [ImageJ2](https://github.com/imagej/imagej2) and
  [BigStitcher](https://github.com/PreibischLab/BigStitcher), which are
  resolved at runtime through PyImageJ and Maven.

Other dependencies are installed separately and remain subject to their
respective upstream licenses.

## License

Copyright 2023-2026 Institute of Automation, Chinese Academy of Sciences.

Licensed under the Apache License, Version 2.0. See `LICENSE` for details.
