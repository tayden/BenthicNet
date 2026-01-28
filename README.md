# BenthicNet

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

BenthicNet is a large-scale dataset containing underwater photographs of benthic habitats on the seafloor. This repository provides tools for downloading, processing, and managing the BenthicNet imagery collection.

## Dataset

The BenthicNet CSV metadata files required for downloading the images are available at:

**[https://www.frdr-dfdr.ca/repo/dataset/24c6c813-d0ff-4461-8174-3f960f1edc0d](https://www.frdr-dfdr.ca/repo/dataset/24c6c813-d0ff-4461-8174-3f960f1edc0d)**

Download the CSV files from FRDR, then use the tools in this repository to download and process the imagery.

## Installation

### Basic Installation

```bash
pip install -e .
```

### With Plotting Support

Plotting features require cartopy, which has system dependencies:

```bash
# Ubuntu/Debian
sudo apt-get install libproj-dev proj-data proj-bin libgeos-dev

# Install with conda (recommended for cartopy)
conda create --name benthicnet -y
conda activate benthicnet
conda install -c conda-forge cartopy -y
pip install -e .[plot]
```

### Development Installation

```bash
pip install -e .[dev]
pre-commit install
```

## Usage

### Download Images to Directory

Download images listed in a CSV file to a directory structure:

```bash
benthicnet-download input.csv output_dir/
```

### Download Images to Tarballs

Download images into tarball archives (one per dataset):

```bash
benthicnet-download-tar input.csv output_dir/
```

This creates:
- `output_dir/tar/<dataset>.tar` - Images archived as `<site>/<image>.jpg`
- `output_dir/csv/<dataset>.csv` - Metadata for successfully downloaded images
- `output_dir/errors/<dataset>.log` - Failed URLs (if any)

### Convert Images

Convert images in tarballs to smaller JPEGs:

```bash
benthicnet-convert-tar input_dir/ output_dir/ --csv-todo todo.csv --target-len 512 --jpeg-quality 95
```

### Subsample by Distance

Subsample a CSV file based on spatial distance between samples:

```bash
benthicnet-subsample input.csv output.csv --distance 2.5 --target-population 1000
```

### Transfer Between Tarballs

Copy files between tarballs by matching URLs:

```bash
benthicnet-tar2tar source_tar_dir/ dest_tar_dir/ source.csv dest.csv
```

## Common Options

All CLI tools support:

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Increase verbosity (can be repeated) |
| `-q, --quiet` | Decrease verbosity (can be repeated) |
| `--no-progress-bar` | Disable tqdm progress bar |
| `--nproc N` | Number of parallel processes |
| `--iproc I` | Process index (0 to N-1) for parallel execution |

## Parallel Processing

For large downloads, run multiple processes in parallel:

```bash
# Terminal 1
benthicnet-download-tar input.csv output_dir/ --nproc 4 --iproc 0

# Terminal 2
benthicnet-download-tar input.csv output_dir/ --nproc 4 --iproc 1

# Terminal 3
benthicnet-download-tar input.csv output_dir/ --nproc 4 --iproc 2

# Terminal 4
benthicnet-download-tar input.csv output_dir/ --nproc 4 --iproc 3
```

## CSV Format

BenthicNet CSV files contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `dataset` | string | Dataset name |
| `site` | string | Site/deployment name |
| `image` | string | Image filename |
| `url` | string | URL to download the image |
| `latitude` | float | Latitude coordinate |
| `longitude` | float | Longitude coordinate |
| `datetime` | datetime | Timestamp of image capture |
| `depth` | float | Water depth (meters) |
| `altitude` | float | Altitude above seafloor (meters) |
| `platform` | string | Imaging platform type |

## Python API

```python
import benthicnet.io as io
from benthicnet.download_images_tar import download_images_by_dataset_from_csv
from benthicnet.subsample import subsample_distance_sitewise

# Read a BenthicNet CSV file
df = io.read_csv("benthicnet.csv")

# Download images to tarballs
download_images_by_dataset_from_csv("input.csv", "output_dir/")

# Subsample by distance
df_subsampled = subsample_distance_sitewise(df, distance=2.5, target_population=1000)
```

## License

MIT License
