# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BenthicNet is a dataset and toolkit for managing underwater photographs of benthic (seafloor) habitats. The package provides utilities for downloading, processing, converting, and partitioning large collections of benthic imagery stored in CSV metadata files and tarball archives.

## Build and Install

```bash
# Basic install
pip install -e .

# With plotting support (requires cartopy dependencies - see README.rst)
pip install -e .[plot]

# Development install
pip install -e .[dev]
```

## Code Quality

Pre-commit hooks are configured for linting and formatting:
```bash
pre-commit install
pre-commit run --all-files
```

Uses **black** (profile: black) and **isort** (profile: black) for formatting. Target Python versions: 3.7-3.11.

## CLI Commands

The package provides five CLI entry points:

- `benthicnet-download` - Download images from CSV file to directory structure
- `benthicnet-download-tar` - Download images from CSV file into tarballs (one per dataset)
- `benthicnet-convert-tar` - Convert images in tarballs to smaller JPEGs
- `benthicnet-subsample` - Subsample CSV by spatial distance between samples
- `benthicnet-tar2tar` - Copy files between tarballs by matching URLs

All CLIs support `--verbose/-v` (increase verbosity), `--quiet/-q` (decrease verbosity), and parallel processing via `--nproc`/`--iproc` for partitioning work across multiple processes.

## Architecture

### Core Data Flow

CSV files contain image metadata (dataset, site, image, url, latitude, longitude, datetime, depth, etc.) → Images are downloaded/processed into tarballs organized by dataset → CSVs are updated to reflect actual downloaded files.

### Key Modules

- **benthicnet/io.py** - CSV reading with BenthicNet dtypes, filename sanitization, output path determination. Key functions: `read_csv()`, `sanitize_filename_series()`, `determine_outpath()`, `row2basename()`.

- **benthicnet/download_images.py** - Download individual images to directory structure. Handles rate limiting for pangaea.de (max 180 requests/30s).

- **benthicnet/download_images_tar.py** - Download images directly into tarballs, organized by dataset. Creates per-dataset CSV and error logs.

- **benthicnet/convert_images_tar.py** - Convert images between tarballs (resize, JPEG compression). Aligns source/destination by URL or output path.

- **benthicnet/subsample.py** - Spatial subsampling using Haversine distances and BallTree. Key function: `subsample_distance_sitewise()` handles per-site subsampling with configurable target populations and automatic subsite detection.

- **benthicnet/partition.py** - Train/test partitioning using spatial coordinates to ensure geographic separation between partitions. Uses BallTree for efficient spatial queries.

- **benthicnet/tar2tar.py** - Transfer files between tarballs by matching URLs between source and destination CSVs.

### Output Structure

Download operations create:
```
output_dir/
  tar/dataset_name.tar    # Images archived as site/image.jpg
  csv/dataset_name.csv    # Metadata for successfully downloaded images
  errors/dataset_name.log # Failed URLs (if any)
```

### Key Patterns

- DataFrames use columns: dataset, site, image, url, latitude, longitude, datetime, depth, altitude, platform
- Output paths follow pattern: `<dataset>/<site>/<image.ext>`
- Tarball member paths: `<site>/<image.ext>` (dataset is the tarball name)
- Rate limiting implemented for pangaea.de with retry logic and Retry-After header support
