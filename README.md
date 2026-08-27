# Soil Moisture Detection with Sentinel-2

[![Validate repository](https://github.com/hailathamzeh/Soil-moisture-detection/actions/workflows/validate.yml/badge.svg)](https://github.com/hailathamzeh/Soil-moisture-detection/actions/workflows/validate.yml)

An EqraTech research project for estimating surface soil-moisture conditions from Sentinel-2 imagery and historical weather observations. The repository contains the full collection, training, evaluation, and desktop-interface code while keeping private data, API credentials, and large model weights outside GitHub.

## Project overview

The workflow combines:

- Sentinel-2 B8A (narrow near-infrared) and B11 (short-wave infrared) bands
- Open-Meteo soil-moisture observations used as date-aligned labels
- EfficientNetV2-S models adapted from three RGB input channels to two spectral channels
- Classification into low, moderate, and high moisture conditions
- Regression for a continuous soil-moisture estimate
- A PyQt desktop interface for Sentinel Hub download, local batch prediction, and Moisture Stress Index visualization

The classifier thresholds used during training are:

| Class | Soil moisture fraction | Percentage equivalent |
|---|---:|---:|
| Low | `< 0.10` | `< 10%` |
| Moderate | `0.10 to 0.20` | `10% to 20%` |
| High | `> 0.20` | `> 20%` |

## Repository structure

```text
.
├── app/
│   ├── UserInterface.py
│   ├── leaflet_map.html
│   └── test_map.py
├── data/
│   └── README.md
├── models/
│   └── README.md
├── notebooks/
│   ├── data_collection/
│   ├── evaluation/
│   └── training/
├── outputs/
├── tools/
│   └── validate_repository.py
├── .env.example
├── .gitignore
└── requirements.txt
```

Training and evaluation outputs from the original experiments are preserved inside their notebooks. Outputs from the collection notebooks were cleared because they contained generated imagery and machine-specific runtime details.

## Original experiment results

These values are the outputs recorded in the included notebooks and reflect the original train/test splits. They are not presented as external benchmark results.

| Experiment | Recorded result |
|---|---:|
| 1 km classification | 84.38% test accuracy |
| 2 km classification | 85.42% test accuracy |
| 3 km classification | 70.00% test accuracy |
| Regression | MAE 0.0187, RMSE 0.0326 |

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS or Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy the environment template:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# macOS or Linux
cp .env.example .env
```

Add your own Sentinel Hub OAuth credentials to `.env`. Do not commit this file.

## Data and model files

Datasets and checkpoints are intentionally not included. See [data/README.md](data/README.md) for the expected data layout and [models/README.md](models/README.md) for checkpoint names.

You can store data and models outside this repository by changing `SOIL_DATA_DIR`, `SOIL_MODEL_DIR`, and `SOIL_MODEL_PATH` in `.env`. Paths may be absolute or relative to the repository root.

## Run the notebooks

Start Jupyter from the repository root so the shared path configuration resolves consistently:

```bash
jupyter lab
```

A typical order is:

1. `notebooks/data_collection/weather_collection.ipynb`
2. `notebooks/data_collection/sentinel_image_collection.ipynb` or the extended version
3. A notebook under `notebooks/training/`
4. The matching notebook under `notebooks/evaluation/`

The data-collection notebooks call external services and may incur usage subject to those services' terms and quotas.

## Run the desktop interface

Place a trusted classification state dictionary at `models/classification_3km.pth`, or set `SOIL_MODEL_PATH` in `.env`. Then run:

```bash
python app/UserInterface.py
```

The interface supports:

- entering coordinates and a date, then downloading a Sentinel-2 patch through Sentinel Hub
- choosing a location on an embedded Leaflet/OpenStreetMap map
- classifying a local folder of B8A/B11 GeoTIFF pairs by date
- displaying a grayscale composite and the B11/B8A Moisture Stress Index

Online map tiles require an internet connection. Online prediction also requires valid Sentinel Hub credentials.

## Reproducibility notes

- Random train/test splits in the original notebooks use fixed seeds where originally defined, but GPU operations and library versions can still introduce small differences.
- Model weights, private observations, and satellite rasters are external artifacts, so reproducing the exact recorded metrics requires the original data split and checkpoints.
- `requirements.txt` replaces the original machine-specific Rasterio wheel path with installable package versions.
- `tools/validate_repository.py` checks notebook JSON, Python syntax, secret patterns, hard-coded local paths, and disallowed large artifact types.

## Privacy and security

The `.gitignore` excludes credentials, datasets, GeoTIFFs, archives, databases, generated outputs, environments, and model checkpoints. Before publishing any future change, run:

```bash
python tools/validate_repository.py
git status --short
```

If credentials have ever been exposed, rotate them at the provider rather than only deleting them from source code.

## Acknowledgements

- [Copernicus Sentinel-2](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2)
- [Sentinel Hub](https://www.sentinel-hub.com/)
- [Open-Meteo](https://open-meteo.com/)
- [OpenStreetMap](https://www.openstreetmap.org/) and [Leaflet](https://leafletjs.com/)
