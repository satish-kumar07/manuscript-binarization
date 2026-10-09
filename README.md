# manuscript-binarization

UNet-based binarization of degraded historical manuscripts using the DIBCO datasets.

## Project layout

- `src/` — data inspection, manifest construction, training, inference, metrics, and analysis scripts.
- `data/manifest.csv` — image and ground-truth pairs; the image datasets themselves are not included.
- `data/splits.json` — frozen train, validation, and test assignments.
- `outputs/` — experiment logs and validation result tables. Model checkpoints, predictions, and generated figures are excluded.
- `report/` — project report draft.

## Setup

Install the Python dependencies used by the scripts, including PyTorch, Pillow, NumPy, and scikit-image. Place the DIBCO image datasets in the expected `data/` layout before running the scripts.

Run scripts from the repository root with `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH = 'src'`). Training and inference commands depend on the dataset and a suitable PyTorch installation.
