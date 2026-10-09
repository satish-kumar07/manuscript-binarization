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

## Results

The frozen test split was scored once under the evaluation protocol recorded in `report/draft.md`. Values are per-page means by group; the single-model mean averages the three seed runs, and the listed F range is across seed-level group means. The ensemble with TTA is secondary/exploratory.

| System | Group | Pages | F (%) | Precision (%) | Recall (%) | PSNR (dB) | Three-seed F range (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| Otsu | 2018 | 10 | 51.45 | 42.50 | 78.85 | 9.79 | - |
| Sauvola (window 51, k 0.2) | 2018 | 10 | 64.79 | 63.06 | 75.10 | 13.15 | - |
| unet_base | 2018 | 10 | 83.64 | 81.24 | 87.77 | 16.84 | - |
| unet_seed1 | 2018 | 10 | 83.42 | 80.93 | 87.53 | 16.92 | - |
| unet_seed2 | 2018 | 10 | 82.61 | 80.01 | 87.33 | 16.63 | - |
| UNet single-model mean | 2018 | 10 | 83.22 | 80.73 | 87.54 | 16.80 | 82.61-83.64 |
| Ensemble + TTA (exploratory) | 2018 | 10 | 84.31 | 82.76 | 87.45 | 17.11 | - |
| Otsu | 2019A | 10 | 72.75 | 65.56 | 92.10 | 15.50 | - |
| Sauvola (window 51, k 0.2) | 2019A | 10 | 70.48 | 60.16 | 93.15 | 14.77 | - |
| unet_base | 2019A | 10 | 56.28 | 43.34 | 82.19 | 12.34 | - |
| unet_seed1 | 2019A | 10 | 60.99 | 47.34 | 86.67 | 12.83 | - |
| unet_seed2 | 2019A | 10 | 54.07 | 42.46 | 77.41 | 12.26 | - |
| UNet single-model mean | 2019A | 10 | 57.11 | 44.38 | 82.09 | 12.48 | 54.07-60.99 |
| Ensemble + TTA (exploratory) | 2019A | 10 | 58.46 | 45.56 | 82.86 | 12.68 | - |
| Otsu | 2019B | 10 | 23.52 | 13.54 | 99.91 | 2.82 | - |
| Sauvola (window 51, k 0.2) | 2019B | 10 | 56.10 | 43.67 | 83.85 | 10.10 | - |
| unet_base | 2019B | 10 | 56.94 | 55.97 | 61.64 | 12.19 | - |
| unet_seed1 | 2019B | 10 | 61.30 | 58.48 | 67.72 | 12.43 | - |
| unet_seed2 | 2019B | 10 | 57.98 | 57.92 | 61.31 | 12.22 | - |
| UNet single-model mean | 2019B | 10 | 58.74 | 57.46 | 63.55 | 12.28 | 56.94-61.30 |
| Ensemble + TTA (exploratory) | 2019B | 10 | 59.60 | 58.71 | 63.94 | 12.47 | - |
| Otsu | ALL | 30 | 49.24 | 40.53 | 90.29 | 9.37 | - |
| Sauvola (window 51, k 0.2) | ALL | 30 | 63.79 | 55.63 | 84.03 | 12.67 | - |
| unet_base | ALL | 30 | 65.62 | 60.18 | 77.20 | 13.79 | - |
| unet_seed1 | ALL | 30 | 68.57 | 62.25 | 80.64 | 14.06 | - |
| unet_seed2 | ALL | 30 | 64.88 | 60.13 | 75.35 | 13.71 | - |
| UNet single-model mean | ALL | 30 | 66.36 | 60.86 | 77.73 | 13.85 | 64.88-68.57 |
| Ensemble + TTA (exploratory) | ALL | 30 | 67.46 | 62.34 | 78.08 | 14.09 | - |

### Reproduction and provenance

The recorded one-time test evaluation command was:

```powershell
python src/test_eval.py
```

The post-test statistics and gallery can be regenerated from saved predictions/scores and the local images with `python src/post_test_analysis.py`; this script does not load a model or perform inference. The one-time evaluation used these frozen inputs (SHA-256, from `outputs/test_provenance.txt`):

| File | SHA-256 |
|---|---|
| `data/splits.json` | `CA38FCD262E007ABABE783FEBB4D0C94429064EF1CE4593A244E2E8CEF4A2420` |
| `outputs/checkpoints/unet_base_best.pt` | `2EC16AF261BDF9187DFBB8C650532BDD3BA23FCA109851555D55320F3138209A` |
| `outputs/checkpoints/unet_seed1_best.pt` | `7CB2EFA8EB1ECC5DCCB6A614E4D5F9741850F9BCEE93617A8040146E4A052CF7` |
| `outputs/checkpoints/unet_seed2_best.pt` | `006738B0FD888E9743DB7D32A9817D38E6E925964118C0ED765D2D1CB553A32B` |

The test image data and checkpoint files must be available locally to reproduce the evaluation.
