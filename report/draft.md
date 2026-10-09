# UNet Binarization of Degraded Historical Manuscripts

## Introduction

Historical document binarization separates foreground writing from paper, stains, and other background marks. This project trains a UNet on DIBCO pages and compares its predictions with conventional thresholding baselines.

TODO: Add project motivation, research question, and citations to DIBCO and prior binarization work.

## Data and split

The manifest contains 136 image/ground-truth pairs across ten available DIBCO years. All listed pairs have matching dimensions; no GT has a dark-pixel fraction above 0.5. 2015 is absent from the available dataset.

| Year | Pages | Split | Notes |
|---|---:|---|---|
| 2009 | 10 | Train | Handwritten and printed |
| 2010 | 10 | Train | Skeleton GT variants excluded |
| 2011 | 16 | Train | 8 handwritten, 8 machine-printed |
| 2012 | 14 | Train | |
| 2013 | 16 | Train | |
| 2014 | 10 | Train | |
| 2016 | 10 | Train | |
| **Train total** | **86** | | |
| 2017 | 20 | Validation | |
| 2018 | 10 | Test | Not evaluated |
| 2019 | 20 | Test | 10 Track A, 10 Track B; not evaluated |
| **Test total** | **30** | | |
| **Total** | **136** | | |

The split is frozen in `data/splits.json`: train on 2009–2014 and 2016, validate on 2017, and test on 2018 and 2019. The final test scores are intentionally pending.

## Methods

The base system is a four-level UNet with batch normalization, RGB input, and base width 16. It trains on random 256×256 patches, with a preference for patches containing ink. Augmentation uses flips, right-angle rotations, brightness/contrast jitter, per-channel gain, and light noise. The loss is an equal-weight combination of binary cross-entropy and Dice loss. Training uses Adam with learning rate 0.001, batch size 16, 1,600 patches per epoch, and a cosine learning-rate schedule for 30 epochs. The best checkpoint is selected by full-page validation F-measure using overlapping-window inference and threshold 0.5.

Ensemble inference averages probability maps from the three base-seed checkpoints. The optional TTA setting averages the original image and horizontal, vertical, and combined flips before thresholding.

TODO: Add implementation and hardware details required by the final report.

## Baselines

| Method | Validation F (%) | Precision (%) | Recall (%) | PSNR (dB) |
|---|---:|---:|---:|---:|
| Otsu | 78.14 | 71.87 | 92.66 | 13.98 |
| Sauvola (window 51, k 0.2) | 82.53 | 83.10 | 84.87 | 15.19 |

Sauvola parameters were selected using training data only. Its 82.53% validation F is the baseline target for the UNet.

## Ablations

### Base run seed variation

| Run | Seed | Best epoch | Best validation F (%) | PSNR at best F (dB) |
|---|---:|---:|---:|---:|
| `unet_base` | 0 | 27 | 91.79 | 18.64 |
| `unet_seed1` | 1 | 17 | 92.37 | 18.88 |
| `unet_seed2` | 2 | 22 | 91.90 | 18.72 |
| **Mean** | | | **92.02** | |

The best-F scores span 0.58 points (sample standard deviation approximately 0.31). The best epoch is selected on this same validation set, so these maxima are optimistic estimates.

### Single-run ablations

| Variant | Change | Best epoch | Best validation F (%) | PSNR at best F (dB) |
|---|---|---:|---:|---:|
| Bleed + low contrast | `--bleed 0.3 --lowcon 0.2` | 16 | 91.12 | 18.17 |
| Grayscale | `--in_ch 1` | 19 | 91.88 | 18.63 |
| BCE only | `--loss bce` | 15 | 92.14 | 18.64 |
| Wider UNet | `--base 32` | 19 | 91.83 | 18.62 |
| Larger patches | `--patch 384 --batch 8` | 15 | 92.12 | 18.75 |

Each ablation was run once. Scores near the three-seed base mean are inconclusive given the observed seed variation; the preselected practical improvement threshold is 92.6% F. The base RGB, BCE+Dice, patch-256, width-16 configuration is retained. The patch-384 run used its recorded patch size at inference.

## Error analysis

Validation page 13 has high recall but low precision across all three base seeds, consistent with show-through being mistaken for foreground ink. The page-13 scores are:

| Run | F (%) | Precision (%) | Recall (%) | PSNR (dB) |
|---|---:|---:|---:|---:|
| `unet_base` | 77.85 | 64.89 | 97.29 | 14.06 |
| `unet_seed1` | 76.99 | 63.82 | 97.01 | 13.86 |
| `unet_seed2` | 78.40 | 65.99 | 96.57 | 14.23 |

The error map uses red for false positives, blue for false negatives, and gray for true positives: `outputs/page13_errors.png`.

Page 17 varies substantially between seeds:

| Run | F (%) | Precision (%) | Recall (%) | PSNR (dB) |
|---|---:|---:|---:|---:|
| `unet_base` (seed 0) | 83.05 | 92.93 | 75.08 | 17.97 |
| `unet_seed1` | 92.70 | 89.49 | 96.15 | 21.03 |
| `unet_seed2` | 85.59 | 94.41 | 78.27 | 18.62 |

The combined bleed/low-contrast augmentation and BCE-only loss improved page 17 in their single runs, but page 13 remained difficult. These page-specific observations do not establish a general improvement.

## Results

The ensemble evaluation used only the 20-page 2017 validation split. Metrics are means over pages; the single-model rows average scores across the three seeds.

| System | F (%) | Precision (%) | Recall (%) | PSNR (dB) | Page 13 F (%) | Page 17 F (%) |
|---|---:|---:|---:|---:|---:|---:|
| Single-model mean | 92.02 | 91.71 | 92.85 | 18.74 | 77.75 | 87.11 |
| Probability ensemble | 92.31 | 92.02 | 93.07 | 18.90 | 77.77 | 88.43 |
| Single-model mean + TTA | 92.29 | 92.09 | 92.99 | 18.91 | 78.07 | 88.59 |
| Ensemble + TTA | 92.48 | 92.24 | 93.18 | 18.99 | 78.08 | 90.16 |

The ensemble adoption rule was fixed in advance at validation F ≥ 92.6%. Ensemble + TTA reached 92.48%, so the decision is **do not adopt the ensemble**; retain the base model configuration. The complete table is saved in `outputs/ensemble_val.csv`.

TODO: Add final test results only after explicit approval to evaluate the frozen test split.

## Limitations

- Validation contains 20 pages from one year; best-checkpoint selection on that set introduces selection optimism.
- Each architectural, loss, or augmentation ablation was run with one seed; only the base configuration has three seeds.
- Pseudo-F-measure and DRD are not included in the current metrics.
- Track-specific 2019 test results and the OCR comparison remain TODO until the test evaluation is approved and settings are frozen.
- TODO: Add dataset citations, qualitative figure references, and a discussion of generalization beyond these DIBCO pages.
