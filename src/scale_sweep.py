"""Seven-factor validation-only v1 scale sweep; never loads test pages."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.filters import threshold_sauvola

from dataset import load_split
from infer import load_model, predict_page
from metrics import score


FACTORS = (0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)
CHECKPOINTS = (
    ("unet_base", Path("outputs/checkpoints/unet_base_best.pt")),
    ("unet_seed1", Path("outputs/checkpoints/unet_seed1_best.pt")),
    ("unet_seed2", Path("outputs/checkpoints/unet_seed2_best.pt")),
)
METRICS = ("F", "P", "R", "PSNR")


def resize_page_image(image_path, factor):
    with Image.open(image_path) as src:
        width, height = src.size
        size = (max(1, round(width * factor)), max(1, round(height * factor)))
        rgb = src.convert("RGB").resize(size, Image.Resampling.BICUBIC)
        gray = src.convert("L").resize(size, Image.Resampling.BICUBIC)
    rgb_array = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return rgb_array, np.asarray(gray)


def resize_gt_area(gt_path, factor):
    with Image.open(gt_path) as src:
        width, height = src.size
        size = (max(1, round(width * factor)), max(1, round(height * factor)))
        original = np.asarray(src.convert("L")) < 128
        mask_image = Image.fromarray(original.astype(np.uint8) * 255)
        # BOX resampling computes area coverage before the 0.5 foreground threshold.
        resized = mask_image.resize(size, Image.Resampling.BOX)
    return original, np.asarray(resized) >= 128


def summarize(per_page, factors, gt_counts, n_pages):
    rows = []
    plot = {"unet_mean": [], "unet_min": [], "unet_max": [], "sauvola": []}
    for factor in factors:
        seed_summaries = {}
        for name, _ in CHECKPOINTS:
            scores = per_page[(factor, name)]
            seed_summaries[name] = {metric: float(np.mean([s[metric] for s in scores])) for metric in METRICS}
            rows.append({"scale_factor": factor, "method": name, **seed_summaries[name],
                         "F_seed_min": "", "F_seed_max": "", "n_pages": n_pages,
                         **gt_counts[factor]})
        seed_f = [seed_summaries[name]["F"] for name, _ in CHECKPOINTS]
        seed_mean = {metric: float(np.mean([seed_summaries[name][metric] for name, _ in CHECKPOINTS]))
                     for metric in METRICS}
        rows.append({"scale_factor": factor, "method": "UNet single-model mean", **seed_mean,
                     "F_seed_min": min(seed_f), "F_seed_max": max(seed_f), "n_pages": n_pages,
                     **gt_counts[factor]})

        sauvola_scores = per_page[(factor, "Sauvola (window 51, k 0.2)")]
        sauvola_mean = {metric: float(np.mean([s[metric] for s in sauvola_scores])) for metric in METRICS}
        rows.append({"scale_factor": factor, "method": "Sauvola (window 51, k 0.2)", **sauvola_mean,
                     "F_seed_min": "", "F_seed_max": "", "n_pages": n_pages,
                     **gt_counts[factor]})
        plot["unet_mean"].append(seed_mean["F"])
        plot["unet_min"].append(min(seed_f))
        plot["unet_max"].append(max(seed_f))
        plot["sauvola"].append(sauvola_mean["F"])
    return rows, plot


def write_rows(path, rows):
    fields = ("scale_factor", "method", "F", "P", "R", "PSNR", "F_seed_min", "F_seed_max",
              "n_pages", "gt_ink_pixels_original_total", "gt_ink_pixels_resized_total",
              "gt_pixels_lost_vs_original", "gt_area_loss_original_equiv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    # Deliberately load only the frozen validation list for model evaluation.
    val_items = load_split("data/splits.json", "val")
    if len(val_items) != 20 or any(item["year"] != "2017" for item in val_items):
        raise RuntimeError("Expected exactly 20 pages from validation year 2017")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for name, checkpoint in CHECKPOINTS:
        model, in_ch, patch = load_model(str(checkpoint), device)
        if in_ch != 3:
            raise ValueError(f"Expected RGB checkpoint {name}, got in_ch={in_ch}")
        models[name] = (model, patch)

    per_page = {}
    gt_counts = {factor: {"gt_ink_pixels_original_total": 0,
                          "gt_ink_pixels_resized_total": 0,
                          "gt_pixels_lost_vs_original": 0,
                          "gt_area_loss_original_equiv": 0.0}
                 for factor in FACTORS}
    for page_index, item in enumerate(val_items, start=1):
        for factor in FACTORS:
            image, gray = resize_page_image(item["image"], factor)
            original_gt, gt = resize_gt_area(item["gt"], factor)
            # Lost foreground area is reported as original-resolution-equivalent pixels.
            original_area = original_gt.size
            resized_area = gt.size
            expected_scaled_ink = original_gt.sum() * resized_area / original_area
            gt_counts[factor]["gt_ink_pixels_original_total"] += int(original_gt.sum())
            gt_counts[factor]["gt_ink_pixels_resized_total"] += int(gt.sum())
            gt_counts[factor]["gt_area_loss_original_equiv"] += (
                max(0.0, expected_scaled_ink - int(gt.sum())) * original_area / resized_area
            )

            sauvola_pred = gray < threshold_sauvola(gray, window_size=51, k=0.2)
            per_page.setdefault((factor, "Sauvola (window 51, k 0.2)"), []).append(score(sauvola_pred, gt))
            for name, _ in CHECKPOINTS:
                model, patch = models[name]
                probability = predict_page(model, image, patch=patch, overlap=patch // 4, device=device)
                per_page.setdefault((factor, name), []).append(score(probability >= 0.5, gt))
        print(f"validation page {page_index}/{len(val_items)}", flush=True)

    for factor in FACTORS:
        original_count = gt_counts[factor]["gt_ink_pixels_original_total"]
        resized_count = gt_counts[factor]["gt_ink_pixels_resized_total"]
        gt_counts[factor]["gt_pixels_lost_vs_original"] = max(0, original_count - resized_count)
    rows, plot = summarize(per_page, FACTORS, gt_counts, len(val_items))
    write_rows(Path("outputs/val_scale_sensitivity.csv"), rows)
    fig, axis = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    axis.plot(FACTORS, plot["unet_mean"], marker="o", label="UNet seed mean")
    axis.fill_between(FACTORS, plot["unet_min"], plot["unet_max"], alpha=0.2, label="UNet seed min-max")
    axis.plot(FACTORS, plot["sauvola"], marker="s", label="Sauvola (51, 0.2)")
    axis.set_xlabel("Image scale factor")
    axis.set_ylabel("Mean validation F-measure (%)")
    axis.set_xticks(FACTORS)
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.savefig("outputs/val_scale_sensitivity.png", dpi=180)
    plt.close(fig)

    print("validation-only sweep complete; test split was not loaded")
    for factor in FACTORS:
        print(f"scale {factor:.2f}: original GT ink={gt_counts[factor]['gt_ink_pixels_original_total']}, "
              f"resized GT ink={gt_counts[factor]['gt_ink_pixels_resized_total']}, "
              f"lost vs original={gt_counts[factor]['gt_pixels_lost_vs_original']}, "
              f"area-adjusted loss={gt_counts[factor]['gt_area_loss_original_equiv']:.1f} original-equivalent px")


if __name__ == "__main__":
    main()
