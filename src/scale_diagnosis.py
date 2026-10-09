"""Descriptive GT statistics and validation-only image-scale sensitivity experiment."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_sauvola
from skimage.morphology import skeletonize

from infer import load_model, predict_page
from metrics import score


SPLITS_PATH = Path("data/splits.json")
MANIFEST_PATH = Path("data/manifest.csv")
CHECKPOINTS = (
    ("unet_base", Path("outputs/checkpoints/unet_base_best.pt")),
    ("unet_seed1", Path("outputs/checkpoints/unet_seed1_best.pt")),
    ("unet_seed2", Path("outputs/checkpoints/unet_seed2_best.pt")),
)
FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0)
METRICS = ("F", "P", "R", "PSNR")
INK_THRESHOLD = 128


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def page_key(item):
    return str(item["year"]), str(item.get("track", "")), str(item["id"])


def test_split_label(item):
    if item["year"] == "2018":
        return "2018"
    if item["year"] == "2019" and item["track"] in ("A", "B"):
        return "2019" + item["track"]
    raise ValueError(f"Unexpected test year/track: {item['year']}/{item['track']}")


def output_csv(path, rows, fields, note=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if note:
            handle.write(f"# {note}\n")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_page_groups(manifest, splits):
    manifest_map = {page_key(row): row for row in manifest}
    return {
        "train": [manifest_map[page_key(item)] for item in splits["train"]],
        "val": [manifest_map[page_key(item)] for item in splits["val"]],
        "2018": [row for row in manifest if row["year"] == "2018"],
        "2019A": [row for row in manifest if row["year"] == "2019" and row["track"] == "A"],
        "2019B": [row for row in manifest if row["year"] == "2019" and row["track"] == "B"],
    }


def descriptive_statistics(groups):
    stroke_rows = []
    appearance_rows = []
    for group, pages in groups.items():
        page_widths = []
        ink_medians, nonink_medians, contrasts = [], [], []
        for page in pages:
            gt = np.asarray(Image.open(page["gt"]).convert("L")) < INK_THRESHOLD
            distance = distance_transform_edt(gt)
            skeleton = skeletonize(gt)
            skeleton_distances = distance[skeleton]
            if skeleton_distances.size == 0:
                raise ValueError(f"GT contains no skeleton pixels: {page['gt']}")
            page_widths.append(2.0 * float(np.median(skeleton_distances)))

            gray = np.asarray(Image.open(page["image"]).convert("L"))
            ink_values, nonink_values = gray[gt], gray[~gt]
            if not ink_values.size or not nonink_values.size:
                raise ValueError(f"GT has an empty class: {page['gt']}")
            ink_med = float(np.median(ink_values))
            nonink_med = float(np.median(nonink_values))
            ink_medians.append(ink_med)
            nonink_medians.append(nonink_med)
            contrasts.append(nonink_med - ink_med)

        stroke_rows.append({
            "split": group,
            "n_pages": len(page_widths),
            "stroke_width_median_px": float(np.median(page_widths)),
            "stroke_width_min_px": float(np.min(page_widths)),
            "stroke_width_max_px": float(np.max(page_widths)),
        })
        appearance_rows.append({
            "split": group,
            "n_pages": len(ink_medians),
            "ink_gray_median": float(np.median(ink_medians)),
            "ink_gray_min": float(np.min(ink_medians)),
            "ink_gray_max": float(np.max(ink_medians)),
            "nonink_gray_median": float(np.median(nonink_medians)),
            "nonink_gray_min": float(np.min(nonink_medians)),
            "nonink_gray_max": float(np.max(nonink_medians)),
            "contrast_median": float(np.median(contrasts)),
            "contrast_min": float(np.min(contrasts)),
            "contrast_max": float(np.max(contrasts)),
        })

    test_note = "GT for test 2018/2019A/2019B was used only for descriptive statistics, not for any selection."
    output_csv(
        Path("outputs/stroke_width_stats.csv"), stroke_rows,
        ("split", "n_pages", "stroke_width_median_px", "stroke_width_min_px", "stroke_width_max_px"),
        note=test_note,
    )
    output_csv(
        Path("outputs/appearance_stats.csv"), appearance_rows,
        ("split", "n_pages", "ink_gray_median", "ink_gray_min", "ink_gray_max",
         "nonink_gray_median", "nonink_gray_min", "nonink_gray_max",
         "contrast_median", "contrast_min", "contrast_max"),
        note=test_note,
    )
    return stroke_rows, appearance_rows


def resized_rgb_and_gray(path, factor):
    with Image.open(path) as source:
        width, height = source.size
        target = (max(1, round(width * factor)), max(1, round(height * factor)))
        rgb = source.convert("RGB").resize(target, Image.Resampling.BICUBIC)
        gray = source.convert("L").resize(target, Image.Resampling.BICUBIC)
    rgb_array = np.asarray(rgb, dtype=np.float32) / 255.0
    return rgb_array.transpose(2, 0, 1), np.asarray(gray)


def resized_gt(path, factor):
    with Image.open(path) as source:
        width, height = source.size
        target = (max(1, round(width * factor)), max(1, round(height * factor)))
        mask = source.convert("L").resize(target, Image.Resampling.NEAREST)
    return np.asarray(mask) < INK_THRESHOLD


def evaluate_validation(val_items):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for name, checkpoint in CHECKPOINTS:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model, in_ch, patch = load_model(str(checkpoint), device)
        if in_ch != 3:
            raise ValueError(f"Expected the frozen RGB checkpoint for {name}, got in_ch={in_ch}")
        models[name] = {"model": model, "patch": patch}

    per_page = defaultdict(list)
    for page_index, item in enumerate(val_items, start=1):
        for factor in FACTORS:
            gt = resized_gt(item["gt"], factor)
            gray = resized_rgb_and_gray(item["image"], factor)[1]
            sauvola = gray < threshold_sauvola(gray, window_size=51, k=0.2)
            per_page[(factor, "Sauvola")].append(score(sauvola, gt))

            for name, _ in CHECKPOINTS:
                image, _ = resized_rgb_and_gray(item["image"], factor)
                entry = models[name]
                probability = predict_page(
                    entry["model"], image, patch=entry["patch"],
                    overlap=entry["patch"] // 4, device=device,
                )
                per_page[(factor, name)].append(score(probability >= 0.5, gt))
        print(f"validation scale page {page_index}/{len(val_items)}", flush=True)

    rows = []
    seed_mean_f = []
    seed_min_f = []
    seed_max_f = []
    sauvola_f = []
    for factor in FACTORS:
        seed_summaries = {}
        for name, _ in CHECKPOINTS:
            values = per_page[(factor, name)]
            seed_summaries[name] = {
                metric: float(np.mean([row[metric] for row in values]))
                for metric in METRICS
            }
            rows.append({"scale_factor": factor, "method": name, **seed_summaries[name],
                         "F_seed_min": "", "F_seed_max": "", "n_pages": len(val_items)})

        all_seed_f = [seed_summaries[name]["F"] for name, _ in CHECKPOINTS]
        seed_mean = {metric: float(np.mean([seed_summaries[name][metric] for name, _ in CHECKPOINTS]))
                     for metric in METRICS}
        rows.append({"scale_factor": factor, "method": "UNet single-model mean", **seed_mean,
                     "F_seed_min": min(all_seed_f), "F_seed_max": max(all_seed_f),
                     "n_pages": len(val_items)})
        seed_mean_f.append(seed_mean["F"])
        seed_min_f.append(min(all_seed_f))
        seed_max_f.append(max(all_seed_f))

        sauvola_values = per_page[(factor, "Sauvola")]
        sauvola_mean = {metric: float(np.mean([row[metric] for row in sauvola_values]))
                        for metric in METRICS}
        rows.append({"scale_factor": factor, "method": "Sauvola (window 51, k 0.2)",
                     **sauvola_mean, "F_seed_min": "", "F_seed_max": "",
                     "n_pages": len(val_items)})
        sauvola_f.append(sauvola_mean["F"])

    output_csv(
        Path("outputs/val_scale_sensitivity.csv"), rows,
        ("scale_factor", "method", "F", "P", "R", "PSNR", "F_seed_min", "F_seed_max", "n_pages"),
    )
    fig, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    axis.plot(FACTORS, seed_mean_f, marker="o", label="UNet seed mean")
    axis.fill_between(FACTORS, seed_min_f, seed_max_f, alpha=0.2, label="UNet seed min-max")
    axis.plot(FACTORS, sauvola_f, marker="s", label="Sauvola (51, 0.2)")
    axis.set_xlabel("Image scale factor")
    axis.set_ylabel("Mean validation F-measure (%)")
    axis.set_xticks(FACTORS)
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.savefig("outputs/val_scale_sensitivity.png", dpi=180)
    plt.close(fig)
    return rows


def main():
    manifest = read_csv(MANIFEST_PATH)
    with SPLITS_PATH.open(encoding="utf-8") as handle:
        splits = json.load(handle)
    groups = build_page_groups(manifest, splits)
    stroke_rows, appearance_rows = descriptive_statistics(groups)

    # The model experiment receives only the explicitly selected validation list.
    val_items = splits["val"]
    if len(val_items) != 20 or any(item["year"] != "2017" for item in val_items):
        raise RuntimeError("Expected exactly the 20 frozen 2017 validation pages")
    scale_rows = evaluate_validation(val_items)

    print(f"Wrote descriptive stroke-width and appearance stats for "
          f"{sum(row['n_pages'] for row in stroke_rows)} split entries.")
    print(f"Wrote {len(scale_rows)} validation scale-sensitivity rows.")
    print("Test GT was used only for descriptive statistics; no test model evaluation was performed.")


if __name__ == "__main__":
    main()
