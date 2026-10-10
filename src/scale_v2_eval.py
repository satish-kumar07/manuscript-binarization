"""Evaluate trained v2 checkpoints on validation pages only, alongside saved v1."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import load_split
from infer import load_model, predict_page
from metrics import score
from scale_sweep import FACTORS, METRICS, resize_gt_area, resize_page_image


V1_PATH = Path("outputs/val_scale_sensitivity.csv")
V2_CHECKPOINTS = (
    ("unet_v2_s0", Path("outputs/checkpoints/unet_v2_s0_best.pt")),
    ("unet_v2_s1", Path("outputs/checkpoints/unet_v2_s1_best.pt")),
    ("unet_v2_s2", Path("outputs/checkpoints/unet_v2_s2_best.pt")),
)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    # The evaluator never reads the test split.
    val_items = load_split("data/splits.json", "val")
    if len(val_items) != 20 or any(item["year"] != "2017" for item in val_items):
        raise RuntimeError("Expected exactly 20 pages from validation year 2017")
    v1_rows = read_rows(V1_PATH)
    v1_by_factor_method = {(float(row["scale_factor"]), row["method"]): row for row in v1_rows}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for name, checkpoint in V2_CHECKPOINTS:
        model, in_ch, patch = load_model(str(checkpoint), device)
        if in_ch != 3:
            raise ValueError(f"Expected RGB checkpoint {name}, got in_ch={in_ch}")
        models[name] = (model, patch)

    page_scores = {}
    for page_index, item in enumerate(val_items, start=1):
        for factor in FACTORS:
            image, _ = resize_page_image(item["image"], factor)
            _, gt = resize_gt_area(item["gt"], factor)
            for name, _ in V2_CHECKPOINTS:
                model, patch = models[name]
                prob = predict_page(model, image, patch=patch, overlap=patch // 4, device=device)
                page_scores.setdefault((factor, name), []).append(score(prob >= 0.5, gt))
        print(f"validation page {page_index}/{len(val_items)}", flush=True)

    output_rows = []
    plot = {"v1_mean": [], "v1_min": [], "v1_max": [],
            "v2_mean": [], "v2_min": [], "v2_max": [], "sauvola": []}
    for factor in FACTORS:
        v1_mean_row = v1_by_factor_method[(factor, "UNet single-model mean")]
        gt_fields = {
            "gt_ink_pixels_original_total": int(float(v1_mean_row["gt_ink_pixels_original_total"])),
            "gt_ink_pixels_resized_total": int(float(v1_mean_row["gt_ink_pixels_resized_total"])),
            "gt_pixels_lost_vs_original": int(float(v1_mean_row["gt_pixels_lost_vs_original"])),
            "gt_area_loss_original_equiv": float(v1_mean_row["gt_area_loss_original_equiv"]),
        }
        v1_mean = {metric: float(v1_mean_row[metric]) for metric in METRICS}
        output_rows.append({
            "scale_factor": factor, "method": "UNet v1 single-model mean", **v1_mean,
            "F_seed_min": float(v1_mean_row["F_seed_min"]),
            "F_seed_max": float(v1_mean_row["F_seed_max"]),
            "n_pages": len(val_items), **gt_fields,
        })
        plot["v1_mean"].append(v1_mean["F"])
        plot["v1_min"].append(float(v1_mean_row["F_seed_min"]))
        plot["v1_max"].append(float(v1_mean_row["F_seed_max"]))

        v2_seed_means = {}
        for name, _ in V2_CHECKPOINTS:
            values = page_scores[(factor, name)]
            v2_seed_means[name] = {metric: float(np.mean([row[metric] for row in values]))
                                   for metric in METRICS}
            output_rows.append({"scale_factor": factor, "method": name, **v2_seed_means[name],
                                "F_seed_min": "", "F_seed_max": "", "n_pages": len(val_items),
                                **gt_fields})
        v2_seed_f = [v2_seed_means[name]["F"] for name, _ in V2_CHECKPOINTS]
        v2_mean = {metric: float(np.mean([v2_seed_means[name][metric] for name, _ in V2_CHECKPOINTS]))
                   for metric in METRICS}
        output_rows.append({"scale_factor": factor, "method": "UNet v2 single-model mean", **v2_mean,
                            "F_seed_min": min(v2_seed_f), "F_seed_max": max(v2_seed_f),
                            "n_pages": len(val_items), **gt_fields})
        plot["v2_mean"].append(v2_mean["F"])
        plot["v2_min"].append(min(v2_seed_f))
        plot["v2_max"].append(max(v2_seed_f))

        sauvola = v1_by_factor_method[(factor, "Sauvola (window 51, k 0.2)")]
        sauvola_metrics = {metric: float(sauvola[metric]) for metric in METRICS}
        output_rows.append({"scale_factor": factor, "method": "Sauvola (window 51, k 0.2)",
                            **sauvola_metrics, "F_seed_min": "", "F_seed_max": "",
                            "n_pages": len(val_items), **gt_fields})
        plot["sauvola"].append(sauvola_metrics["F"])

    fields = ("scale_factor", "method", "F", "P", "R", "PSNR", "F_seed_min", "F_seed_max",
              "n_pages", "gt_ink_pixels_original_total", "gt_ink_pixels_resized_total",
              "gt_pixels_lost_vs_original", "gt_area_loss_original_equiv")
    with Path("outputs/val_scale_sensitivity_v2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    fig, axis = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    axis.plot(FACTORS, plot["v1_mean"], marker="o", label="UNet v1 seed mean")
    axis.fill_between(FACTORS, plot["v1_min"], plot["v1_max"], alpha=0.16, label="UNet v1 seed min-max")
    axis.plot(FACTORS, plot["v2_mean"], marker="o", label="UNet v2 seed mean")
    axis.fill_between(FACTORS, plot["v2_min"], plot["v2_max"], alpha=0.20, label="UNet v2 seed min-max")
    axis.plot(FACTORS, plot["sauvola"], marker="s", label="Sauvola (51, 0.2)")
    axis.set_xlabel("Image scale factor")
    axis.set_ylabel("Mean validation F-measure (%)")
    axis.set_xticks(FACTORS)
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.savefig("outputs/val_scale_sensitivity_v2.png", dpi=180)
    plt.close(fig)
    print("v2 validation-only scale evaluation complete; test split was not loaded")


if __name__ == "__main__":
    main()
