"""Summarize saved test scores and render qualitative pages; runs no model inference."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.filters import threshold_sauvola


MANIFEST_PATH = Path("data/manifest.csv")
SPLITS_PATH = Path("data/splits.json")
PER_PAGE_PATH = Path("outputs/test_per_page.csv")
PREDICTIONS_DIR = Path("outputs/test_predictions")
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10_000
SAUVOLA_SYSTEM = "Sauvola (window 51, k 0.2)"
METRICS = ("F", "P", "R", "PSNR")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def page_key(item):
    return (str(item["year"]), str(item.get("track", "")), str(item["id"]))


def test_group(year, track):
    if year == "2018":
        return "2018"
    if year == "2019" and track in ("A", "B"):
        return "2019" + track
    raise ValueError(f"Unexpected test page year/track: {year}/{track}")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dataset_statistics(manifest, splits):
    by_key = {page_key(row): row for row in manifest}
    groups = {
        "train": [by_key[page_key(item)] for item in splits["train"]],
        "val": [by_key[page_key(item)] for item in splits["val"]],
        "2018": [row for row in manifest if row["year"] == "2018"],
        "2019A": [row for row in manifest if row["year"] == "2019" and row["track"] == "A"],
        "2019B": [row for row in manifest if row["year"] == "2019" and row["track"] == "B"],
    }
    results = []
    for name, pages in groups.items():
        heights, widths, ink_fracs = [], [], []
        for page in pages:
            with Image.open(page["image"]) as image:
                width, height = image.size
            gt = np.asarray(Image.open(page["gt"]).convert("L"))
            heights.append(height)
            widths.append(width)
            ink_fracs.append(float((gt < 128).mean()))
        results.append({
            "split": name,
            "n_pages": len(pages),
            "height_median": float(np.median(heights)),
            "height_min": int(min(heights)),
            "height_max": int(max(heights)),
            "width_median": float(np.median(widths)),
            "width_min": int(min(widths)),
            "width_max": int(max(widths)),
            "gt_ink_fraction_median": float(np.median(ink_fracs)),
        })
    write_csv(Path("outputs/data_stats.csv"), results)
    return results


def bootstrap_comparison(score_rows):
    per_page = defaultdict(dict)
    for row in score_rows:
        per_page[(row["year"], row["track"], row["id"])][row["system"]] = row
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = []
    for group in ("2018", "2019A", "2019B", "ALL"):
        differences = []
        for (year, track, _), systems in per_page.items():
            actual_group = test_group(year, track)
            if group != "ALL" and actual_group != group:
                continue
            differences.append(
                float(systems["UNet single-model mean"]["F"])
                - float(systems[SAUVOLA_SYSTEM]["F"])
            )
        differences = np.asarray(differences, dtype=np.float64)
        indices = rng.integers(0, len(differences), size=(BOOTSTRAP_RESAMPLES, len(differences)))
        bootstrap_means = differences[indices].mean(axis=1)
        low, high = np.percentile(bootstrap_means, [2.5, 97.5])
        results.append({
            "group": group,
            "n_pages": int(len(differences)),
            "mean_delta_F_pp": float(differences.mean()),
            "ci95_low_pp": float(low),
            "ci95_high_pp": float(high),
            "n_pages_unet_beats_sauvola": int(np.count_nonzero(differences > 0)),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        })
    write_csv(Path("outputs/test_unet_vs_sauvola.csv"), results)
    return results


def save_gallery(manifest, score_rows):
    manifest_by_key = {page_key(row): row for row in manifest}
    lookup = {(row["year"], row["track"], row["id"], row["system"]): row for row in score_rows}
    base_2018 = [row for row in score_rows if row["system"] == "unet_base" and row["year"] == "2018"]
    best_2018 = max(base_2018, key=lambda row: float(row["F"]))
    specifications = (
        ("2019B", "B", "11"),
        ("2019A", "A", "8"),
        ("2019B", "B", "15"),
        ("2018", "", best_2018["id"]),
    )
    notes = []
    for label, track, page_id in specifications:
        year = "2018" if label == "2018" else "2019"
        item = manifest_by_key[(year, track, page_id)]
        original = np.asarray(Image.open(item["image"]).convert("L"))
        gt = np.asarray(Image.open(item["gt"]).convert("L")) < 128
        pred_path = PREDICTIONS_DIR / f"unet_base_{label}_{page_id}.png"
        unet_pred = np.asarray(Image.open(pred_path).convert("L")) < 128
        # Reconstruct the already fixed Sauvola baseline for visualization only.
        sauvola_pred = original < threshold_sauvola(original, window_size=51, k=0.2)

        error = np.full((*gt.shape, 3), 255, dtype=np.uint8)
        error[unet_pred & gt] = (128, 128, 128)
        error[unet_pred & ~gt] = (255, 0, 0)
        error[~unet_pred & gt] = (0, 0, 255)

        fig, axes = plt.subplots(1, 5, figsize=(19, 4.5), constrained_layout=True)
        panels = (
            (original, "Original", "gray"),
            (gt, "Ground truth", "gray"),
            (unet_pred, "UNet base", "gray"),
            (sauvola_pred, "Sauvola (51, 0.2)", "gray"),
            (error, "FP red; FN blue; TP gray", None),
        )
        for axis, (data, title, cmap) in zip(axes, panels):
            if data.dtype == bool:
                axis.imshow(data, cmap=cmap)
            else:
                axis.imshow(data, cmap=cmap, vmin=0, vmax=255)
            axis.set_title(title, fontsize=10)
            axis.axis("off")
        fig.suptitle(f"Test page {label}/{page_id}", fontsize=13)
        fig.savefig(Path("outputs") / f"test_failure_{label}_{page_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        base = lookup[(year, track, page_id, "unet_base")]
        sauvola = lookup[(year, track, page_id, SAUVOLA_SYSTEM)]
        notes.append(
            f"{label}/{page_id}: unet_base F={float(base['F']):.2f}, P={float(base['P']):.2f}, "
            f"R={float(base['R']):.2f}; Sauvola F={float(sauvola['F']):.2f}, "
            f"P={float(sauvola['P']):.2f}, R={float(sauvola['R']):.2f}."
        )
    Path("outputs/test_failure_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return best_2018, notes


def main():
    manifest = read_csv(MANIFEST_PATH)
    with SPLITS_PATH.open(encoding="utf-8") as handle:
        splits = json.load(handle)
    score_rows = read_csv(PER_PAGE_PATH)

    stats = dataset_statistics(manifest, splits)
    comparisons = bootstrap_comparison(score_rows)
    best_2018, notes = save_gallery(manifest, score_rows)
    print(f"Best 2018 unet_base page: {best_2018['id']} (F={float(best_2018['F']):.2f})")
    print(f"Wrote data statistics for {sum(row['n_pages'] for row in stats)} split entries.")
    print(f"Wrote paired bootstrap comparisons ({BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}).")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
