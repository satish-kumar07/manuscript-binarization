"""One-time, fixed-protocol evaluation on the frozen DIBCO test split."""
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.filters import threshold_otsu, threshold_sauvola

from dataset import load_gt, load_image, load_split
from ensemble_infer import predict_tta
from infer import load_model, predict_page
from metrics import score


SPLITS_PATH = Path("data/splits.json")
CHECKPOINTS = (
    ("unet_base", Path("outputs/checkpoints/unet_base_best.pt")),
    ("unet_seed1", Path("outputs/checkpoints/unet_seed1_best.pt")),
    ("unet_seed2", Path("outputs/checkpoints/unet_seed2_best.pt")),
)
PER_PAGE_PATH = Path("outputs/test_per_page.csv")
SUMMARY_PATH = Path("outputs/test_summary.csv")
PREDICTION_DIR = Path("outputs/test_predictions")
METRIC_KEYS = ("F", "P", "R", "PSNR")
GROUP_ORDER = ("2018", "2019A", "2019B", "ALL")
SYSTEM_ORDER = (
    "Otsu",
    "Sauvola (window 51, k 0.2)",
    "unet_base",
    "unet_seed1",
    "unet_seed2",
    "UNet single-model mean",
    "Ensemble + TTA (exploratory)",
)


def group_name(item):
    if item["year"] == "2018":
        return "2018"
    if item["year"] == "2019" and item["track"] in ("A", "B"):
        return "2019" + item["track"]
    raise ValueError(f"Unexpected test page year/track: {item['year']}/{item['track']}")


def page_row(system, item, values):
    return {
        "system": system,
        "group": group_name(item),
        "year": item["year"],
        "track": item["track"],
        "id": item["id"],
        **values,
    }


def binary_save(path, pred):
    # Match infer.py: foreground ink is black; background is white.
    Image.fromarray((~pred.astype(bool) * 255).astype(np.uint8)).save(path)


def summarize(page_rows):
    summaries = []
    base_systems = ("Otsu", "Sauvola (window 51, k 0.2)", *[n for n, _ in CHECKPOINTS])

    for group in GROUP_ORDER:
        group_pages = [r for r in page_rows if group == "ALL" or r["group"] == group]
        if not group_pages:
            continue
        n_pages = len({(r["year"], r["track"], r["id"]) for r in group_pages if r["system"] == "Otsu"})

        def add_summary(system, selected, f_min="", f_max=""):
            row = {"system": system, "group": group, "n_pages": n_pages}
            for metric in METRIC_KEYS:
                row[metric] = float(np.mean([r[metric] for r in selected]))
            row["F_seed_min"] = f_min
            row["F_seed_max"] = f_max
            summaries.append(row)

        for system in base_systems:
            selected = [r for r in group_pages if r["system"] == system]
            add_summary(system, selected)

        # Mean each metric over seeds per page, then over pages. Also retain the
        # range of the three seed-level group means for F-measure.
        seed_rows = {name: [r for r in group_pages if r["system"] == name] for name, _ in CHECKPOINTS}
        avg_rows = []
        for idx in range(len(seed_rows["unet_base"])):
            avg_rows.append({metric: float(np.mean([seed_rows[name][idx][metric] for name, _ in CHECKPOINTS]))
                             for metric in METRIC_KEYS})
        seed_means_f = [float(np.mean([r["F"] for r in seed_rows[name]])) for name, _ in CHECKPOINTS]
        add_summary("UNet single-model mean", avg_rows, min(seed_means_f), max(seed_means_f))

        selected = [r for r in group_pages if r["system"] == "Ensemble + TTA (exploratory)"]
        add_summary("Ensemble + TTA (exploratory)", selected)

    return summaries


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not SPLITS_PATH.is_file():
        raise FileNotFoundError(SPLITS_PATH)
    for _, checkpoint in CHECKPOINTS:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    items = load_split(str(SPLITS_PATH), "test")
    if len(items) != 30:
        raise RuntimeError(f"Expected 30 frozen test pages, found {len(items)}")
    observed = {group_name(item) for item in items}
    if observed != {"2018", "2019A", "2019B"}:
        raise RuntimeError(f"Unexpected test groups: {sorted(observed)}")
    for item in items:
        for key in ("image", "gt"):
            if not Path(item[key]).is_file():
                raise FileNotFoundError(item[key])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for name, checkpoint in CHECKPOINTS:
        model, in_ch, patch = load_model(str(checkpoint), device)
        models[name] = {"model": model, "in_ch": in_ch, "patch": patch}

    page_rows = []
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    for page_index, item in enumerate(items, start=1):
        image_gray = np.asarray(Image.open(item["image"]).convert("L"))
        gt = load_gt(item["gt"]) > 0.5
        page_suffix = f"{item['year']}{item['track']}_{item['id']}"

        page_rows.append(page_row("Otsu", item,
                                  score(image_gray < threshold_otsu(image_gray), gt)))
        sauvola_threshold = threshold_sauvola(image_gray, window_size=51, k=0.2)
        page_rows.append(page_row("Sauvola (window 51, k 0.2)", item,
                                  score(image_gray < sauvola_threshold, gt)))

        tta_sum = None
        seed_page_scores = []
        for name, _ in CHECKPOINTS:
            entry = models[name]
            image = load_image(item["image"], entry["in_ch"])
            prob = predict_page(entry["model"], image, patch=entry["patch"],
                                overlap=entry["patch"] // 4, device=device)
            pred = prob >= 0.5
            seed_score = score(pred, gt)
            seed_page_scores.append(seed_score)
            page_rows.append(page_row(name, item, seed_score))
            if name == "unet_base":
                binary_save(PREDICTION_DIR / f"unet_base_{page_suffix}.png", pred)

            tta_prob = predict_tta(entry["model"], image, entry["patch"], device)
            if tta_sum is None:
                tta_sum = tta_prob.astype(np.float64)
            else:
                tta_sum += tta_prob

        page_rows.append(page_row(
            "UNet single-model mean",
            item,
            {metric: float(np.mean([scores[metric] for scores in seed_page_scores]))
             for metric in METRIC_KEYS},
        ))
        ensemble_tta_pred = (tta_sum / len(CHECKPOINTS)) >= 0.5
        page_rows.append(page_row("Ensemble + TTA (exploratory)", item,
                                  score(ensemble_tta_pred, gt)))
        binary_save(PREDICTION_DIR / f"ensemble_tta_{page_suffix}.png", ensemble_tta_pred)
        print(f"processed test page {page_index}/{len(items)}: {page_suffix}", flush=True)

    summaries = summarize(page_rows)
    write_csv(PER_PAGE_PATH, page_rows,
              ("system", "group", "year", "track", "id", "F", "P", "R", "PSNR"))
    write_csv(SUMMARY_PATH, summaries,
              ("system", "group", "n_pages", "F", "P", "R", "PSNR", "F_seed_min", "F_seed_max"))

    print(f"device: {device}; evaluated only frozen test split ({len(items)} pages)")
    print(f"{'system':35s} {'group':7s} {'n':>3s} {'F':>7s} {'PSNR':>7s} {'F seed range':>17s}")
    for row in summaries:
        f_range = (f"{row['F_seed_min']:.2f}–{row['F_seed_max']:.2f}"
                   if row["F_seed_min"] != "" else "")
        print(f"{row['system']:35s} {row['group']:7s} {row['n_pages']:3d} "
              f"{row['F']:7.2f} {row['PSNR']:7.2f} {f_range:>17s}")
    print(f"saved {PER_PAGE_PATH}, {SUMMARY_PATH}, and predictions under {PREDICTION_DIR}")


if __name__ == "__main__":
    main()
