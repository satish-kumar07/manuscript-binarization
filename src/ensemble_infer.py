"""Validation-only probability ensemble for the three RGB seed checkpoints."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from dataset import load_gt, load_image, load_split
from infer import load_model, predict_page
from metrics import score


CHECKPOINTS = (
    ("unet_base", "outputs/checkpoints/unet_base_best.pt"),
    ("unet_seed1", "outputs/checkpoints/unet_seed1_best.pt"),
    ("unet_seed2", "outputs/checkpoints/unet_seed2_best.pt"),
)


def predict_tta(model, image, patch, device):
    """Average probabilities across original, horizontal, vertical, and both flips."""
    variants = (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    )
    probs = []
    for flip_y, flip_x in variants:
        view = image
        if flip_y:
            view = view[:, ::-1, :]
        if flip_x:
            view = view[:, :, ::-1]
        pred = predict_page(
            model,
            np.ascontiguousarray(view),
            patch=patch,
            overlap=patch // 4,
            device=device,
        )
        if flip_x:
            pred = pred[:, ::-1]
        if flip_y:
            pred = pred[::-1, :]
        probs.append(pred)
    return np.mean(probs, axis=0)


def mean_scores(prob_maps, ground_truths):
    rows = [score(prob > 0.5, gt) for prob, gt in zip(prob_maps, ground_truths)]
    return {key: float(np.mean([row[key] for row in rows])) for key in ("F", "P", "R", "PSNR")}


def page_f(prob_maps, ground_truths, items, page_id):
    matches = [i for i, item in enumerate(items) if item["year"] == "2017" and item["id"] == page_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one 2017/{page_id} validation page, found {len(matches)}")
    i = matches[0]
    return score(prob_maps[i] > 0.5, ground_truths[i])["F"]


def main():
    ap = argparse.ArgumentParser(description="Score the three-seed ensemble on validation only.")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--out", default="outputs/ensemble_val.csv")
    ap.add_argument("--tta", action="store_true", help="also evaluate four-flip test-time augmentation")
    args = ap.parse_args()

    items = load_split(args.splits, "val")
    if not items:
        raise RuntimeError("The validation split is empty")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ground_truths = [load_gt(item["gt"]) > 0.5 for item in items]
    predictions = []
    tta_predictions = []

    for name, checkpoint in CHECKPOINTS:
        model, in_ch, patch = load_model(checkpoint, device)
        basic, tta = [], []
        for item in items:
            image = load_image(item["image"], in_ch)
            basic.append(predict_page(model, image, patch=patch, overlap=patch // 4, device=device))
            if args.tta:
                tta.append(predict_tta(model, image, patch, device))
        predictions.append((name, basic))
        if args.tta:
            tta_predictions.append((name, tta))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    ensemble = [np.mean([seed_maps[i] for _, seed_maps in predictions], axis=0)
                for i in range(len(items))]
    rows = []

    # Mean of the three individually thresholded model scores.
    individual_scores = [mean_scores(maps, ground_truths) for _, maps in predictions]
    individual_page13 = [page_f(maps, ground_truths, items, "13") for _, maps in predictions]
    individual_page17 = [page_f(maps, ground_truths, items, "17") for _, maps in predictions]
    rows.append({
        "system": "single_model_mean",
        **{key: float(np.mean([s[key] for s in individual_scores])) for key in ("F", "P", "R", "PSNR")},
        "page_2017_13_F": float(np.mean(individual_page13)),
        "page_2017_17_F": float(np.mean(individual_page17)),
        "n_pages": len(items),
    })

    ens_scores = mean_scores(ensemble, ground_truths)
    rows.append({
        "system": "ensemble",
        **ens_scores,
        "page_2017_13_F": page_f(ensemble, ground_truths, items, "13"),
        "page_2017_17_F": page_f(ensemble, ground_truths, items, "17"),
        "n_pages": len(items),
    })

    if args.tta:
        tta_mean_scores = [mean_scores(maps, ground_truths) for _, maps in tta_predictions]
        tta_page13 = [page_f(maps, ground_truths, items, "13") for _, maps in tta_predictions]
        tta_page17 = [page_f(maps, ground_truths, items, "17") for _, maps in tta_predictions]
        rows.append({
            "system": "single_model_tta_mean",
            **{key: float(np.mean([s[key] for s in tta_mean_scores])) for key in ("F", "P", "R", "PSNR")},
            "page_2017_13_F": float(np.mean(tta_page13)),
            "page_2017_17_F": float(np.mean(tta_page17)),
            "n_pages": len(items),
        })
        ensemble_tta = [np.mean([seed_maps[i] for _, seed_maps in tta_predictions], axis=0)
                        for i in range(len(items))]
        tta_ens_scores = mean_scores(ensemble_tta, ground_truths)
        rows.append({
            "system": "ensemble_tta",
            **tta_ens_scores,
            "page_2017_13_F": page_f(ensemble_tta, ground_truths, items, "13"),
            "page_2017_17_F": page_f(ensemble_tta, ground_truths, items, "17"),
            "n_pages": len(items),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ("system", "F", "P", "R", "PSNR", "page_2017_13_F", "page_2017_17_F", "n_pages")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"device: {device}; split: val ({len(items)} pages); TTA: {args.tta}")
    print(f"{'system':24s} {'F':>6s} {'P':>6s} {'R':>6s} {'PSNR':>7s} {'page13':>8s} {'page17':>8s}")
    for row in rows:
        print(f"{row['system']:24s} {row['F']:6.2f} {row['P']:6.2f} {row['R']:6.2f} "
              f"{row['PSNR']:7.2f} {row['page_2017_13_F']:8.2f} {row['page_2017_17_F']:8.2f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
