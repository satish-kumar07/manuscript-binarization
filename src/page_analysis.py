"""Validation-only per-page checks for the DIBCO 2017 difficult pages."""
from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image, ImageDraw

from dataset import load_gt, load_image
from infer import load_model, predict_page
from metrics import score


CHECKPOINTS = ("unet_base", "unet_seed1", "unet_seed2")


def main():
    val = json.loads(Path("data/splits.json").read_text(encoding="utf-8"))["val"]
    pages = {item["id"]: item for item in val if item["year"] == "2017"}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    item17 = pages["17"]
    image17 = load_image(item17["image"], 3)
    gt17 = load_gt(item17["gt"]) > 0.5
    print("Page 17 seed check:")
    for name in CHECKPOINTS:
        model, in_ch, patch = load_model(f"outputs/checkpoints/{name}_best.pt", device)
        prob = predict_page(model, image17, patch=patch, overlap=patch // 4, device=device)
        result = score(prob > 0.5, gt17)
        print(f"  {name} page 17 ({item17['year']}/{item17['id']}): "
              f"F={result['F']:.2f} P={result['P']:.2f} R={result['R']:.2f} PSNR={result['PSNR']:.2f}")
        del model

    item13 = pages["13"]
    image13 = load_image(item13["image"], 3)
    gt13 = load_gt(item13["gt"]) > 0.5
    base_pred = None
    print("Page 13 seed check:")
    for name in CHECKPOINTS:
        model, in_ch, patch = load_model(f"outputs/checkpoints/{name}_best.pt", device)
        prob = predict_page(model, image13, patch=patch, overlap=patch // 4, device=device)
        pred = prob > 0.5
        result = score(pred, gt13)
        print(f"  {name} page 13 ({item13['year']}/{item13['id']}): "
              f"F={result['F']:.2f} P={result['P']:.2f} R={result['R']:.2f} PSNR={result['PSNR']:.2f}")
        if name == "unet_base":
            base_pred = pred
        del model

    fp = base_pred & ~gt13
    fn = ~base_pred & gt13
    tp = base_pred & gt13
    errors = np.full((*gt13.shape, 3), 255, dtype=np.uint8)
    errors[tp] = (190, 190, 190)
    errors[fp] = (255, 45, 45)
    errors[fn] = (20, 170, 255)
    original = np.asarray(Image.open(item13["image"]).convert("RGB"))
    gt_panel = np.repeat((~gt13 * 255).astype(np.uint8)[..., None], 3, axis=2)
    pred_panel = np.repeat((~base_pred * 255).astype(np.uint8)[..., None], 3, axis=2)
    panels = (
        ("Original", original),
        ("Ground truth", gt_panel),
        ("Prediction", pred_panel),
        ("Errors: FP red, FN blue, TP gray", errors),
    )
    width, height = original.shape[1], original.shape[0]
    label_height = 34
    canvas = Image.new("RGB", (width * len(panels), height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, pixels) in enumerate(panels):
        draw.text((i * width + 8, 8), label, fill="black")
        canvas.paste(Image.fromarray(pixels), (i * width, label_height))
    out = Path("outputs/page13_errors.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
