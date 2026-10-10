import csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from dataset import load_gt, load_image, load_split
from infer import load_model, predict_page
from metrics import score

SPLITS = Path("data/splits.json")
CHECKPOINTS = [(f"unet_v2_s{i}", Path(f"outputs/checkpoints/unet_v2_s{i}_best.pt")) for i in range(3)]
OUT_PAGES = Path("outputs/test_v2_per_page.csv")
OUT_SUMMARY = Path("outputs/test_v2_summary.csv")
PRED_DIR = Path("outputs/test_v2_predictions")
METRICS = ("F", "P", "R", "PSNR")
GROUPS = ("2018", "2019A", "2019B", "ALL")
SAVE_PAGES = {("2019", "A", "8"), ("2019", "B", "11"), ("2019", "B", "15"), ("2018", "", "6")}

def group(item):
    return item["year"] if item["year"] == "2018" else "2019" + item["track"]

def write(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    items = load_split(str(SPLITS), "test")
    if len(items) != 30:
        raise RuntimeError(f"Expected 30 frozen test pages, found {len(items)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {name: load_model(str(path), device) for name, path in CHECKPOINTS}
    page_rows = []
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(items, 1):
        image_gray = np.asarray(Image.open(item["image"]).convert("L"))
        gt = load_gt(item["gt"]) > 0.5
        seed_scores = []
        for name, _ in CHECKPOINTS:
            model, in_ch, patch = models[name]
            image = load_image(item["image"], in_ch)
            prob = predict_page(model, image, patch=patch, overlap=patch // 4, device=device)
            pred = prob >= 0.5
            values = score(pred, gt); seed_scores.append(values)
            page_rows.append({"system": name, "group": group(item), "year": item["year"], "track": item["track"], "id": item["id"], **values})
            if (item["year"], item["track"], item["id"]) in SAVE_PAGES:
                out = PRED_DIR / f"{name}_{item['year']}{item['track']}_{item['id']}.png"
                Image.fromarray((~pred * 255).astype(np.uint8)).save(out)
        avg = {m: float(np.mean([s[m] for s in seed_scores])) for m in METRICS}
        page_rows.append({"system": "v2 seed-mean", "group": group(item), "year": item["year"], "track": item["track"], "id": item["id"], **avg})
        print(f"processed test page {index}/30: {item['year']}{item['track']}_{item['id']}", flush=True)
    summary = []
    for g in GROUPS:
        selected = [r for r in page_rows if g == "ALL" or r["group"] == g]
        n = len({(r["year"], r["track"], r["id"]) for r in selected if r["system"] == "v2 seed-mean"})
        for system in [n for n, _ in CHECKPOINTS] + ["v2 seed-mean"]:
            rows = [r for r in selected if r["system"] == system]
            summary.append({"system": system, "group": g, "n_pages": n, **{m: float(np.mean([r[m] for r in rows])) for m in METRICS}})
    write(OUT_PAGES, page_rows, ("system", "group", "year", "track", "id", *METRICS))
    write(OUT_SUMMARY, summary, ("system", "group", "n_pages", *METRICS))
    print(f"device: {device}; evaluated frozen test split (30 pages); threshold=0.5; TTA=off")
    print(f"saved {OUT_PAGES}, {OUT_SUMMARY}, and selected predictions under {PRED_DIR}")

if __name__ == "__main__": main()
