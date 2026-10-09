# src/baselines.py  -- Otsu and Sauvola on the frozen split
import json, itertools
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu, threshold_sauvola
from metrics import score

def load(item):
    img = np.array(Image.open(item["image"]).convert("L"))
    gt = np.array(Image.open(item["gt"]).convert("L")) < 128      # True = ink
    return img, gt

def otsu(img):
    return img < threshold_otsu(img)

def sauvola(img, window=51, k=0.2):
    return img < threshold_sauvola(img, window_size=window, k=k)

def evaluate(items, fn, **kw):
    rows = []
    for it in items:
        img, gt = load(it)
        s = score(fn(img, **kw), gt)
        rows.append({**{k: it[k] for k in ("year", "track", "id")}, **s})
    return rows

def mean(rows, key): return float(np.mean([r[key] for r in rows]))

if __name__ == "__main__":
    sp = json.load(open("data/splits.json"))
    # Sauvola parameters are tuned on TRAIN only; validation is then scored once.
    best, best_f = None, -1
    for w, k in itertools.product([25, 51, 75], [0.1, 0.2, 0.3]):
        f = mean(evaluate(sp["train"], sauvola, window=w, k=k), "F")
        print(f"train  Sauvola w={w} k={k}: F={f:.2f}")
        if f > best_f: best, best_f = (w, k), f
    print("\nBest Sauvola on train:", best)
    print(f"\n{'method':10s} {'F':>6s} {'P':>6s} {'R':>6s} {'PSNR':>6s}   (validation = 2017)")
    for name, fn, kw in [("Otsu", otsu, {}), ("Sauvola", sauvola, dict(window=best[0], k=best[1]))]:
        rows = evaluate(sp["val"], fn, **kw)
        print(f"{name:10s} {mean(rows,'F'):6.2f} {mean(rows,'P'):6.2f} {mean(rows,'R'):6.2f} {mean(rows,'PSNR'):6.2f}")
