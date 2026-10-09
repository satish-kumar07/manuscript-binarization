# src/metrics.py  -- binary masks: True/1 = text (ink), False/0 = background
import numpy as np

def f_measure(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return 100 * f, 100 * p, 100 * r          # percent, as in DIBCO tables

def psnr(pred, gt):
    mse = np.mean((pred.astype(float) - gt.astype(float)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)

def score(pred, gt):
    f, p, r = f_measure(pred, gt)
    return {"F": f, "P": p, "R": r, "PSNR": psnr(pred, gt)}
