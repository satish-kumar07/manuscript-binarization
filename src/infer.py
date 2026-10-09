# src/infer.py  -- sliding-window prediction with overlap, plus a scoring CLI
import argparse, math
from pathlib import Path
import numpy as np, torch
from PIL import Image
from dataset import load_image, load_gt, load_split
from model import UNet
from metrics import score

@torch.no_grad()
def predict_page(model, img, patch=256, overlap=64, batch=8, device="cpu"):
    """img: (C, H, W) float in [0,1] -> probability map (H, W)."""
    model.eval()
    _, h, w = img.shape
    s = patch - overlap
    H = max(0, math.ceil((h - patch) / s)) * s + patch
    W = max(0, math.ceil((w - patch) / s)) * s + patch
    x = np.pad(img, ((0, 0), (0, H - h), (0, W - w)), mode="edge")
    acc, cnt = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    coords = [(y, xx) for y in range(0, H - patch + 1, s) for xx in range(0, W - patch + 1, s)]
    for i in range(0, len(coords), batch):
        cs = coords[i:i + batch]
        t = torch.from_numpy(np.stack([x[:, y:y+patch, xx:xx+patch] for y, xx in cs])).to(device)
        p = torch.sigmoid(model(t))[:, 0].cpu().numpy()
        for (y, xx), pp in zip(cs, p):
            acc[y:y+patch, xx:xx+patch] += pp; cnt[y:y+patch, xx:xx+patch] += 1
    return (acc / cnt)[:h, :w]

def load_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device)
    m = UNet(ck["in_ch"], ck["base"]).to(device)
    m.load_state_dict(ck["state"]); return m, ck["in_ch"], ck.get("patch", 256)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--out", default="outputs/predictions")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, in_ch, patch = load_model(a.ckpt, dev)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    groups = {}
    for it in load_split(a.splits, a.split):
        pred = predict_page(model, load_image(it["image"], in_ch), patch=patch,
                            overlap=patch // 4, device=dev) > 0.5
        s = score(pred, load_gt(it["gt"]) > 0.5)
        groups.setdefault((it["year"], it["track"]), []).append(s)
        Image.fromarray((~pred * 255).astype(np.uint8)).save(f"{a.out}/{it['year']}{it['track']}_{it['id']}.png")
    print(f"{'group':10s} {'n':>3s} {'F':>6s} {'P':>6s} {'R':>6s} {'PSNR':>6s}")
    for (y, t), ss in sorted(groups.items()):
        print(f"{y+t:10s} {len(ss):3d} " + " ".join(f"{np.mean([x[k] for x in ss]):6.2f}" for k in ("F", "P", "R", "PSNR")))
