# src/train.py
import argparse, csv, random, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from dataset import PatchDataset, load_image, load_gt, load_split
from model import UNet
from losses import BCEDiceLoss
from infer import predict_page
from metrics import score

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="run1")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--patches_per_epoch", type=int, default=1600)
ap.add_argument("--patch", type=int, default=256)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--base", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--in_ch", type=int, default=3, choices=[1, 3])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--loss", default="bcedice", choices=["bcedice", "bce"])
ap.add_argument("--bleed", type=float, default=0.0)
ap.add_argument("--lowcon", type=float, default=0.0)
ap.add_argument("--scale_aug", action="store_true",
                help="enable random log-uniform crop-scale augmentation")
ap.add_argument("--val_pages", type=int, default=0, help="0 = all validation pages")
ap.add_argument("--splits", default="data/splits.json")
a = ap.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

train_items = load_split(a.splits, "train")
val_items = load_split(a.splits, "val")
if a.val_pages: val_items = val_items[:a.val_pages]
loader = DataLoader(PatchDataset(train_items, a.patch, a.patches_per_epoch, True, a.in_ch,
                                 a.bleed, a.lowcon, scale_aug=a.scale_aug),
                    batch_size=a.batch, num_workers=0)
val = [(load_image(i["image"], a.in_ch), load_gt(i["gt"]) > 0.5) for i in val_items]

model = UNet(a.in_ch, a.base).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
loss_fn = BCEDiceLoss() if a.loss == "bcedice" else BCEDiceLoss(w_bce=1.0, w_dice=0.0)
Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)
log = open(f"outputs/{a.name}_log.csv", "w", newline=""); w = csv.writer(log)
w.writerow(["epoch", "train_loss", "val_F", "val_PSNR", "seconds"])
best = -1
for ep in range(1, a.epochs + 1):
    t0 = time.time(); model.train(); losses = []
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
        losses.append(loss.item())
    sched.step()
    sc = [score(predict_page(model, im, a.patch, a.patch // 4, device=dev) > 0.5, gt) for im, gt in val]
    f, ps = np.mean([s["F"] for s in sc]), np.mean([s["PSNR"] for s in sc])
    print(f"epoch {ep:3d}  loss {np.mean(losses):.4f}  val F {f:.2f}  PSNR {ps:.2f}  ({time.time()-t0:.0f}s)")
    w.writerow([ep, round(np.mean(losses), 4), round(f, 2), round(ps, 2), round(time.time() - t0)]); log.flush()
    if f > best:
        best = f
        torch.save({"state": model.state_dict(), "in_ch": a.in_ch, "base": a.base,
                    "patch": a.patch, "epoch": ep, "val_F": float(f)},
                   f"outputs/checkpoints/{a.name}_best.pt")
print(f"best val F: {best:.2f}")
