# src/build_manifest.py
# Pairs every DIBCO page image with its ground truth, writes data/manifest.csv,
# and creates data/splits.json ONCE (it is never overwritten, so the split stays frozen).
import re, csv, json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("data/raw")
EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}
SKIP_DIRS = {"binevalweights", "dibco_metrics"}  # evaluation support files, not pages
GT_RE = re.compile(r"[_\-]?(est)?gt$", re.I)   # H01_estGT, HW1_GT, 1_gt -> id
SKEL_RE = re.compile(r"skel", re.I)            # 2010 skeleton GT: not used

TRAIN = {"2009", "2010", "2011", "2012", "2013", "2014", "2016"}
VAL = {"2017"}
TEST = {"2018", "2019"}


def track_of(rel_path):
    s = str(rel_path).lower().replace("_", "").replace(" ", "").replace("-", "")
    return "A" if "tracka" in s else "B" if "trackb" in s else ""


def is_gt_file(f, year_dir):
    """Recognize GT by its filename suffix or its containing directory (2009/2019)."""
    if GT_RE.search(f.stem):
        return True
    rel_parts = [part.lower() for part in f.relative_to(year_dir).parts[:-1]]
    return any("gt" in part or "groundtruth" in part for part in rel_parts)


rows, problems = [], []
for year_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    imgs, gts = {}, {}
    for f in sorted(year_dir.rglob("*")):
        if f.suffix.lower() not in EXTS:
            continue
        if any(part.lower() in SKIP_DIRS for part in f.relative_to(year_dir).parts) or SKEL_RE.search(f.stem):
            continue
        tr = track_of(f.relative_to(year_dir))
        if is_gt_file(f, year_dir):
            gts[(tr, GT_RE.sub("", f.stem).lower())] = f
        else:
            imgs[(tr, f.stem.lower())] = f

    for k, ip in imgs.items():
        gp = gts.get(k)
        if gp is None:
            problems.append((year_dir.name, "image has no GT", ip.as_posix()))
            continue
        with Image.open(ip) as a, Image.open(gp) as b:
            size_ok = a.size == b.size
            gt_dark = float((np.array(b.convert("L")) < 128).mean())
        rows.append(dict(year=year_dir.name, track=k[0], id=k[1], image=ip.as_posix(),
                         gt=gp.as_posix(), size_match=size_ok, gt_dark=round(gt_dark, 4)))
        if not size_ok:
            problems.append((year_dir.name, "image/GT size mismatch", ip.as_posix()))
        if gt_dark > 0.5:
            problems.append((year_dir.name, "GT looks inverted (dark > 0.5)", gp.as_posix()))
    for k, gp in gts.items():
        if k not in imgs:
            problems.append((year_dir.name, "GT has no image", gp.as_posix()))

if rows:
    with open("data/manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
else:
    raise RuntimeError(f"No paired pages found under {ROOT}")

split_file = Path("data/splits.json")
if split_file.exists():
    print("splits.json already exists: left untouched (frozen).")
else:
    splits = {"train": [], "val": [], "test": []}
    for r in rows:
        name = "train" if r["year"] in TRAIN else "val" if r["year"] in VAL else \
               "test" if r["year"] in TEST else None
        if name:
            splits[name].append({k: r[k] for k in ("year", "track", "id", "image", "gt")})
    split_file.write_text(json.dumps(splits, indent=2))
    print("splits.json created.")

print("\nPaired pages per year:")
for y in sorted({r["year"] for r in rows}):
    print(f"  {y}: {sum(r['year'] == y for r in rows)}")
print("\nProblems:" if problems else "\nNo problems found.")
for p in problems:
    print("  ", *p)
