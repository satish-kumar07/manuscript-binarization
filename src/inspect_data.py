from pathlib import Path
from PIL import Image
import numpy as np

root = Path("data/raw")
exts = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}

for year in sorted(p for p in root.iterdir() if p.is_dir()):
    files = sorted(f for f in year.rglob("*") if f.suffix.lower() in exts)
    print(f"\n=== {year.name}: {len(files)} image files ===")
    for f in files:
        im = Image.open(f)
        a = np.array(im.convert("L"))
        dark = (a < 128).mean()
        print(f"{str(f.relative_to(year)):55s} {im.size} mode={im.mode} dark={dark:.3f}")
