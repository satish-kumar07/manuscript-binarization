# src/dataset.py
import json, random
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def load_image(path, in_ch=3):
    """-> float32 array, shape (C, H, W), values in [0, 1]."""
    im = Image.open(path).convert("RGB" if in_ch == 3 else "L")
    a = np.asarray(im, dtype=np.float32) / 255.0
    return a.transpose(2, 0, 1) if in_ch == 3 else a[None]

def load_gt(path):
    """-> float32 array (H, W); 1 = ink, 0 = background."""
    return (np.asarray(Image.open(path).convert("L")) < 128).astype(np.float32)

def load_split(splits_path, name):
    return json.load(open(splits_path))[name]

class PatchDataset:
    """Random patches from whole pages. Pages are cached in RAM."""
    def __init__(self, items, patch=256, n_per_epoch=1600, train=True, in_ch=3,
                 bleed=0.0, lowcon=0.0):
        self.imgs = [load_image(it["image"], in_ch) for it in items]
        self.gts = [load_gt(it["gt"]) for it in items]
        self.patch, self.n, self.train, self.in_ch = patch, n_per_epoch, train, in_ch
        self.bleed, self.lowcon = bleed, lowcon      # probabilities of the two extra augmentations

    def __len__(self):
        return self.n

    def _crop(self):
        i = random.randrange(len(self.imgs))
        img, gt = self.imgs[i], self.gts[i]
        p = self.patch
        _, h, w = img.shape
        if h < p or w < p:                                  # pad small pages
            ph, pw = max(0, p - h), max(0, p - w)
            img = np.pad(img, ((0, 0), (0, ph), (0, pw)), mode="edge")
            gt = np.pad(gt, ((0, ph), (0, pw)), mode="edge")
            h, w = gt.shape
        y, x = random.randint(0, h - p), random.randint(0, w - p)
        return img[:, y:y+p, x:x+p], gt[y:y+p, x:x+p]

    def __getitem__(self, _):
        for _try in range(5):                               # prefer patches that contain ink
            im, mk = self._crop()
            if mk.mean() > 0.005:
                break
        if self.train:
            if random.random() < self.bleed:                # fake reverse-side show-through
                _, other = self._crop()
                ink = gaussian_filter(other[:, ::-1], sigma=random.uniform(1.5, 3.5))
                tint = np.array([0.85, 0.92, 1.0], np.float32)[:, None, None] if im.shape[0] == 3 else 1.0
                im = im * (1 - random.uniform(0.15, 0.45) * np.clip(ink * 3, 0, 1) * tint)
            if random.random() < self.lowcon:               # faint, low-contrast print
                m = im.mean((1, 2), keepdims=True)
                im = (im - m) * random.uniform(0.4, 0.8) + m
            if random.random() < .5: im, mk = im[:, :, ::-1], mk[:, ::-1]
            if random.random() < .5: im, mk = im[:, ::-1, :], mk[::-1, :]
            k = random.randrange(4)
            im, mk = np.rot90(im, k, (1, 2)), np.rot90(mk, k, (0, 1))
            gain = np.random.uniform(0.9, 1.1, (im.shape[0], 1, 1)).astype(np.float32)
            im = im * gain * random.uniform(0.7, 1.3) + random.uniform(-0.15, 0.15)
            im = im + np.random.normal(0, 0.02, im.shape).astype(np.float32)
            im = np.clip(im, 0, 1)
        return np.ascontiguousarray(im, dtype=np.float32), np.ascontiguousarray(mk[None], dtype=np.float32)