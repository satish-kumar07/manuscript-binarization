"""Save augmented training patches for visual inspection before v2 training."""
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import PatchDataset, load_split


def main():
    random.seed(17)
    np.random.seed(17)
    items = load_split("data/splits.json", "train")
    dataset = PatchDataset(items, patch=256, n_per_epoch=8, train=True, in_ch=3, scale_aug=True)

    fig, axes = plt.subplots(8, 2, figsize=(8, 20), constrained_layout=True)
    for index in range(8):
        image, mask = dataset[index]
        axes[index, 0].imshow(image.transpose(1, 2, 0))
        axes[index, 0].set_title(f"Scale-augmented patch {index + 1}")
        axes[index, 1].imshow(mask[0], cmap="gray", vmin=0, vmax=1)
        axes[index, 1].set_title("Resized GT mask")
        axes[index, 0].axis("off")
        axes[index, 1].axis("off")

    output = Path("outputs/scale_aug_preview.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(f"saved {output} ({len(dataset)} preview patches sampled from train only)")


if __name__ == "__main__":
    main()
