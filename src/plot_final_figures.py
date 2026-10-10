import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

def read_csv(name):
    with (OUT / name).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def select(rows, **conditions):
    return next(r for r in rows if all(r[k] == v for k, v in conditions.items()))

def scale_figure():
    v1 = read_csv("val_scale_sensitivity.csv")
    v2 = read_csv("val_scale_sensitivity_v2.csv")
    factors = sorted({float(r["scale_factor"]) for r in v1} & {float(r["scale_factor"]) for r in v2})
    def values(rows, method):
        selected = [select(rows, scale_factor=str(x), method=method) for x in factors]
        return (np.array([float(r["F"]) for r in selected]),
                np.array([float(r["F_seed_min"]) for r in selected]),
                np.array([float(r["F_seed_max"]) for r in selected]))
    v1_mean, v1_min, v1_max = values(v1, "UNet single-model mean")
    v2_mean, v2_min, v2_max = values(v2, "UNet v2 single-model mean")
    sauvola = np.array([float(select(v1, scale_factor=str(x), method="Sauvola (window 51, k 0.2)")["F"]) for x in factors])
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    ax.plot(factors, v1_mean, marker="o", linewidth=2, label="v1 seed mean")
    ax.fill_between(factors, v1_min, v1_max, alpha=.18)
    ax.plot(factors, v2_mean, marker="o", linewidth=2, label="v2 seed mean")
    ax.fill_between(factors, v2_min, v2_max, alpha=.18)
    ax.plot(factors, sauvola, marker="o", linewidth=2, label="Sauvola")
    ax.set_xscale("log", base=2)
    ax.set_xticks(factors, [f"{x:g}" for x in factors])
    ax.set_xlabel("Rescale factor")
    ax.set_ylabel("Validation F (%)")
    ax.set_title("Validation F-measure by image scale")
    ax.grid(True, which="major", alpha=.25)
    ax.legend()
    fig.savefig(OUT / "fig_scale_sensitivity.png", dpi=180)
    plt.close(fig)

def test_figure():
    summary = read_csv("test_summary.csv")
    v2_summary = read_csv("test_v2_summary.csv")
    v1_pages = read_csv("test_per_page.csv")
    v2_pages = read_csv("test_v2_per_page.csv")
    groups = ["2018", "2019A", "2019B", "ALL"]
    specs = [
        ("Otsu", summary, v1_pages, "Otsu", "Otsu"),
        ("Sauvola", summary, v1_pages, "Sauvola (window 51, k 0.2)", "Sauvola (window 51, k 0.2)"),
        ("v1 seed mean", summary, v1_pages, "UNet single-model mean", "UNet single-model mean"),
        ("v2 seed mean (post-hoc)", v2_summary, v2_pages, "v2 seed-mean", "v2 seed-mean"),
    ]
    fig, ax = plt.subplots(figsize=(11.2, 5.5), constrained_layout=True)
    x = np.arange(len(groups)); width = .19
    for j, (label, sums, pages, system, page_system) in enumerate(specs):
        means=[]; lows=[]; highs=[]
        for g in groups:
            sr=select(sums, group=g, system=system)
            per=[float(r["F"]) for r in pages if (g=="ALL" or r["group"]==g) and r["system"]==page_system]
            mean=float(sr["F"]); means.append(mean)
            lows.append(mean-min(per)); highs.append(max(per)-mean)
        ax.bar(x + (j-1.5)*width, means, width, label=label,
               yerr=np.array([lows, highs]), capsize=2, error_kw={"elinewidth":.8})
    ax.set_xticks(x, ["2018", "2019A", "2019B", "All"])
    ax.set_ylabel("Test F (%)")
    ax.set_title("Test F-measure by group")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=.25)
    ax.set_axisbelow(True)
    ax.legend(ncols=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(OUT / "fig_test_by_group.png", dpi=180)
    plt.close(fig)

if __name__ == "__main__":
    scale_figure()
    test_figure()
    print("saved outputs/fig_scale_sensitivity.png and outputs/fig_test_by_group.png")





