"""Builds figures/results.png from evaluation_results.csv (run evaluate.py first)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("evaluation_results.csv")
keep = {
    "Popularity (weighted rating)": "Popularity",
    "Foundation: Banik content, single seed": "Foundation\n(Banik)",
    "E1 Content profile": "E1 Content\nprofile",
    "E2 PureSVD CF profile": "E2 PureSVD\nCF",
    "E3 Hybrid (0.1/0.8/0.1)": "E3 Hybrid",
    "E3+E4 Hybrid + MMR (div=0.30)": "E3+E4 Hybrid\n+ MMR 0.30",
}
df = df[df.System.isin(keep)].copy()
df["label"] = df.System.map(keep)
order = list(keep.values())
colors = {"warm": "#2a78d6", "cold": "#eb6834"}
names = {"warm": "Warm start (full history)", "cold": "Cold start (3 liked titles)"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
x = np.arange(len(order)); w = 0.38
for ax, metric, title in [(axes[0], "Precision@10", "A. Accuracy: Precision@10"),
                          (axes[1], "ILD", "B. Variety: intra-list diversity")]:
    for j, sc in enumerate(["warm", "cold"]):
        vals = [df[(df.Scenario == sc) & (df.label == o)][metric].iloc[0] for o in order]
        bars = ax.bar(x + (j - 0.5) * w, vals, w - 0.03, color=colors[sc], label=names[sc],
                      hatch="//" if sc == "cold" else None, edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}" if metric == "ILD" else f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.5, color="#333")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6); ax.set_axisbelow(True)
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("figures/results.png", dpi=200)
print("saved")
