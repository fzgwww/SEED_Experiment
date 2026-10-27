# -*- coding: utf-8 -*-
"""Grouped bar chart of BaselinePro Neg / Neu / Pos scores by seed."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

seeds = ["41", "42", "43", "44", "45"]
neg = [0.3726, 0.9427, 0.5808, 0.7151, 0.6040]
neu = [0.1111, 0.2778, 0.6090, 0.3889, 0.5117]
pos = [0.8056, 0.2778, 0.4306, 0.3750, 0.7100]

x = np.arange(len(seeds))
width = 0.26
colors = {"Neg": "#2563EB", "Neu": "#808080", "Pos": "#FACC15"}

fig, ax = plt.subplots(figsize=(8, 5))
bars = [
    ax.bar(x - width, neg, width, label="Neg", color=colors["Neg"]),
    ax.bar(x,          neu, width, label="Neu", color=colors["Neu"]),
    ax.bar(x + width, pos, width, label="Pos", color=colors["Pos"]),
]

for group in bars:
    for rect in group:
        h = rect.get_height()
        ax.annotate(
            f"{h:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

ax.set_xticks(x)
ax.set_xticklabels([f"Seed {s}" for s in seeds])
ax.set_ylabel("Score")
ax.set_title("BaselinePro: Neg / Neu / Pos by Seed")
ax.set_ylim(0, 1.05)
ax.legend(loc="upper right", frameon=False)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("baseline_grouped_bar.png", dpi=150)
print("saved baseline_grouped_bar.png")
