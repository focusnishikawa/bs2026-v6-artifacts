#!/usr/bin/env python3
"""Regenerate the three result figures from the 2026-08-11 re-experiment.

  fig1_rank_distribution_v6 : rank of the correct species, three configurations
  fig3_finetune_curve_v6    : loss / top-1 curves of the re-run fine-tuning
  fig4_iucn_recall_v6       : per-IUCN-category recall, with the structural
                              ceiling of VU drawn explicitly (20 of the 60 VU
                              images are not Accipitriformes, so Stage 4 cannot
                              identify them and recall cannot exceed 40/60)
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
DATA = Path(__file__).resolve().parents[1].parent / "data_v6"

CFG = ["IN-Base", "FT-Flat", "FT-Hier"]
COLORS = {"IN-Base": "#6e6e6e", "FT-Flat": "#1f6fb4", "FT-Hier": "#c0392b"}
LABELS = {"IN-Base": "IN-Base (ImageNet features)",
          "FT-Flat": "FT-Flat (fine-tuned, flat)",
          "FT-Hier": "FT-Hier (fine-tuned, hierarchical)"}

# ---------------------------------------------------------------- fig 1
rank = json.loads((DATA / "rank_dist_v6.json").read_text())
ranks = ["1", "2", "3", "4", "5", "6"]
fig, ax = plt.subplots(figsize=(8.9, 4.1))
w = 0.26
x = np.arange(len(ranks))
for i, c in enumerate(CFG):
    vals = [rank[c][r] / 260 * 100 for r in ranks]
    ax.bar(x + (i - 1) * w, vals, w, label=LABELS[c], color=COLORS[c])
ax.set_xticks(x)
ax.set_xticklabels(["1", "2", "3", "4", "5", r"$\geq$6"])
ax.set_xlabel("Rank of correct species (1 = top-1 hit)")
ax.set_ylabel("Frequency (%)")
ax.legend(fontsize=9, frameon=True)
ax.grid(axis="y", color="0.9")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("fig1_rank_distribution_v6.pdf")
fig.savefig("fig1_rank_distribution_v6.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- fig 3
rows = list(csv.DictReader(open(DATA / "finetune_log.csv")))
ep = [int(r["epoch"]) for r in rows]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
axes[0].plot(ep, [float(r["train_loss"]) for r in rows], "o-", color="#1f6fb4",
             ms=4, label="train")
axes[0].plot(ep, [float(r["val_loss"]) for r in rows], "s-", color="#c0392b",
             ms=4, label="val")
axes[0].set_title("(a) Loss", fontsize=11)
axes[0].set_ylabel("Cross-entropy loss")
axes[1].plot(ep, [float(r["train_acc"]) for r in rows], "o-", color="#1f6fb4",
             ms=4, label="train")
axes[1].plot(ep, [float(r["val_acc"]) for r in rows], "s-", color="#c0392b",
             ms=4, label="val")
axes[1].set_title("(b) Accuracy", fontsize=11)
axes[1].set_ylabel("Top-1 accuracy")
for a in axes:
    a.set_xlabel("Epoch")
    a.grid(color="0.9", linestyle="--")
    a.set_axisbelow(True)
    a.legend(fontsize=9)
fig.tight_layout()
fig.savefig("fig3_finetune_curve_v6.pdf")
fig.savefig("fig3_finetune_curve_v6.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- fig 4
# hit counts / totals from RESULTS_v6_rebuild.md
REC = {
    "CR": {"IN-Base": (14, 30), "FT-Flat": (15, 30), "FT-Hier": (17, 30)},
    "EN": {"IN-Base": (20, 50), "FT-Flat": (34, 50), "FT-Hier": (35, 50)},
    "VU": {"IN-Base": (25, 60), "FT-Flat": (26, 60), "FT-Hier": (22, 60)},
}
CEIL = {"CR": 30 / 30, "EN": 50 / 50, "VU": 40 / 60}
cats = list(REC)
fig, ax = plt.subplots(figsize=(8.0, 4.4))
x = np.arange(len(cats))
for i, c in enumerate(CFG):
    vals = [REC[k][c][0] / REC[k][c][1] for k in cats]
    b = ax.bar(x + (i - 1) * w, vals, w, label=LABELS[c], color=COLORS[c])
    ax.bar_label(b, fmt="%.2f", fontsize=8, padding=2)
for i, k in enumerate(cats):
    ax.hlines(CEIL[k], i - 1.6 * w, i + 1.6 * w, color="black",
              linestyle=":", linewidth=1.6,
              label="structural ceiling" if i == 0 else None)
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.set_ylabel("Recall")
ax.set_ylim(0, 1.12)
ax.legend(fontsize=8.5, ncol=2, frameon=False,
          loc="upper center", bbox_to_anchor=(0.5, -0.11))
ax.grid(axis="y", color="0.9")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("fig4_iucn_recall_v6.pdf")
fig.savefig("fig4_iucn_recall_v6.png", dpi=200)
print("wrote fig1_rank_distribution_v6 / fig3_finetune_curve_v6 / fig4_iucn_recall_v6")
