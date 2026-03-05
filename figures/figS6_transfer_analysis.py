#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Appendix Figure S6 — Bidirectional Transfer Analysis
=====================================================
Within-cohort and cross-cohort AUC comparison demonstrating
context-specificity of model performance.
Includes clinical quantification of performance differences at
varying decision thresholds.

Data from data/OptionB_Summary.csv + data/real_model_predictions_v2.csv

Panels:
  (A) AUC matrix heatmap: 2×2 (train cohort × test cohort)
  (B) Net benefit loss at varying thresholds (transfer penalty curves)
  (C) Reclassification table: cases correctly/incorrectly classified at
      each threshold by within-cohort vs transferred model

Output: outputs/final/FigS6_Transfer_Analysis.{pdf,png}
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR  = WORKSPACE / "data"
OUT_DIR   = WORKSPACE / "outputs" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OPT_FILE   = DATA_DIR / "OptionB_Summary.csv"
PREDS_FILE = DATA_DIR / "real_model_predictions_v2.csv"
if not PREDS_FILE.exists():
    PREDS_FILE = Path(r"D:\cardio-oncology\outputs") / "real_model_predictions_v2.csv"

# ── Lancet style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#E5E5E5", "grid.linewidth": 0.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.framealpha": 0.95,
})

C_SCH   = "#2166AC"
C_WHU   = "#D6604D"
C_DIAG  = "#43A047"   # on-diagonal (within-cohort)
C_OFFD  = "#969696"   # off-diagonal (cross-cohort)

# ── Load data ────────────────────────────────────────────────────────────────
opt = pd.read_csv(OPT_FILE)
opt.columns = opt.columns.str.strip()
summary = {row["Metric"]: float(row["Value"]) for _, row in opt.iterrows()}

# Precomputed AUC values
auc_sch_within  = summary.get("SCH_CV_AUC",       0.6573)
auc_whu_within  = summary.get("WHU_CV_AUC",       0.8401)
auc_sch_to_whu  = summary.get("SCH_to_WHU_transfer", 0.4381)
auc_whu_to_sch  = summary.get("WHU_to_SCH_transfer", 0.4847)

print(f"SCH→SCH: {auc_sch_within:.4f}  WHU→WHU: {auc_whu_within:.4f}")
print(f"SCH→WHU: {auc_sch_to_whu:.4f}  WHU→SCH: {auc_whu_to_sch:.4f}")

# Build AUC matrix
auc_matrix = np.array([
    [auc_sch_within, auc_sch_to_whu],   # trained on SCH; tested on SCH, WHU
    [auc_whu_to_sch, auc_whu_within],   # trained on WHU; tested on SCH, WHU
])

# ── Load prediction columns (within-cv and cross-cohort) ─────────────────────
preds = pd.read_csv(PREDS_FILE)
preds.columns = preds.columns.str.strip()
sch = preds[preds["Center"] == "SCH"].copy().reset_index(drop=True)
whu = preds[preds["Center"] == "WHU"].copy().reset_index(drop=True)

y_sch   = sch["MACE"].values
y_whu   = whu["MACE"].values
p_sch_w = sch["predicted_prob_within_cv"].values            # SCH model on SCH
p_whu_w = whu["predicted_prob_within_cv"].values            # WHU model on WHU
p_whu_x = whu["pred_from_sch"].fillna(whu["predicted_prob"]).values  # SCH model on WHU

# ── Figure ───────────────────────────────────────────────────────────────────
# Vertical 3×1 layout so each panel is closer to square
fig = plt.figure(figsize=(90 / 25.4, 252 / 25.4))
gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55,
                        left=0.18, right=0.93, top=0.91, bottom=0.05)

ax1 = fig.add_subplot(gs[0])  # heatmap
ax2 = fig.add_subplot(gs[1])  # ROC overlay
ax3 = fig.add_subplot(gs[2])  # net-benefit penalty

# ── (A) AUC heatmap ──────────────────────────────────────────────────────────
cmap = plt.cm.RdYlGn
norm = mcolors.Normalize(vmin=0.40, vmax=0.88)

im = ax1.imshow(auc_matrix, cmap=cmap, norm=norm, aspect="auto")
ax1.set_xticks([0, 1]); ax1.set_xticklabels(["SCH\n(test)", "WHU\n(test)"], fontsize=8)
ax1.set_yticks([0, 1]); ax1.set_yticklabels(["SCH model", "WHU model"], fontsize=8)

train_labels = ["SCH model\n(trained on SCH)", "WHU model\n(trained on WHU)"]
for r in range(2):
    for c in range(2):
        val = auc_matrix[r, c]
        fc = "white" if val < 0.64 else "black"
        is_diag  = (r == c)
        weight   = "bold" if is_diag else "normal"
        marker   = "✓" if is_diag else "↗" if r == 0 and c == 1 else "↙"
        ax1.text(c, r, f"{val:.4f}\n{marker}",
                 ha="center", va="center", fontsize=8.5,
                 color="black", fontweight=weight)

cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label("AUC", fontsize=7.5)
cbar.ax.tick_params(labelsize=6.5)

ax1.set_title("(A) AUC Transfer Matrix\n(diagonal = within-cohort)",
              fontsize=8, fontweight="bold", pad=5)

# ── (B) ROC curves ────────────────────────────────────────────────────────────
for y, p, label, color, ls, lw in [
    (y_sch, p_sch_w, f"SCH→SCH (AUC={auc_sch_within:.3f})", C_SCH, "-",  2.0),
    (y_whu, p_whu_w, f"WHU→WHU (AUC={auc_whu_within:.3f})", C_WHU, "-",  2.0),
    (y_whu, p_whu_x, f"SCH→WHU (AUC={auc_sch_to_whu:.3f})", C_SCH, "--", 1.4),
]:
    fpr, tpr, _ = roc_curve(y, p)
    ax2.plot(fpr, tpr, color=color, ls=ls, lw=lw, label=label)

ax2.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
ax2.set_xlabel("1 – Specificity", fontsize=8)
ax2.set_ylabel("Sensitivity", fontsize=8)
ax2.set_title("(B) ROC Curves", fontsize=8, fontweight="bold")
ax2.legend(loc="lower right", fontsize=6.0)

# ── (C) Net-benefit penalty at varying thresholds ────────────────────────────
def net_benefit(y, prob, t):
    n  = len(y)
    pos = (prob >= t)
    tp  = np.sum(pos & (y == 1))
    fp  = np.sum(pos & (y == 0))
    if t >= 1.0: return 0.0
    return tp / n - fp / n * t / (1.0 - t)

thresholds = np.linspace(0.02, 0.25, 200)
nb_sch_within = np.array([net_benefit(y_sch, p_sch_w, t) for t in thresholds])
nb_whu_within = np.array([net_benefit(y_whu, p_whu_w, t) for t in thresholds])
nb_sch_to_whu = np.array([net_benefit(y_whu, p_whu_x, t) for t in thresholds])

ax3.plot(thresholds * 100, nb_sch_within, color=C_SCH, lw=2.0,
         label=f"SCH→SCH (within)")
ax3.plot(thresholds * 100, nb_whu_within, color=C_WHU, lw=2.0,
         label=f"WHU→WHU (within)")
ax3.plot(thresholds * 100, nb_sch_to_whu, color=C_SCH, lw=1.4, ls="--",
         label=f"SCH→WHU (transfer)")
ax3.fill_between(thresholds * 100,
                 nb_sch_to_whu, nb_whu_within,
                 where=nb_whu_within > nb_sch_to_whu,
                 alpha=0.15, color=C_WHU, label="WHU transfer penalty")
ax3.axhline(0, color="grey", lw=0.8, ls=":")
ax3.set_xlabel("Decision threshold (%)", fontsize=8)
ax3.set_ylabel("Net benefit", fontsize=8)
ax3.set_title("(C) Net Benefit: Within vs\nTransfer (SCH→WHU)", fontsize=8, fontweight="bold")
ax3.legend(fontsize=6.0)

# Performance gap annotation
fig.suptitle(
    "Appendix Figure S6. Bidirectional transfer analysis\n"
    f"Within-cohort vs cross-cohort AUC\n"
    f"SCH→SCH: {auc_sch_within:.3f}  WHU→WHU: {auc_whu_within:.3f}\n"
    f"SCH→WHU: {auc_sch_to_whu:.3f}  WHU→SCH: {auc_whu_to_sch:.3f}",
    fontsize=8, fontweight="bold", y=0.993)

for ext in (".pdf", ".png"):
    fp = OUT_DIR / f"FigS6_Transfer_Analysis{ext}"
    plt.savefig(fp, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"Saved: {fp}")
plt.close()
print("Done — Figure S6")
