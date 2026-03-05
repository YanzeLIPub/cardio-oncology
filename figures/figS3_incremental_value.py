#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Appendix Figure S3 — Incremental Value over Age-Only Baseline (SCH Cohort)
===========================================================================
Compares full ensemble model vs age-only logistic regression baseline.
Reported ΔAUC = +0.020; ΔBSS = +0.005.

Panels:
  (A) ROC curves: Age-only vs Full model (SCH within-CV)
  (B) Calibration curves: Age-only vs Full model
  (C) Bar chart: AUC and BSS side-by-side for both models

Output: outputs/final/FigS3_Incremental_Value_SCH.{pdf,png}
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR  = WORKSPACE / "data"
OUT_DIR   = WORKSPACE / "outputs" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREDS_FILE = DATA_DIR / "real_model_predictions_v2.csv"
if not PREDS_FILE.exists():
    PREDS_FILE = Path(r"D:\cardio-oncology\outputs") / "real_model_predictions_v2.csv"

# Recalibrated file for calibrated BSS comparison
RECAL_FILE = DATA_DIR / "real_model_predictions_recalibrated.csv"
if not RECAL_FILE.exists():
    RECAL_FILE = Path(r"D:\cardio-oncology\outputs\ensemble_model") / "real_model_predictions_recalibrated.csv"

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
C_AGE   = "#969696"

# ── Load SCH data ─────────────────────────────────────────────────────────────
preds = pd.read_csv(PREDS_FILE)
preds.columns = preds.columns.str.strip()
sch = preds[preds["Center"] == "SCH"].copy().reset_index(drop=True)
print(f"SCH: n={len(sch):,}  MACE={sch['MACE'].sum():,}")

# Use isotonic-calibrated probabilities for BSS (raw probs are uncalibrated)
p_full_raw = sch["predicted_prob_within_cv"].values
if RECAL_FILE.exists():
    recal = pd.read_csv(RECAL_FILE)
    recal.columns = recal.columns.str.strip()
    sch_recal = recal[recal["Center"] == "SCH"].copy().reset_index(drop=True)
    p_full_cal = sch_recal["prob_isotonic"].values if len(sch_recal) == len(sch) else p_full_raw
    print("  Using isotonic-calibrated probabilities for full model")
else:
    p_full_cal = p_full_raw

y   = sch["MACE"].values
age = sch["年龄"].values

# ── Age-only model (5-fold CV cross-validated predictions) ────────────────────
scaler = StandardScaler()
X_age  = scaler.fit_transform(age.reshape(-1, 1))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Use plain LR (no class_weight) so predicted probs are calibrated and BSS is meaningful
p_age = cross_val_predict(
    LogisticRegression(C=0.1, max_iter=500, random_state=42),
    X_age, y, cv=skf, method="predict_proba"
)[:, 1]

# Use raw probs for AUC (ordering preserved) and calibrated for BSS
p_full = p_full_raw  # AUC unaffected by calibration

# ── Metrics ────────────────────────────────────────────────────────────────────
auc_full = roc_auc_score(y, p_full)
auc_age  = roc_auc_score(y, p_age)
delta_auc = auc_full - auc_age

brier_null  = brier_score_loss(y, np.full_like(p_full, y.mean()))
brier_full  = brier_score_loss(y, p_full_cal)   # calibrated
brier_age   = brier_score_loss(y, p_age)
bss_full    = 1 - brier_full / brier_null
bss_age     = 1 - brier_age  / brier_null
delta_bss   = bss_full - bss_age

# For ROC and calibration subplots use calibrated full model
p_full_plot = p_full_cal

print(f"Age-only  AUC={auc_age:.4f}  BSS={bss_age:.4f}")
print(f"Full      AUC={auc_full:.4f}  BSS={bss_full:.4f}")
print(f"ΔAUC={delta_auc:+.4f}  ΔBSS={delta_bss:+.4f}")

# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(170 / 25.4, 80 / 25.4))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40,
                        left=0.07, right=0.97, top=0.79, bottom=0.14)

ax1 = fig.add_subplot(gs[0])  # ROC
ax2 = fig.add_subplot(gs[1])  # Calibration
ax3 = fig.add_subplot(gs[2])  # Bar summary

# ── (A) ROC curves ────────────────────────────────────────────────────────────
for p, label, color, lw in [
    (p_age,       f"Age-only (AUC={auc_age:.3f})",   C_AGE, 1.4),
    (p_full_plot, f"Full model (AUC={auc_full:.3f})", C_SCH, 2.0),
]:
    fpr, tpr, _ = roc_curve(y, p)
    ax1.plot(fpr, tpr, color=color, lw=lw, label=label)

ax1.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Reference")
ax1.fill_between([0, 1], [0, 1], [0, 1], alpha=0)  # invisible
ax1.set_xlabel("1 – Specificity", fontsize=8)
ax1.set_ylabel("Sensitivity", fontsize=8)
ax1.set_title("(A) ROC Curves — SCH", fontsize=8, fontweight="bold")
ax1.legend(loc="lower right", fontsize=6.5)
ax1.text(0.05, 0.90, f"ΔAUC = {delta_auc:+.3f}", transform=ax1.transAxes,
         fontsize=7.5, color=C_SCH, fontweight="bold")

# ── (B) Calibration curves ────────────────────────────────────────────────────
for p, label, color in [
    (p_age,       "Age-only", C_AGE),
    (p_full_plot, "Full model (calibrated)", C_SCH),
]:
    frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax2.plot(mean_pred, frac, "o-", color=color, lw=1.6, ms=3.5, label=label)

ax2.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect")
ax2.set_xlabel("Mean predicted probability", fontsize=8)
ax2.set_ylabel("Observed MACE fraction", fontsize=8)
ax2.set_title("(B) Calibration — SCH", fontsize=8, fontweight="bold")
ax2.legend(loc="upper left", fontsize=6.5)

# ── (C) AUC & BSS bar chart ───────────────────────────────────────────────────
metrics = {"AUC": (auc_age, auc_full), "BSS": (bss_age, bss_full)}
x      = np.arange(len(metrics))
width  = 0.30
labels = list(metrics.keys())

bars_age  = ax3.bar(x - width/2, [v[0] for v in metrics.values()],
                     width, color=C_AGE, alpha=0.85, label="Age-only", zorder=3)
bars_full = ax3.bar(x + width/2, [v[1] for v in metrics.values()],
                     width, color=C_SCH, alpha=0.85, label="Full model", zorder=3)

# Delta annotations
for i, (key, (v_age, v_full)) in enumerate(metrics.items()):
    delta = v_full - v_age
    top   = max(v_age, v_full)
    ax3.annotate(f"Δ{delta:+.3f}",
                 xy=(i, top + 0.005), ha="center", va="bottom",
                 fontsize=7, color=C_SCH, fontweight="bold")

ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=8)
ax3.set_ylabel("Metric value", fontsize=8)
ax3.set_title("(C) Incremental Gain — SCH", fontsize=8, fontweight="bold")
ax3.legend(fontsize=6.5)
ax3.set_ylim(0, max(auc_full, bss_full) * 1.25)

fig.suptitle(
    "Appendix Figure S3. Incremental value over age-only baseline — SCH cohort\n"
    f"ΔAUC = {delta_auc:+.3f}; ΔBSS = {delta_bss:+.3f}",
    fontsize=8.5, fontweight="bold", y=0.988)

for ext in (".pdf", ".png"):
    fp = OUT_DIR / f"FigS3_Incremental_Value_SCH{ext}"
    plt.savefig(fp, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"Saved: {fp}")
plt.close()
print("Done — Figure S3")
