#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4 — MACE Prediction on MIMIC-III Cardio-Oncology Cohort
=============================================================
Two evaluation modes:

  A) Transfer (zero-shot): Apply existing SCH-trained ensemble to MIMIC-III.
     Uses real_model_predictions_recalibrated.csv to retrain a calibrated
     ensemble on SCH data, then predicts on MIMIC-III.

  B) Within-MIMIC CV: Train fresh ensemble on MIMIC-III via 5-fold CV.

Outputs (all to MIMIC-3/results/):
  predictions_transfer.csv   — transfer predictions + labels
  predictions_within_cv.csv  — within-MIMIC CV predictions
  evaluation_metrics.csv     — AUC, sensitivity, specificity, PPV, NPV, BSS
  feature_importance.csv     — XGB gain importance from within-CV model
  figures/                   — ROC, calibration, DCA, feature importance
"""
import sys, warnings, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                              average_precision_score, confusion_matrix,
                              precision_recall_curve)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────
WORKSPACE  = Path(__file__).resolve().parents[1]
DATA_DIR   = Path(__file__).resolve().parent / "data"
RES_DIR    = Path(__file__).resolve().parent / "results"
FIG_DIR    = RES_DIR / "figures"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Add workspace to path for local imports
sys.path.insert(0, str(WORKSPACE))
from models.ensemble import EnsembleModel

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.edgecolor": "#444", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#E8E8E8", "grid.linewidth": 0.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "legend.framealpha": 0.95,
})

C_SCH  = "#2166AC"
C_MIMIC = "#5E35B1"   # purple for MIMIC

print("=" * 66)
print("  Step 4: MACE Prediction — MIMIC-III Cardio-Oncology Cohort")
print("=" * 66)

# ── Model features (same order as SCH/WHU model) ─────────────
MODEL_FEATURES = [
    "Age", "Sex_M",
    "Apolipoprotein B", "LDL Cholesterol", "Total Cholesterol",
    "Cholinesterase", "Albumin", "Total Bile Acids", "Prealbumin",
    "D-Dimer", "CYFRA 21-1", "Cystatin C", "Triglycerides",
    "Adenosine Deaminase", "Globulin", "Plateletcrit", "SCC Antigen",
    "Direct Bilirubin",
    "No. of Physicians", "Hospital Beds", "GDP per Capita", "No. of Hospitals",
    "Avg. Employee Wage",
    "Glucose", "HDL Cholesterol", "Magnesium", "α-HBDH", "ALT",
    "HBcAb", "LDH", "Phosphorus", "AST", "Uric Acid",
    "Alpha-Fetoprotein", "Total Protein", "HbA1c", "Creatinine",
    "Creatine Kinase", "Potassium", "Sodium", "Apolipoprotein AI",
    "Ferritin", "Amylase",
]

# ── Load MIMIC data ───────────────────────────────────────────
feat_df  = pd.read_csv(DATA_DIR / "features_model.csv", low_memory=False)
mace_df  = pd.read_csv(DATA_DIR / "mace_labels.csv",    low_memory=False)

# Merge features + labels
df = feat_df.merge(mace_df[["HADM_ID", "MACE", "MACE_Types"]],
                   on="HADM_ID", how="inner")

# Restrict to admissions with cancer OR keep all (cardio-oncology defined in step 1)
# Use ALL cohort admissions for breadth; model predicts MACE risk per admission
df = df.dropna(subset=["MACE"]).copy()
df["MACE"] = df["MACE"].astype(int)

print(f"\n  Dataset: {len(df):,} admissions")
print(f"  MACE positive: {df['MACE'].sum():,} ({df['MACE'].mean()*100:.1f}%)")
print(f"  MACE negative: {(df['MACE']==0).sum():,}")

# Build feature matrix with proper filling for model columns
for col in MODEL_FEATURES:
    if col not in df.columns:
        df[col] = np.nan

X_raw = df[MODEL_FEATURES].values.astype(float)
y     = df["MACE"].values

# ── Impute + scale ────────────────────────────────────────────
imputer   = SimpleImputer(strategy="median")
scaler    = StandardScaler()
X_imputed = imputer.fit_transform(X_raw)
X_scaled  = scaler.fit_transform(X_imputed)   # used for RF; XGB handles raw

# XGBoost works better without scaling (tree-based), RF benefits from it
X_for_xgb = X_imputed
X_for_rf  = X_scaled

# Combined matrix: pack both into model (EnsembleModel uses same X for both)
X = X_imputed   # XGB and RF will be trained on this

# ────────────────────────────────────────────────────────────────
# MODE A: Transfer from SCH-trained model
# ────────────────────────────────────────────────────────────────
print("\n── Mode A: Transfer from SCH-trained model ─────────────────")
sch_preds_path = WORKSPACE / "data" / "real_model_predictions_recalibrated.csv"
transfer_probs = None
transfer_auc   = None

if sch_preds_path.exists():
    sch = pd.read_csv(sch_preds_path, low_memory=False)
    sch_s = sch[sch["Center"] == "SCH"].copy()
    sch_whu = sch.copy()

    # Identify feature columns in SCH CSV (original Chinese names)
    SCH_FEAT_CHINESE = [
        "年龄", "载脂蛋白B", "低密度脂蛋白胆固醇", "总胆固醇", "胆碱酯酶",
        "白蛋白", "总胆汁酸", "前白蛋白", "D-二聚体", "细胞角蛋白19片段",
        "胱抑素C", "甘油三脂", "腺苷脱氨酶", "球蛋白", "血小板压积",
        "鳞状细胞癌相关抗原", "直接胆红素", "医生数(人)", "淀粉酶",
        "医院、卫生院床位数(张)", "葡萄糖", "性别", "高密度脂蛋白胆固醇",
        "镁", "α-羟丁酸脱氢酶", "丙氨酸氨基转移酶", "人均地区生产总值(元)",
        "医院、卫生院数(个)", "乙肝核心抗体", "乳酸脱氢酶", "磷",
        "天门冬氨酸氨基转移酶", "尿酸", "甲胎蛋白", "总蛋白",
        "糖化血红蛋白", "肌酐", "肌酸激酶", "职工平均工资(元)",
        "钾", "钠", "载脂蛋白AⅠ", "铁蛋白",
    ]
    available_feats = [f for f in SCH_FEAT_CHINESE if f in sch_s.columns]
    print(f"  SCH features available: {len(available_feats)}/{len(SCH_FEAT_CHINESE)}")

    # English mapping
    from utils.feature_names_en import FEAT_EN
    EN_TO_CHINESE = {v: k for k, v in FEAT_EN.items()}
    EN_TO_CHINESE["Sex_M"] = "性别"   # binary encoding

    # Build SCH training matrix
    X_sch_cols = []
    for feat_en in MODEL_FEATURES:
        cn = EN_TO_CHINESE.get(feat_en)
        if cn and cn in sch_s.columns:
            X_sch_cols.append(cn)
        else:
            X_sch_cols.append(None)

    def build_matrix(df_src, col_map):
        cols = []
        for c in col_map:
            if c and c in df_src.columns:
                cols.append(df_src[c].values.astype(float))
            else:
                cols.append(np.full(len(df_src), np.nan))
        return np.column_stack(cols)

    X_sch = build_matrix(sch_s, X_sch_cols)
    y_sch = sch_s["MACE"].values

    imp_sch = SimpleImputer(strategy="median")
    X_sch_imp = imp_sch.fit_transform(X_sch)

    print("  Training SCH ensemble (full fit) ...")
    sch_model = EnsembleModel(xgb_rounds=300, n_folds=5, random_state=42)
    sch_model.fit(X_sch_imp, y_sch)

    # Impute MIMIC using SCH medians
    X_mimic_imp = imp_sch.transform(X_raw)
    transfer_probs = sch_model.predict(X_mimic_imp)
    transfer_auc   = roc_auc_score(y, transfer_probs) if y.sum() > 0 else np.nan
    print(f"  Transfer AUC (SCH→MIMIC): {transfer_auc:.4f}")

    df["prob_transfer"] = transfer_probs
else:
    print("  WARNING: SCH training data not found — skipping transfer mode.")
    df["prob_transfer"] = np.nan

# ────────────────────────────────────────────────────────────────
# MODE B: Within-MIMIC 5-fold CV
# ────────────────────────────────────────────────────────────────
print("\n── Mode B: Within-MIMIC 5-fold CV ──────────────────────────")

if y.sum() < 10:
    print("  WARNING: Too few MACE events for within-CV training. Skipping.")
    df["prob_cv"]  = np.nan
    cv_auc = np.nan
    fold_aucs = []
else:
    mimic_model = EnsembleModel(xgb_rounds=200, n_folds=5, random_state=42)
    oof_probs, fold_aucs = mimic_model.fit_cv(X, y, verbose=True)
    df["prob_cv"] = oof_probs
    cv_auc = roc_auc_score(y, oof_probs)
    print(f"  Within-MIMIC CV AUC: {cv_auc:.4f}")

# ── Isotonic recalibration for CV predictions ─────────────────
from sklearn.isotonic import IsotonicRegression
if not np.all(np.isnan(df["prob_cv"].values)):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(df["prob_cv"].values, y)
    df["prob_cv_cal"] = iso.predict(df["prob_cv"].values)
else:
    df["prob_cv_cal"] = df["prob_cv"]

# ── Evaluation metrics ────────────────────────────────────────
def evaluate(y_true, y_prob, label):
    if np.all(np.isnan(y_prob)) or y_true.sum() == 0:
        return {}
    mask = ~np.isnan(y_prob)
    yt, yp = y_true[mask], y_prob[mask]
    auc    = roc_auc_score(yt, yp)
    ap     = average_precision_score(yt, yp)
    brier  = brier_score_loss(yt, yp)
    null_b = brier_score_loss(yt, np.full_like(yp, yt.mean()))
    bss    = 1 - brier / null_b
    # Threshold at Youden J
    fpr, tpr, thrs = roc_curve(yt, yp)
    j = np.argmax(tpr - fpr)
    thr = thrs[j]
    yhat = (yp >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yhat).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0
    return {
        "Label": label, "N": int(mask.sum()), "N_MACE": int(yt.sum()),
        "AUC": round(auc, 4), "AP": round(ap, 4), "BSS": round(bss, 4),
        "Sensitivity": round(sens, 4), "Specificity": round(spec, 4),
        "PPV": round(ppv, 4), "NPV": round(npv, 4),
        "Optimal_Threshold": round(float(thr), 4),
        "Brier_Score": round(brier, 4),
    }

metrics_rows = []
if not np.all(np.isnan(df["prob_transfer"].values)):
    metrics_rows.append(evaluate(y, df["prob_transfer"].values, "Transfer (SCH→MIMIC)"))
if not np.all(np.isnan(df["prob_cv"].values)):
    metrics_rows.append(evaluate(y, df["prob_cv"].values,      "Within-MIMIC CV (raw)"))
    metrics_rows.append(evaluate(y, df["prob_cv_cal"].values,  "Within-MIMIC CV (isotonic-cal)"))

metrics_df = pd.DataFrame(metrics_rows)
print(f"\n  ── Evaluation Metrics ──")
if len(metrics_df) > 0:
    print(metrics_df.to_string(index=False))

# ── Save predictions ──────────────────────────────────────────
df.to_csv(RES_DIR / "predictions.csv", index=False)
metrics_df.to_csv(RES_DIR / "evaluation_metrics.csv", index=False)
print(f"\n✓ Saved: predictions.csv ({len(df):,} rows)")
print(f"✓ Saved: evaluation_metrics.csv")

# ── Feature importance (within-CV model) ─────────────────────
if y.sum() >= 10:
    # Retrain on full MIMIC for importance
    full_model = EnsembleModel(xgb_rounds=200, random_state=42)
    full_model.fit(X, y)
    imp = full_model.feature_importances(MODEL_FEATURES, top_n=43)
    imp_df = pd.DataFrame(list(imp.items()), columns=["Feature", "Importance_Gain"])
    imp_df = imp_df.sort_values("Importance_Gain", ascending=False)
    imp_df.to_csv(RES_DIR / "feature_importance.csv", index=False)
    print(f"✓ Saved: feature_importance.csv")
else:
    imp_df = pd.DataFrame(columns=["Feature", "Importance_Gain"])

# ── Figure: ROC + Calibration + Feature Importance ───────────
fig = plt.figure(figsize=(170/25.4, 110/25.4))
gs  = gridspec.GridSpec(2, 2, figure=fig, wspace=0.42, hspace=0.50,
                        left=0.10, right=0.97, top=0.88, bottom=0.10)
ax_roc  = fig.add_subplot(gs[0, 0])
ax_cal  = fig.add_subplot(gs[0, 1])
ax_feat = fig.add_subplot(gs[1, :])

# — ROC curves ————————————————————————————————————————————————
plot_configs = []
if not np.all(np.isnan(df["prob_transfer"].values)):
    plot_configs.append((df["prob_transfer"].values, "Transfer SCH→MIMIC", "#D6604D", 2.0))
if not np.all(np.isnan(df["prob_cv_cal"].values)):
    plot_configs.append((df["prob_cv_cal"].values, "Within-MIMIC CV (calibrated)", C_MIMIC, 1.8))

for probs, lbl, col, lw in plot_configs:
    mask = ~np.isnan(probs)
    if mask.sum() < 10 or y[mask].sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(y[mask], probs[mask])
    auc = roc_auc_score(y[mask], probs[mask])
    ax_roc.plot(fpr, tpr, color=col, lw=lw, label=f"{lbl}\n(AUC={auc:.3f})")

ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
ax_roc.set_xlabel("1 – Specificity", fontsize=8)
ax_roc.set_ylabel("Sensitivity", fontsize=8)
ax_roc.set_title("(A) ROC Curves\nMIMIC-III Cardio-Oncology", fontsize=8, fontweight="bold")
ax_roc.legend(fontsize=6.5, loc="lower right")

# — Calibration ───────────────────────────────────────────────
for probs, lbl, col, _ in plot_configs:
    mask = ~np.isnan(probs)
    if mask.sum() < 20 or y[mask].sum() == 0:
        continue
    frac, mean_pred = calibration_curve(y[mask], probs[mask],
                                         n_bins=10, strategy="quantile")
    ax_cal.plot(mean_pred, frac, "o-", color=col, lw=1.6, ms=3.5,
                label=lbl.split("\n")[0])

ax_cal.plot([0, 1], [0, 1], "k--", lw=0.9, alpha=0.5, label="Perfect")
ax_cal.set_xlabel("Mean Predicted Probability", fontsize=8)
ax_cal.set_ylabel("Observed MACE Fraction", fontsize=8)
ax_cal.set_title("(B) Calibration Curves", fontsize=8, fontweight="bold")
ax_cal.legend(fontsize=6.5, loc="upper left")

# — Feature Importance ─────────────────────────────────────────
if len(imp_df) > 0:
    top15 = imp_df.head(15)
    y_pos = np.arange(len(top15))
    ax_feat.barh(y_pos, top15["Importance_Gain"].values,
                 color=C_MIMIC, alpha=0.85, height=0.65, edgecolor="none")
    ax_feat.set_yticks(y_pos)
    ax_feat.set_yticklabels(top15["Feature"].values, fontsize=7)
    ax_feat.set_xlabel("XGBoost Gain (normalised)", fontsize=8)
    ax_feat.set_title("(C) Top-15 Feature Importance (Within-MIMIC Model)",
                      fontsize=8, fontweight="bold")
    ax_feat.invert_yaxis()
    ax_feat.grid(axis="x", color="#E8E8E8", lw=0.5)
else:
    ax_feat.axis("off")
    ax_feat.text(0.5, 0.5, "Insufficient events for\nfeature importance",
                 ha="center", va="center", transform=ax_feat.transAxes, fontsize=9)

fig.suptitle("MIMIC-III Cardio-Oncology: MACE Prediction Results",
             fontsize=9, fontweight="bold", y=0.975)

fig_path = FIG_DIR / "Fig_MIMIC_Results.pdf"
plt.savefig(fig_path, dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig(FIG_DIR / "Fig_MIMIC_Results.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"✓ Saved: {fig_path}")

# ── Figure 2: Cohort characteristic bar chart ─────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(170/25.4, 72/25.4))
fig2.suptitle("MIMIC-III Cardio-Oncology: Cohort Characteristics",
              fontsize=9, fontweight="bold", y=0.995)

# Panel 1: MACE rate
axes2[0].bar(["No MACE", "MACE"],
             [(y == 0).sum(), y.sum()],
             color=["#78909C", "#D32F2F"], width=0.5)
for ax in axes2:
    ax.grid(axis="y", color="#E8E8E8", lw=0.5)
axes2[0].set_ylabel("Count", fontsize=8)
axes2[0].set_title(f"(A) MACE Distribution\nRate={y.mean()*100:.1f}%", fontsize=8, fontweight="bold")

# Panel 2: Age distribution by MACE
age_vals = df["Age"].values
axes2[1].hist(age_vals[y == 0], bins=25, alpha=0.65, color="#78909C", label="No MACE", density=True)
axes2[1].hist(age_vals[y == 1], bins=25, alpha=0.65, color="#D32F2F", label="MACE", density=True)
axes2[1].set_xlabel("Age (years)", fontsize=8)
axes2[1].set_ylabel("Density", fontsize=8)
axes2[1].set_title("(B) Age Distribution by MACE", fontsize=8, fontweight="bold")
axes2[1].legend(fontsize=6.5)

# Panel 3: MACE type breakdown
if not np.all(df["MACE_Types"].isna()):
    mace_types_df = df[df["MACE"] == 1]["MACE_Types"].str.split("|").explode()
    mace_type_counts = mace_types_df.value_counts().head(6)
    axes2[2].barh(range(len(mace_type_counts)), mace_type_counts.values,
                  color=C_MIMIC, alpha=0.85)
    axes2[2].set_yticks(range(len(mace_type_counts)))
    axes2[2].set_yticklabels(mace_type_counts.index, fontsize=7)
    axes2[2].set_xlabel("Count", fontsize=8)
    axes2[2].invert_yaxis()
    axes2[2].set_title("(C) MACE Sub-type Breakdown", fontsize=8, fontweight="bold")

fig2.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(FIG_DIR / "Fig_MIMIC_Cohort.pdf", dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig(FIG_DIR / "Fig_MIMIC_Cohort.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"✓ Saved: Fig_MIMIC_Cohort.pdf")

print(f"\n{'='*66}")
print(f"  All outputs saved to: {RES_DIR}")
print(f"  Figures:              {FIG_DIR}")
print(f"{'='*66}")
print("\nStep 4 complete.\n")
