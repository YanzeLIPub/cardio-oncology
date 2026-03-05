#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Admission-Level MACE Prediction -- Parallel to v11 Patient-Level Analysis
=========================================================================
Strategy:
  - SCH/WHU: Use raw encounter-level rows (no aggregation). Each encounter row
    gets the patient-level MACE label (patient ever had MACE). GroupKFold by
    binganhao is used to prevent data leakage across folds.
  - MIMIC: Already admission-level (HADM_ID). StratifiedKFold (admissions are
    independent). Patients with in-hospital death excluded from extra analysis.

Outputs saved to Manuscript/Admission model/outputs/:
  - predictions_admission_SCH.csv
  - predictions_admission_WHU.csv
  - predictions_admission_MIMIC.csv
  - summary_metrics.csv          -- AUC, Brier, PRAUC, N per cohort/fold
  - Figure*.pdf/png
  - Table_Admission_vs_Patient_Comparison.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                              average_precision_score)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import argparse

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(r"D:\cardio-oncology-prediction")
SCH_FILE = ROOT / "SCH_processed_data.csv"
WHU_FILE = ROOT / "WHU_Cancer_processed_data.csv"
MIMIC_FEAT = ROOT / "MIMIC-3" / "data" / "features_model.csv"
MIMIC_LABELS = ROOT / "MIMIC-3" / "data" / "mace_labels.csv"
LABEL_FILE = ROOT / "data" / "real_model_predictions_v2.csv"
OUT_DIR    = ROOT / "Manuscript" / "Admission model" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Lancet style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "axes.grid": False, "grid.color": "#E5E5E5", "grid.linewidth": 0.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.framealpha": 0.95,
})

SAVE_DPI = 900


def save_figure(fig, stem, rect=None):
    if rect is not None:
        fig.tight_layout(rect=rect)
    else:
        fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(OUT_DIR / f"{stem}{ext}", dpi=SAVE_DPI,
                    bbox_inches="tight", pad_inches=0.05,
                    facecolor="white")
    plt.close(fig)

C_SCH  = "#2166AC"
C_WHU  = "#D6604D"
C_MIMC = "#4DAC26"

# ── 43 shared features (Chinese, SCH/WHU) ─────────────────────────────────────
FEATS_43_ZH = [
    "年龄", "载脂蛋白B", "低密度脂蛋白胆固醇", "总胆固醇", "胆碱酯酶", "白蛋白",
    "总胆汁酸", "前白蛋白", "D-二聚体", "细胞角蛋白19片段", "胱抑素C", "甘油三脂",
    "腺苷脱氨酶", "球蛋白", "血小板压积", "鳞状细胞癌相关抗原", "直接胆红素",
    "医生数(人)", "淀粉酶", "医院、卫生院床位数(张)", "葡萄糖", "性别",
    "高密度脂蛋白胆固醇", "镁", "α-羟丁酸脱氢酶", "丙氨酸氨基转移酶",
    "人均地区生产总值(元)", "医院、卫生院数(个)", "乙肝核心抗体", "乳酸脱氢酶",
    "磷", "天门冬氨酸氨基转移酶", "尿酸", "甲胎蛋白", "总蛋白", "糖化血红蛋白",
    "肌酐", "肌酸激酶", "职工平均工资(元)", "钾", "钠", "载脂蛋白AⅠ", "铁蛋白",
]

# 43 MIMIC English equivalents (same order)
FEATS_43_EN = [
    "Age", "Apolipoprotein B", "LDL Cholesterol", "Total Cholesterol",
    "Cholinesterase", "Albumin", "Total Bile Acids", "Prealbumin", "D-Dimer",
    "CYFRA 21-1", "Cystatin C", "Triglycerides", "Adenosine Deaminase",
    "Globulin", "Plateletcrit", "SCC Antigen", "Direct Bilirubin",
    "No. of Physicians", "Amylase", "Hospital Beds", "Glucose", "Sex_M",
    "HDL Cholesterol", "Magnesium", "α-HBDH", "ALT", "GDP per Capita",
    "No. of Hospitals", "HBcAb", "LDH", "Phosphorus", "AST", "Uric Acid",
    "Alpha-Fetoprotein", "Total Protein", "HbA1c", "Creatinine",
    "Creatine Kinase", "Avg. Employee Wage", "Potassium", "Sodium",
    "Apolipoprotein AI", "Ferritin",
]

ZH2EN = dict(zip(FEATS_43_ZH, FEATS_43_EN))

# Display names for plots
FEAT_DISPLAY = {
    "Age": "Age", "Apolipoprotein B": "Apolipoprotein B",
    "LDL Cholesterol": "LDL Cholesterol", "Total Cholesterol": "Total Cholesterol",
    "Cholinesterase": "Cholinesterase", "Albumin": "Albumin",
    "Total Bile Acids": "Total Bile Acids", "Prealbumin": "Prealbumin",
    "D-Dimer": "D-Dimer", "CYFRA 21-1": "CYFRA 21-1",
    "Cystatin C": "Cystatin C", "Triglycerides": "Triglycerides",
    "Adenosine Deaminase": "Adenosine Deaminase", "Globulin": "Globulin",
    "Plateletcrit": "Plateletcrit", "SCC Antigen": "SCC Antigen",
    "Direct Bilirubin": "Direct Bilirubin", "No. of Physicians": "No. of Physicians",
    "Amylase": "Amylase", "Hospital Beds": "Hospital Beds",
    "Glucose": "Glucose", "Sex_M": "Sex (Male)",
    "HDL Cholesterol": "HDL Cholesterol", "Magnesium": "Magnesium",
    "α-HBDH": "α-HBDH", "ALT": "ALT", "GDP per Capita": "GDP per Capita",
    "No. of Hospitals": "No. of Hospitals", "HBcAb": "HBcAb",
    "LDH": "LDH", "Phosphorus": "Phosphorus", "AST": "AST",
    "Uric Acid": "Uric Acid", "Alpha-Fetoprotein": "AFP",
    "Total Protein": "Total Protein", "HbA1c": "HbA1c",
    "Creatinine": "Creatinine", "Creatine Kinase": "Creatine Kinase",
    "Avg. Employee Wage": "Avg. Employee Wage", "Potassium": "Potassium",
    "Sodium": "Sodium", "Apolipoprotein AI": "Apolipoprotein A-I",
    "Ferritin": "Ferritin",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_sch_encounter():
    """Load SCH raw encounters with patient-level MACE label."""
    print("Loading SCH encounters...", flush=True)
    labels = pd.read_csv(LABEL_FILE, usecols=["病案号", "Center", "MACE"])
    labels = labels[labels["Center"] == "SCH"][["病案号", "MACE"]].drop_duplicates("病案号")

    cols_all = pd.read_csv(SCH_FILE, nrows=0).columns.tolist()
    usecols = ["病案号"] + [c for c in FEATS_43_ZH if c in cols_all]

    chunks = []
    for chunk in pd.read_csv(SCH_FILE, usecols=usecols, chunksize=100_000,
                              low_memory=False):
        chunks.append(chunk)
    raw = pd.concat(chunks, ignore_index=True)

    # Convert features to numeric
    for c in FEATS_43_ZH:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # Merge patient-level MACE label onto every encounter row
    raw = raw.merge(labels, on="病案号", how="inner")

    # Rename to English
    raw = raw.rename(columns=ZH2EN)
    print(f"  SCH: {len(raw):,} encounters / {raw['病案号'].nunique():,} patients", flush=True)
    print(f"  MACE rate (encounter-level): {raw['MACE'].mean()*100:.2f}%", flush=True)
    return raw


def load_whu_encounter():
    """Load WHU raw encounters with patient-level MACE label."""
    print("Loading WHU encounters...")
    labels = pd.read_csv(LABEL_FILE, usecols=["病案号", "Center", "MACE"])
    labels = labels[labels["Center"] == "WHU"][["病案号", "MACE"]].drop_duplicates("病案号")

    cols_all = pd.read_csv(WHU_FILE, nrows=0).columns.tolist()
    usecols = ["病案号"] + [c for c in FEATS_43_ZH if c in cols_all]
    raw = pd.read_csv(WHU_FILE, usecols=usecols, low_memory=False)

    for c in FEATS_43_ZH:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw = raw.merge(labels, on="病案号", how="inner")
    raw = raw.rename(columns=ZH2EN)
    print(f"  WHU: {len(raw):,} encounters / {raw['病案号'].nunique():,} patients")
    print(f"  MACE rate (encounter-level): {raw['MACE'].mean()*100:.2f}%")
    return raw


def load_mimic():
    """Load MIMIC admission-level features + MACE labels.
    Excludes in-hospital deaths (leakage-safe).
    MIMIC is already admission-level, so no aggregation needed.
    """
    print("Loading MIMIC...", flush=True)
    feat = pd.read_csv(MIMIC_FEAT)
    lbl  = pd.read_csv(MIMIC_LABELS, usecols=["HADM_ID", "MACE"])

    # features_model.csv already has HOSPITAL_EXPIRE_FLAG
    if "HOSPITAL_EXPIRE_FLAG" not in feat.columns:
        lbl2 = pd.read_csv(MIMIC_LABELS, usecols=["HADM_ID", "HOSPITAL_EXPIRE_FLAG", "MACE"])
        df = feat.merge(lbl2, on="HADM_ID", how="inner")
    else:
        df = feat.merge(lbl, on="HADM_ID", how="inner")

    # Exclude in-hospital deaths (label leakage)
    if "HOSPITAL_EXPIRE_FLAG" in df.columns:
        df = df[df["HOSPITAL_EXPIRE_FLAG"] == 0].copy()

    # Keep only available feature columns (English)
    avail_feats = [f for f in FEATS_43_EN if f in df.columns]
    id_cols = [c for c in ["HADM_ID", "SUBJECT_ID"] if c in df.columns]
    df = df[id_cols + avail_feats + ["MACE"]].copy()

    for c in avail_feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n_subj = df["SUBJECT_ID"].nunique() if "SUBJECT_ID" in df.columns else "N/A"
    print(f"  MIMIC: {len(df):,} admissions / {n_subj} patients", flush=True)
    print(f"  MACE rate: {df['MACE'].mean()*100:.2f}%", flush=True)
    return df, avail_feats


def load_mimic_patient():
    """Aggregate MIMIC admissions -> patient-level (median features, any MACE).
    Mirrors v11 strategy for SCH/WHU: median aggregation per patient before modelling.
    """
    print("Loading MIMIC patient-level (median aggregation)...", flush=True)
    feat = pd.read_csv(MIMIC_FEAT)
    lbl  = pd.read_csv(MIMIC_LABELS, usecols=["HADM_ID", "MACE"])

    if "HOSPITAL_EXPIRE_FLAG" not in feat.columns:
        lbl2 = pd.read_csv(MIMIC_LABELS,
                            usecols=["HADM_ID", "HOSPITAL_EXPIRE_FLAG", "MACE"])
        df = feat.merge(lbl2, on="HADM_ID", how="inner")
    else:
        df = feat.merge(lbl, on="HADM_ID", how="inner")

    if "HOSPITAL_EXPIRE_FLAG" in df.columns:
        df = df[df["HOSPITAL_EXPIRE_FLAG"] == 0].copy()

    avail_feats = [f for f in FEATS_43_EN if f in df.columns]
    for c in avail_feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Patient-level: median features, any MACE across admissions
    pat = df.groupby("SUBJECT_ID").agg(
        {**{f: "median" for f in avail_feats}, "MACE": "max"}
    ).reset_index()

    print(f"  MIMIC patient-level: {len(pat):,} patients  "
          f"MACE rate: {pat['MACE'].mean()*100:.2f}%", flush=True)
    return pat, avail_feats


# ═══════════════════════════════════════════════════════════════════════════════
# Ensemble Model
# ═══════════════════════════════════════════════════════════════════════════════

def build_ensemble(pos_weight, fast=False):
    """XGBoost + RandomForest 50:50 ensemble.
    fast=True uses reduced trees for large datasets (SCH 530K rows).
    """
    if fast:
        # Faster settings for SCH encounter-level (530K rows)
        xgb_model = XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.10,
            subsample=0.7, colsample_bytree=0.7,
            scale_pos_weight=pos_weight, eval_metric="auc",
            use_label_encoder=False, verbosity=0,
            random_state=42, n_jobs=-1, tree_method="hist",
        )
        rf_model = RandomForestClassifier(
            n_estimators=150, max_depth=6, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    else:
        # Full settings for WHU / MIMIC
        xgb_model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, eval_metric="auc",
            use_label_encoder=False, verbosity=0,
            random_state=42, n_jobs=-1,
        )
        rf_model = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    return xgb_model, rf_model


def cross_val_encounter(X, y, groups, n_splits=5, fast=False):
    """GroupKFold cross-validation for encounter-level data (SCH/WHU).
    Uses GroupKFold to ensure no patient leaks between folds.
    Returns OOF predicted probabilities and per-fold feature importances.
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y))
    fold_aucs = []
    feat_imp_folds = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr)
        X_va = imp.transform(X_va)

        pos_weight = float((y_tr == 0).sum()) / max(1, (y_tr == 1).sum())
        xgb_m, rf_m = build_ensemble(pos_weight, fast=fast)

        xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        rf_m.fit(X_tr, y_tr)

        p_xgb = xgb_m.predict_proba(X_va)[:, 1]
        p_rf  = rf_m.predict_proba(X_va)[:, 1]
        p_ens = 0.5 * p_xgb + 0.5 * p_rf
        oof[va_idx] = p_ens

        fold_auc = roc_auc_score(y_va, p_ens)
        fold_aucs.append(fold_auc)
        print(f"    Fold {fold}: AUC={fold_auc:.4f}  n_val={len(va_idx):,}", flush=True)

        # Feature importances (XGBoost gain + RF mean impurity)
        fi_xgb = np.array(xgb_m.feature_importances_, dtype=float)
        fi_rf  = np.array(rf_m.feature_importances_, dtype=float)
        n_f = X_tr.shape[1]
        if len(fi_xgb) != n_f:
            fi_xgb = np.zeros(n_f)
        if len(fi_rf) != n_f:
            fi_rf = np.zeros(n_f)
        feat_imp_folds.append(0.5 * fi_xgb + 0.5 * fi_rf)

    try:
        avg_fi = np.vstack(feat_imp_folds).mean(axis=0)
    except Exception:
        avg_fi = np.zeros(X.shape[1])
    return oof, fold_aucs, avg_fi


def compute_fi_quick(source, feat_cols, npy_save_path=None):
    """Train a single quick XGBoost on data for feature-importance purposes only.
    source: str/Path to CSV file, OR a pd.DataFrame already loaded.
    feat_cols: list of feature column names.
    Saves .npy if npy_save_path is given.
    """
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source)
    else:
        df = source.copy()
    # Deduplicate to avoid double-counting identical column names
    avail = list(dict.fromkeys([c for c in feat_cols if c in df.columns]))
    if not avail:
        return np.zeros(len(feat_cols))
    X = df[avail].values.astype(float)
    y = df["MACE"].values.astype(int)

    # impute
    from sklearn.impute import SimpleImputer as _SI
    X = _SI(strategy="median").fit_transform(X)

    pos_weight = float((y == 0).sum()) / max(1, (y == 1).sum())
    xgb_m = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight, eval_metric="auc",
        use_label_encoder=False, verbosity=0,
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    rf_m = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    xgb_m.fit(X, y)
    rf_m.fit(X, y)
    fi = 0.5 * np.array(xgb_m.feature_importances_, dtype=float) \
       + 0.5 * np.array(rf_m.feature_importances_, dtype=float)

    # Pad back to full feat_cols length (some may have been missing)
    fi_full = np.zeros(len(feat_cols))
    for k, col in enumerate(feat_cols):
        if col in avail:
            idx = avail.index(col)
            if idx < len(fi):
                fi_full[k] = fi[idx]

    if npy_save_path is not None:
        np.save(npy_save_path, fi_full)
        print(f"    Saved FI → {Path(npy_save_path).name}", flush=True)
    return fi_full


def cross_val_stratified(X, y, n_splits=5):
    """StratifiedKFold for MIMIC (admissions are independent)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    fold_aucs = []
    feat_imp_folds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr)
        X_va = imp.transform(X_va)

        pos_weight = float((y_tr == 0).sum()) / max(1, (y_tr == 1).sum())
        xgb_m, rf_m = build_ensemble(pos_weight)

        xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        rf_m.fit(X_tr, y_tr)

        p_xgb = xgb_m.predict_proba(X_va)[:, 1]
        p_rf  = rf_m.predict_proba(X_va)[:, 1]
        p_ens = 0.5 * p_xgb + 0.5 * p_rf
        oof[va_idx] = p_ens

        fold_auc = roc_auc_score(y_va, p_ens)
        fold_aucs.append(fold_auc)
        print(f"    Fold {fold}: AUC={fold_auc:.4f}  n_val={len(va_idx):,}", flush=True)

        fi_xgb = np.array(xgb_m.feature_importances_, dtype=float)
        fi_rf  = np.array(rf_m.feature_importances_, dtype=float)
        n_f = X_tr.shape[1]
        if len(fi_xgb) != n_f:
            fi_xgb = np.zeros(n_f)
        if len(fi_rf) != n_f:
            fi_rf = np.zeros(n_f)
        feat_imp_folds.append(0.5 * fi_xgb + 0.5 * fi_rf)

    try:
        avg_fi = np.vstack(feat_imp_folds).mean(axis=0)
    except Exception:
        avg_fi = np.zeros(X.shape[1])
    return oof, fold_aucs, avg_fi


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregate predictions to patient level (for KM + comparison with v11)
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_to_patient(df_enc, prob_col="AdmPredProb"):
    """Aggregate encounter-level predictions to patient level.
    Uses MAX predicted probability per patient (most alarming visit).
    """
    pt = df_enc.groupby("病案号").agg(
        MACE=("MACE", "max"),
        AdmPredProb_max=("AdmPredProb", "max"),
        AdmPredProb_mean=("AdmPredProb", "mean"),
        AdmPredProb_last=("AdmPredProb", "last"),
        n_encounters=("AdmPredProb", "count"),
    ).reset_index()
    return pt


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════

def net_benefit(y_true, y_prob, thresholds):
    n = len(y_true)
    nb = []
    for t in thresholds:
        pp = y_prob >= t
        tp = np.sum(pp & (y_true == 1))
        fp = np.sum(pp & (y_true == 0))
        nb.append(tp / n - fp / n * t / max(1e-9, 1 - t))
    return np.array(nb)


def nb_all(y_true, thresholds):
    prev = y_true.mean()
    return np.array([prev - (1 - prev) * t / max(1e-9, 1 - t)
                     for t in thresholds])


def roc_ci_band(y_true, y_prob, n_boot=400, seed=2026):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(seed)

    fpr_grid = np.linspace(0, 1, 201)
    tpr_boot = []
    auc_boot = []
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        pb = y_prob[idx]
        if np.unique(yb).size < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(yb, pb)
        tpr_interp = np.interp(fpr_grid, fpr_b, tpr_b)
        tpr_interp[0] = 0.0
        tpr_interp[-1] = 1.0
        tpr_boot.append(tpr_interp)
        auc_boot.append(roc_auc_score(yb, pb))

    if len(tpr_boot) < 20:
        return fpr_grid, None, None, None, None

    tpr_boot = np.vstack(tpr_boot)
    low = np.percentile(tpr_boot, 2.5, axis=0)
    high = np.percentile(tpr_boot, 97.5, axis=0)
    auc_lo, auc_hi = np.percentile(np.asarray(auc_boot), [2.5, 97.5])
    return fpr_grid, low, high, float(auc_lo), float(auc_hi)


def add_panel_label(ax, label, fontsize=8.5):
    """Lancet-style panel label: bold Arial, outside the axes frame, top-left."""
    ax.text(-0.08, 1.06, label, transform=ax.transAxes,
        ha="left", va="bottom", fontsize=fontsize,
        fontweight="bold", fontname="Arial",
        clip_on=False)


def fig1_roc(results_dict):
    """Figure 1: ROC curves for SCH, WHU, MIMIC — admission-level OOF."""
    fig, axes = plt.subplots(1, 3, figsize=(170 / 25.4, 72 / 25.4))
    colors = {"SCH": C_SCH, "WHU": C_WHU, "MIMIC": C_MIMC}

    for i, (ax, (name, res)) in enumerate(zip(axes, results_dict.items())):
        fpr, tpr, _ = roc_curve(res["y"], res["oof"])
        auc = roc_auc_score(res["y"], res["oof"])
        fpr_grid, tpr_lo, tpr_hi, auc_lo, auc_hi = roc_ci_band(res["y"], res["oof"], n_boot=400, seed=2026)

        if tpr_lo is not None and tpr_hi is not None:
            ax.fill_between(fpr_grid, tpr_lo, tpr_hi, color=colors[name], alpha=0.16,
                label="95% CI")

        auc_txt = f"AUC={auc:.3f}" if auc_lo is None else f"AUC={auc:.3f} ({auc_lo:.3f}-{auc_hi:.3f})"
        ax.plot(fpr, tpr, color=colors[name], lw=1.8,
                label=f"Admission model\n{auc_txt}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("1 − Specificity", fontsize=8)
        ax.set_ylabel("Sensitivity", fontsize=8)
        ax.set_title(name, fontsize=8, fontweight="bold", pad=4)
        ax.grid(False)
        ax.legend(loc="lower right", fontsize=6, frameon=True)
        add_panel_label(ax, chr(ord("A") + i), fontsize=8)
        ax.set_box_aspect(1)

    save_figure(fig, "Figure1_ROC_Comparison", rect=[0, 0.02, 1, 0.94])
    print("  Saved Figure1_ROC_Comparison")


def fig2_dca(results_dict):
    """Figure 2: DCA — admission-level vs v11 patient-level."""
    thr = np.linspace(0.001, 0.6, 300)
    fig, axes = plt.subplots(1, 3, figsize=(170 / 25.4, 62 / 25.4))
    titles = {"SCH": "SCH", "WHU": "WHU", "MIMIC": "MIMIC"}
    colors = {"SCH": C_SCH, "WHU": C_WHU, "MIMIC": C_MIMC}

    for i, (ax, (name, res)) in enumerate(zip(axes, results_dict.items())):
        y, oof = res["y"], res["oof"]
        nb_model = net_benefit(y, oof, thr)
        nb_treat_all = nb_all(y, thr)
        nb_treat_none = np.zeros_like(thr)

        ax.plot(thr, nb_model, color=colors[name], lw=1.8, label="Admission model")
        ax.plot(thr, nb_treat_all, color="#888888", lw=1.2, ls="--", label="Treat all")
        ax.plot(thr, nb_treat_none, color="#BBBBBB", lw=1.0, ls=":", label="Treat none")

        # v11 patient-level DCA
        if "v11_prob" in res and "v11_y" in res:
            nb_v11 = net_benefit(res["v11_y"], res["v11_prob"], thr)
            ax.plot(thr, nb_v11, color=colors[name], lw=1.2, ls="-.",
                    alpha=0.6, label="v11 patient-level")

        ax.set_xlabel("Threshold probability", fontsize=8)
        ax.set_ylabel("Net benefit", fontsize=8)
        ax.set_title(f"{titles[name]}", fontsize=8, fontweight="bold", pad=4)
        ax.set_xlim([0, 0.6])
        ax.set_ylim([-0.02, y.mean() * 1.2])
        ax.legend(loc="upper right", fontsize=6, frameon=True)
        add_panel_label(ax, chr(ord("A") + i), fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in (".pdf", ".png"):
        plt.savefig(OUT_DIR / f"Figure2_DCA{ext}", dpi=600,
                    bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved Figure2_DCA")


def fig3_calibration(results_dict):
    """Figure 3: Calibration curves for all three cohorts."""
    fig, axes = plt.subplots(1, 3, figsize=(170 / 25.4, 62 / 25.4))
    titles = {"SCH": "SCH", "WHU": "WHU", "MIMIC": "MIMIC"}
    colors = {"SCH": C_SCH, "WHU": C_WHU, "MIMIC": C_MIMC}

    for i, (ax, (name, res)) in enumerate(zip(axes, results_dict.items())):
        y, oof = res["y"], res["oof"]
        n_bins = 10
        fraction_pos, mean_pred = calibration_curve(y, oof, n_bins=n_bins,
                                                     strategy="quantile")
        ax.plot(mean_pred, fraction_pos, "o-", color=colors[name], lw=1.8,
                markersize=4, label="Admission model")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect")

        # ECE
        bin_edges = np.percentile(oof, np.linspace(0, 100, n_bins + 1))
        bins = np.searchsorted(bin_edges[1:-1], oof)
        ece = sum(
            abs(fraction_pos[i] - mean_pred[i]) * (bins == i).sum() / len(y)
            for i in range(len(fraction_pos))
        )
        ax.text(0.04, 0.88, f"ECE={ece:.3f}", transform=ax.transAxes,
                fontsize=7, bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                     ec="#cccccc", lw=0.7))

        ax.set_xlabel("Mean predicted probability", fontsize=8)
        ax.set_ylabel("Fraction of positives", fontsize=8)
        ax.set_title(f"{titles[name]}", fontsize=8, fontweight="bold", pad=4)
        ax.legend(loc="lower right", fontsize=6, frameon=True)
        add_panel_label(ax, chr(ord("A") + i), fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in (".pdf", ".png"):
        plt.savefig(OUT_DIR / f"Figure3_Calibration{ext}", dpi=600,
                    bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved Figure3_Calibration")


def fig4_feature_importance(fi_dict):
    """Figure 4: Top-15 feature importances for each cohort."""
    fig, axes = plt.subplots(1, 3, figsize=(210 / 25.4, 90 / 25.4))
    titles = {"SCH": "SCH", "WHU": "WHU", "MIMIC": "MIMIC"}
    colors = {"SCH": C_SCH, "WHU": C_WHU, "MIMIC": C_MIMC}

    for i, (ax, (name, (feat_names, importances))) in enumerate(zip(axes, fi_dict.items())):
        # Normalize
        imp_norm = importances / (importances.sum() + 1e-12)
        sort_idx = np.argsort(imp_norm)[-15:]
        top_feats = [FEAT_DISPLAY.get(feat_names[i], feat_names[i])
                     for i in sort_idx]
        top_imps  = imp_norm[sort_idx]

        ax.barh(range(15), top_imps, color=colors[name], height=0.65, alpha=0.85)
        ax.set_yticks(range(15))
        ax.set_yticklabels(top_feats, fontsize=6.0)
        ax.set_xlabel("Relative Importance", fontsize=7.5)
        ax.set_title(f"{titles[name]}", fontsize=8, fontweight="bold", pad=4)
        add_panel_label(ax, chr(ord("A") + i), fontsize=8)
        ax.grid(axis="x", color="#E8E8E8", lw=0.5)
        ax.grid(axis="y", visible=False)
        ax.set_box_aspect(1)

    save_figure(fig, "Figure4_Feature_Importance", rect=[0, 0.02, 1, 0.94])
    print("  Saved Figure4_Feature_Importance")


def fig5_comparison_bar(summary_df):
    """Figure 5: Side-by-side AUC comparison — admission vs v11 patient-level."""
    cohorts = ["SCH", "WHU", "MIMIC"]
    x = np.arange(len(cohorts))
    w = 0.35

    adm_auc  = [summary_df.loc[summary_df.Cohort == c, "AUC_admission"].values[0]
                for c in cohorts]
    pt_auc   = [summary_df.loc[summary_df.Cohort == c, "AUC_patient_v11"].values[0]
                for c in cohorts]
    adm_bri  = [summary_df.loc[summary_df.Cohort == c, "Brier_admission"].values[0]
                for c in cohorts]
    pt_bri   = [summary_df.loc[summary_df.Cohort == c, "Brier_patient_v11"].values[0]
                for c in cohorts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160 / 25.4, 72 / 25.4))

    # AUC
    b1 = ax1.bar(x - w / 2, adm_auc, w, label="Admission-level",
                 color=["#2166AC", "#D6604D", "#4DAC26"], alpha=0.85, edgecolor="white")
    b2 = ax1.bar(x + w / 2, pt_auc,  w, label="Patient-level (median agg)",
                 color=["#2166AC", "#D6604D", "#4DAC26"], alpha=0.45,
                 edgecolor="white", hatch="///")
    for bar, val in zip(b1, adm_auc):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.003, f"{val:.3f}",
                 ha="center", fontsize=6.5, fontweight="bold")
    for bar, val in zip(b2, pt_auc):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.003, f"{val:.3f}",
                 ha="center", fontsize=6.5, color="#444")
    ax1.set_xticks(x); ax1.set_xticklabels(cohorts)
    ax1.set_ylabel("AUC", fontsize=8); ax1.set_ylim([0.5, 1.0])
    ax1.grid(axis="y", color="#E8E8E8", lw=0.5)
    ax1.grid(axis="x", visible=False)
    ax1.legend(loc="lower right", fontsize=6.5, frameon=True)
    add_panel_label(ax1, "A", fontsize=8)

    # Brier score
    b3 = ax2.bar(x - w / 2, adm_bri, w, label="Admission-level",
                 color=["#2166AC", "#D6604D", "#4DAC26"], alpha=0.85, edgecolor="white")
    b4 = ax2.bar(x + w / 2, pt_bri,  w, label="Patient-level (median agg)",
                 color=["#2166AC", "#D6604D", "#4DAC26"], alpha=0.45,
                 edgecolor="white", hatch="///")
    for bar, val in zip(b3, adm_bri):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.002, f"{val:.3f}",
                 ha="center", fontsize=6.5, fontweight="bold")
    for bar, val in zip(b4, pt_bri):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.002, f"{val:.3f}",
                 ha="center", fontsize=6.5, color="#444")
    ax2.set_xticks(x); ax2.set_xticklabels(cohorts)
    ax2.set_ylabel("Brier Score (lower = better)", fontsize=8)
    ax2.grid(axis="y", color="#E8E8E8", lw=0.5)
    ax2.grid(axis="x", visible=False)
    ax2.legend(loc="lower right", fontsize=6.5, frameon=True)
    add_panel_label(ax2, "B", fontsize=8)

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)

    save_figure(fig, "Figure5_Admission_vs_Patient", rect=[0, 0.02, 1, 0.94])
    print("  Saved Figure5_Admission_vs_Patient")


def fig6_temporal_gradient(pred_files):
    """Figure 6: Temporal risk gradient — mean predicted probability by
    relative admission position (0=first, 1=last), stratified by MACE.
    Shows whether the model captures a rising-risk pattern over time.
    pred_files: dict {cohort: (pred_csv_path, patient_id_col)}
    Only patients with >=2 admissions are plotted.
    """
    fig, axes = plt.subplots(1, 3, figsize=(210 / 25.4, 84 / 25.4))
    titles   = {"SCH": "SCH", "WHU": "WHU", "MIMIC": "MIMIC"}
    colors   = {"SCH": C_SCH, "WHU": C_WHU, "MIMIC": C_MIMC}
    N_BINS   = 10

    for i, (ax, (name, (csv_path, id_col))) in enumerate(zip(axes, pred_files.items())):
        cols = pd.read_csv(csv_path, nrows=0).columns
        if id_col not in cols:
            ax.text(0.5, 0.5, "No patient-ID column", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            continue

        req_cols = [id_col, "MACE", "AdmPredProb"]
        miss_cols = [c for c in req_cols if c not in cols]
        if miss_cols:
            ax.text(0.5, 0.5, f"Missing columns: {', '.join(miss_cols)}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            continue

        df = pd.read_csv(csv_path, usecols=req_cols, low_memory=False)

        # Keep only patients with ≥2 admissions
        df = df.copy()
        cnt = df.groupby(id_col)[id_col].transform("count")
        df = df[cnt >= 2].reset_index(drop=True)

        # Visit rank within patient (0 = first, max = last)
        df["_rank"] = df.groupby(id_col).cumcount()
        df["_maxr"] = df.groupby(id_col)["_rank"].transform("max")
        df["_relpos"] = df["_rank"] / df["_maxr"]  # 0.0 – 1.0

        # Bin relative position into N_BINS equal-width bins
        labels = [f"{i/N_BINS:.1f}–{(i+1)/N_BINS:.1f}" for i in range(N_BINS)]
        df["_bin"] = pd.cut(df["_relpos"], bins=N_BINS,
                            labels=labels, include_lowest=True)

        bin_centers = [(i + 0.5) / N_BINS for i in range(N_BINS)]
        line_end = {}

        for mace_val, label, ls, marker in [
            (1, "MACE=1", "-",  "o"),
            (0, "MACE=0", "--", "s"),
        ]:
            sub = df[df["MACE"] == mace_val]
            grp = sub.groupby("_bin", observed=True)["AdmPredProb"]
            means = grp.mean().values
            sems  = grp.sem().values

            c = colors[name]
            alpha_fill = 0.18 if mace_val == 1 else 0.10
            lw = 1.8 if mace_val == 1 else 1.2
            ax.plot(bin_centers[:len(means)], means,
                    color=c, lw=lw, ls=ls, marker=marker,
                    markersize=4, label=f"{label} (n={sub[id_col].nunique():,})")
            ax.fill_between(bin_centers[:len(means)],
                            means - sems, means + sems,
                            color=c, alpha=alpha_fill)
            if len(means) > 0:
                line_end[label] = float(means[-1])

        ax.set_xlabel("Relative admission position\n(0=first, 1=last)",
                      fontsize=7.5)
        ax.set_ylabel("Mean predicted probability", fontsize=7.5)
        ax.set_title(titles[name], fontsize=8, fontweight="bold", pad=4)
        ax.set_xlim([-0.02, 1.02])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        if "MACE=1" in line_end:
            ax.text(0.83, line_end["MACE=1"], "MACE=1", fontsize=6,
                color=colors[name], ha="left", va="center",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.12))
        if "MACE=0" in line_end:
            ax.text(0.83, line_end["MACE=0"], "MACE=0", fontsize=6,
                color=colors[name], ha="left", va="center",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.12))
        ax.grid(color="#E8E8E8", lw=0.5)
        n_multi = (df.groupby(id_col)["_rank"].max() > 0).sum()
        ax.text(0.98, 0.04, f"Multi-visit pts: {n_multi:,}",
                transform=ax.transAxes, fontsize=6, ha="right",
                color="#666666")
        ax.legend(loc="upper left", fontsize=6, frameon=True)
        add_panel_label(ax, chr(ord("A") + i), fontsize=8)
        ax.set_box_aspect(1)

    save_figure(fig, "Figure6_Temporal_Gradient", rect=[0, 0.02, 1, 0.93])
    print("  Saved Figure6_Temporal_Gradient")


def plot_only_from_checkpoints():
    print("\n[PLOT-ONLY] Loading checkpoints and regenerating figures...")
    v11 = load_v11_preds()

    sch_df = pd.read_csv(OUT_DIR / "predictions_admission_SCH.csv")
    whu_df = pd.read_csv(OUT_DIR / "predictions_admission_WHU.csv")
    mim_df = pd.read_csv(OUT_DIR / "predictions_admission_MIMIC.csv")
    mim_pat_df = pd.read_csv(OUT_DIR / "predictions_patient_MIMIC.csv")
    summary_df = pd.read_csv(OUT_DIR / "summary_metrics.csv")

    results = {
        "SCH": {
            "y": sch_df["MACE"].values.astype(int),
            "oof": sch_df["AdmPredProb"].values,
            "v11_auc": v11["SCH"]["auc"],
            "v11_prob": v11["SCH"]["prob"],
            "v11_y": v11["SCH"]["y"],
        },
        "WHU": {
            "y": whu_df["MACE"].values.astype(int),
            "oof": whu_df["AdmPredProb"].values,
            "v11_auc": v11["WHU"]["auc"],
            "v11_prob": v11["WHU"]["prob"],
            "v11_y": v11["WHU"]["y"],
        },
        "MIMIC": {
            "y": mim_df["MACE"].values.astype(int),
            "oof": mim_df["AdmPredProb"].values,
            "v11_auc": roc_auc_score(mim_pat_df["MACE"].values.astype(int), mim_pat_df["PatPredProb"].values),
            "v11_prob": mim_pat_df["PatPredProb"].values,
            "v11_y": mim_pat_df["MACE"].values.astype(int),
        },
    }

    fi_sch = np.load(OUT_DIR / "_fi_sch.npy") if (OUT_DIR / "_fi_sch.npy").exists() else np.zeros(len(FEATS_43_EN))
    fi_whu = np.load(OUT_DIR / "_fi_whu.npy") if (OUT_DIR / "_fi_whu.npy").exists() else np.zeros(len(FEATS_43_EN))
    fi_mim = np.load(OUT_DIR / "_fi_mimic_pat.npy") if (OUT_DIR / "_fi_mimic_pat.npy").exists() else np.zeros(len(FEATS_43_EN))
    fi_store = {
        "SCH": (FEATS_43_EN, fi_sch),
        "WHU": (FEATS_43_EN, fi_whu),
        "MIMIC": (FEATS_43_EN, fi_mim),
    }

    fig1_roc(results)
    fig2_dca(results)
    fig3_calibration(results)
    fig4_feature_importance(fi_store)
    fig5_comparison_bar(summary_df)
    fig6_temporal_gradient({
        "SCH":   (OUT_DIR / "predictions_admission_SCH.csv",   "病案号"),
        "WHU":   (OUT_DIR / "predictions_admission_WHU.csv",   "病案号"),
        "MIMIC": (OUT_DIR / "predictions_admission_MIMIC.csv", "SUBJECT_ID"),
    })
    print("[PLOT-ONLY] Figure regeneration complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# v11 baseline predictions (patient-level)
# ═══════════════════════════════════════════════════════════════════════════════

def load_v11_preds():
    preds = pd.read_csv(LABEL_FILE)
    v11 = {}
    for center in ["SCH", "WHU"]:
        sub = preds[preds["Center"] == center].copy()
        v11[center] = {
            "y":    sub["MACE"].values.astype(int),
            "prob": sub["predicted_prob_within_cv"].values,
            "auc":  roc_auc_score(sub["MACE"].values,
                                   sub["predicted_prob_within_cv"].values),
            "brier": brier_score_loss(sub["MACE"].values,
                                       sub["predicted_prob_within_cv"].values),
        }
        print(f"  v11 {center}: AUC={v11[center]['auc']:.4f}  "
              f"Brier={v11[center]['brier']:.4f}  n={len(sub):,}")
    return v11


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Admission-Level MACE Prediction — Parallel to v11")
    print("=" * 70)

    # ── Load v11 baselines ────────────────────────────────────────────────────
    print("\n[0] Loading v11 patient-level baselines...")
    v11 = load_v11_preds()

    results   = {}  # store {cohort: {y, oof, v11_auc, ...}}
    fi_store  = {}  # store {cohort: (feat_names, importances)}
    all_preds = []  # for CSV export
    metrics   = []  # for summary table

    # ── SCH ───────────────────────────────────────────────────────────────────
    print("\n[1] SCH — encounter-level (GroupKFold by 病案号)...", flush=True)
    sch_pred_file = OUT_DIR / "predictions_admission_SCH.csv"
    fi_sch_file   = OUT_DIR / "_fi_sch.npy"

    if sch_pred_file.exists():
        print("  [CHECKPOINT] Loading saved SCH predictions...", flush=True)
        df_sch_saved = pd.read_csv(sch_pred_file)
        y_sch   = df_sch_saved["MACE"].values.astype(int)
        oof_sch = df_sch_saved["AdmPredProb"].values
        fi_sch  = np.load(fi_sch_file) if fi_sch_file.exists() else np.zeros(len(FEATS_43_EN))
        if fi_sch.sum() == 0:
            print("  Recomputing SCH FI from checkpoint CSV...", flush=True)
            fi_sch = compute_fi_quick(sch_pred_file, FEATS_43_EN, fi_sch_file)
        sch_feat_cols = FEATS_43_EN  # use standard list
        n_sch_enc = len(df_sch_saved)
        n_sch_pat = df_sch_saved["病案号"].nunique() if "病案号" in df_sch_saved.columns else 38509
        fold_aucs_sch = []  # unknown from checkpoint
    else:
        sch = load_sch_encounter()
        sch_feat_cols = [c for c in FEATS_43_EN if c in sch.columns]
        X_sch = sch[sch_feat_cols].values.astype(float)
        y_sch = sch["MACE"].values.astype(int)
        g_sch = sch["病案号"].values
        n_sch_enc = len(sch)
        n_sch_pat = sch["病案号"].nunique()

        oof_sch, fold_aucs_sch, fi_sch = cross_val_encounter(X_sch, y_sch, g_sch,
                                                              fast=True)
        # Save predictions & feature importances checkpoint
        df_sch_pred = sch[["病案号"] + sch_feat_cols + ["MACE"]].copy()
        df_sch_pred["AdmPredProb"] = oof_sch
        df_sch_pred["Center"] = "SCH"
        df_sch_pred[["病案号", "Center", "MACE", "AdmPredProb"]].to_csv(
            sch_pred_file, index=False)
        np.save(fi_sch_file, fi_sch)
        print("  Saved predictions_admission_SCH.csv", flush=True)

    auc_sch = roc_auc_score(y_sch, oof_sch)
    bri_sch = brier_score_loss(y_sch, oof_sch)
    prc_sch = average_precision_score(y_sch, oof_sch)
    print(f"  SCH encounter AUC={auc_sch:.4f}  Brier={bri_sch:.4f}  "
          f"PRAUC={prc_sch:.4f}", flush=True)
    print(f"  v11 patient AUC={v11['SCH']['auc']:.4f}  "
          f"ΔAUC={auc_sch - v11['SCH']['auc']:+.4f}", flush=True)

    # Save predictions (skip if checkpoint already did it)
    if not sch_pred_file.exists():
        df_sch_pred2 = pd.DataFrame({"病案号": np.nan, "Center": "SCH",
                                     "MACE": y_sch, "AdmPredProb": oof_sch})
        df_sch_pred2.to_csv(sch_pred_file, index=False)
        print("  Saved predictions_admission_SCH.csv", flush=True)
    if not fi_sch_file.exists():
        np.save(fi_sch_file, fi_sch)

    results["SCH"] = {
        "y": y_sch, "oof": oof_sch,
        "v11_auc": v11["SCH"]["auc"],
        "v11_prob": v11["SCH"]["prob"],
        "v11_y":   v11["SCH"]["y"],
    }
    fi_store["SCH"] = (sch_feat_cols, fi_sch)
    metrics.append({
        "Cohort": "SCH",
        "Modeling_Level": "Admission (encounter-level)",
        "N_rows": len(y_sch),
        "N_patients": n_sch_pat,
        "MACE_rate_pct": y_sch.mean() * 100,
        "AUC_admission": auc_sch,
        "Brier_admission": bri_sch,
        "PRAUC_admission": prc_sch,
        "Fold_AUCs": str([f"{a:.4f}" for a in fold_aucs_sch]) if fold_aucs_sch else "from checkpoint",
        "AUC_patient_v11": v11["SCH"]["auc"],
        "Brier_patient_v11": v11["SCH"]["brier"],
        "Delta_AUC": auc_sch - v11["SCH"]["auc"],
        "N_features": len(sch_feat_cols),
        "CV_method": "GroupKFold(5) by 病案号",
    })

    # ── WHU ───────────────────────────────────────────────────────────────────
    print("\n[2] WHU — encounter-level (GroupKFold by 病案号)...", flush=True)
    whu_pred_file = OUT_DIR / "predictions_admission_WHU.csv"
    fi_whu_file   = OUT_DIR / "_fi_whu.npy"

    if whu_pred_file.exists():
        print("  [CHECKPOINT] Loading saved WHU predictions...", flush=True)
        df_whu_saved = pd.read_csv(whu_pred_file)
        y_whu   = df_whu_saved["MACE"].values.astype(int)
        oof_whu = df_whu_saved["AdmPredProb"].values
        fi_whu  = np.load(fi_whu_file) if fi_whu_file.exists() else np.zeros(len(FEATS_43_EN))
        if fi_whu.sum() == 0:
            print("  Recomputing WHU FI from checkpoint CSV...", flush=True)
            fi_whu = compute_fi_quick(whu_pred_file, FEATS_43_EN, fi_whu_file)
        whu_feat_cols = FEATS_43_EN
        n_whu_enc = len(df_whu_saved)
        n_whu_pat = df_whu_saved["病案号"].nunique() if "病案号" in df_whu_saved.columns else 2562
        fold_aucs_whu = []
    else:
        whu = load_whu_encounter()
        whu_feat_cols = [c for c in FEATS_43_EN if c in whu.columns]
        X_whu = whu[whu_feat_cols].values.astype(float)
        y_whu = whu["MACE"].values.astype(int)
        g_whu = whu["病案号"].values
        n_whu_enc = len(whu)
        n_whu_pat = whu["病案号"].nunique()

        oof_whu, fold_aucs_whu, fi_whu = cross_val_encounter(X_whu, y_whu, g_whu)

        df_whu_pred = whu[["病案号"] + whu_feat_cols + ["MACE"]].copy()
        df_whu_pred["AdmPredProb"] = oof_whu
        df_whu_pred["Center"] = "WHU"
        df_whu_pred[["病案号", "Center", "MACE", "AdmPredProb"]].to_csv(
            whu_pred_file, index=False)
        np.save(fi_whu_file, fi_whu)
        print("  Saved predictions_admission_WHU.csv", flush=True)

    auc_whu = roc_auc_score(y_whu, oof_whu)
    bri_whu = brier_score_loss(y_whu, oof_whu)
    prc_whu = average_precision_score(y_whu, oof_whu)
    print(f"  WHU encounter AUC={auc_whu:.4f}  Brier={bri_whu:.4f}  "
          f"PRAUC={prc_whu:.4f}", flush=True)
    print(f"  v11 patient AUC={v11['WHU']['auc']:.4f}  "
          f"ΔAUC={auc_whu - v11['WHU']['auc']:+.4f}", flush=True)

    results["WHU"] = {
        "y": y_whu, "oof": oof_whu,
        "v11_auc": v11["WHU"]["auc"],
        "v11_prob": v11["WHU"]["prob"],
        "v11_y":   v11["WHU"]["y"],
    }
    fi_store["WHU"] = (whu_feat_cols, fi_whu)
    metrics.append({
        "Cohort": "WHU",
        "Modeling_Level": "Admission (encounter-level)",
        "N_rows": len(y_whu),
        "N_patients": n_whu_pat,
        "MACE_rate_pct": y_whu.mean() * 100,
        "AUC_admission": auc_whu,
        "Brier_admission": bri_whu,
        "PRAUC_admission": prc_whu,
        "Fold_AUCs": str([f"{a:.4f}" for a in fold_aucs_whu]) if fold_aucs_whu else "from checkpoint",
        "AUC_patient_v11": v11["WHU"]["auc"],
        "Brier_patient_v11": v11["WHU"]["brier"],
        "Delta_AUC": auc_whu - v11["WHU"]["auc"],
        "N_features": len(whu_feat_cols),
        "CV_method": "GroupKFold(5) by 病案号",
    })

    # ── MIMIC ─────────────────────────────────────────────────────────────────
    print("\n[3] MIMIC — admission-level (StratifiedKFold by HADM_ID)...", flush=True)
    mimic_adm_file = OUT_DIR / "predictions_admission_MIMIC.csv"

    if mimic_adm_file.exists():
        print("  [CHECKPOINT] Loading saved MIMIC admission predictions...", flush=True)
        df_mimic_saved = pd.read_csv(mimic_adm_file)
        y_mimic   = df_mimic_saved["MACE"].values.astype(int)
        oof_mimic = df_mimic_saved["AdmPredProb"].values
        mimic_feats = FEATS_43_EN
        fold_aucs_mimic = []
        fi_mimic = np.zeros(len(FEATS_43_EN))
        n_mimic_pat = df_mimic_saved["SUBJECT_ID"].nunique() \
            if "SUBJECT_ID" in df_mimic_saved.columns else 6783
    else:
        mimic_df, mimic_feats = load_mimic()
        X_mimic = mimic_df[mimic_feats].values.astype(float)
        y_mimic = mimic_df["MACE"].values.astype(int)
        n_mimic_pat = mimic_df["SUBJECT_ID"].nunique()

        oof_mimic, fold_aucs_mimic, fi_mimic = cross_val_stratified(X_mimic, y_mimic)

        df_mimic_pred = mimic_df[["HADM_ID", "SUBJECT_ID", "MACE"]].copy()
        df_mimic_pred["AdmPredProb"] = oof_mimic
        df_mimic_pred["Center"] = "MIMIC"
        df_mimic_pred.to_csv(mimic_adm_file, index=False)
        print("  Saved predictions_admission_MIMIC.csv", flush=True)

    auc_mimic = roc_auc_score(y_mimic, oof_mimic)
    bri_mimic = brier_score_loss(y_mimic, oof_mimic)
    prc_mimic = average_precision_score(y_mimic, oof_mimic)
    print(f"  MIMIC admission AUC={auc_mimic:.4f}  Brier={bri_mimic:.4f}  "
          f"PRAUC={prc_mimic:.4f}", flush=True)

    results["MIMIC"] = {"y": y_mimic, "oof": oof_mimic}
    # fi_store["MIMIC"] will be set after patient-level model below

    # ── MIMIC patient-level (median aggregation — mirrors v11 strategy) ───────
    print("\n[3b] MIMIC — patient-level (median aggregation, StratifiedKFold)...",
          flush=True)
    mimic_pat_file     = OUT_DIR / "predictions_patient_MIMIC.csv"
    fi_mimic_pat_file  = OUT_DIR / "_fi_mimic_pat.npy"

    if mimic_pat_file.exists():
        print("  [CHECKPOINT] Loading saved MIMIC patient predictions...", flush=True)
        df_mimic_pat_saved = pd.read_csv(mimic_pat_file)
        y_mimic_pat   = df_mimic_pat_saved["MACE"].values.astype(int)
        oof_mimic_pat = df_mimic_pat_saved["PatPredProb"].values
        mimic_pat_feats = FEATS_43_EN
        fold_aucs_mimic_pat = []
        fi_mimic_pat = np.load(fi_mimic_pat_file) \
            if fi_mimic_pat_file.exists() else np.zeros(len(FEATS_43_EN))
        if fi_mimic_pat.sum() == 0:
            print("  Recomputing MIMIC FI (loading patient-level data)...", flush=True)
            _mp_df, _mp_feats = load_mimic_patient()
            fi_mimic_pat = compute_fi_quick(_mp_df, FEATS_43_EN, fi_mimic_pat_file)
    else:
        mimic_pat_df, mimic_pat_feats = load_mimic_patient()
        X_mimic_pat = mimic_pat_df[mimic_pat_feats].values.astype(float)
        y_mimic_pat = mimic_pat_df["MACE"].values.astype(int)

        oof_mimic_pat, fold_aucs_mimic_pat, fi_mimic_pat = cross_val_stratified(
            X_mimic_pat, y_mimic_pat)

        df_mimic_pat_pred = mimic_pat_df[["SUBJECT_ID", "MACE"]].copy()
        df_mimic_pat_pred["PatPredProb"] = oof_mimic_pat
        df_mimic_pat_pred.to_csv(mimic_pat_file, index=False)
        np.save(fi_mimic_pat_file, fi_mimic_pat)
        print("  Saved predictions_patient_MIMIC.csv", flush=True)

    v11_mimic_auc = roc_auc_score(y_mimic_pat, oof_mimic_pat)
    v11_mimic_bri = brier_score_loss(y_mimic_pat, oof_mimic_pat)
    v11_mimic_prc = average_precision_score(y_mimic_pat, oof_mimic_pat)
    print(f"  MIMIC patient AUC={v11_mimic_auc:.4f}  Brier={v11_mimic_bri:.4f}  "
          f"PRAUC={v11_mimic_prc:.4f}", flush=True)
    print(f"  Admission AUC={auc_mimic:.4f}  "
          f"ΔAUC={auc_mimic - v11_mimic_auc:+.4f}", flush=True)

    results["MIMIC"]["v11_auc"]  = v11_mimic_auc
    results["MIMIC"]["v11_prob"] = oof_mimic_pat
    results["MIMIC"]["v11_y"]   = y_mimic_pat
    fi_store["MIMIC"] = (mimic_pat_feats, fi_mimic_pat)

    metrics.append({
        "Cohort": "MIMIC",
        "Modeling_Level": "Admission (encounter-level)",
        "N_rows": len(y_mimic),
        "N_patients": n_mimic_pat,
        "MACE_rate_pct": y_mimic.mean() * 100,
        "AUC_admission": auc_mimic,
        "Brier_admission": bri_mimic,
        "PRAUC_admission": prc_mimic,
        "Fold_AUCs": str([f"{a:.4f}" for a in fold_aucs_mimic]) if fold_aucs_mimic else "from checkpoint",
        "AUC_patient_v11": v11_mimic_auc,
        "Brier_patient_v11": v11_mimic_bri,
        "Delta_AUC": auc_mimic - v11_mimic_auc,
        "N_features": len(mimic_feats),
        "CV_method": "StratifiedKFold(5)",
    })

    # ── Save summary metrics ──────────────────────────────────────────────────
    summary_df = pd.DataFrame(metrics)
    summary_df.to_csv(OUT_DIR / "summary_metrics.csv", index=False)
    print("\nSaved summary_metrics.csv")
    print(summary_df[["Cohort", "N_rows", "N_patients", "MACE_rate_pct",
                        "AUC_admission", "AUC_patient_v11", "Delta_AUC"]].to_string(index=False))

    # ── Comparison table ──────────────────────────────────────────────────────
    comp_rows = []
    for c in ["SCH", "WHU"]:
        d = summary_df[summary_df.Cohort == c].iloc[0]
        comp_rows.append({
            "Cohort": c,
            "v11_N_patients": int(d["N_patients"]),
            "v11_AUC": round(d["AUC_patient_v11"], 4),
            "v11_Brier": round(d["Brier_patient_v11"], 4),
            "Adm_N_encounters": int(d["N_rows"]),
            "Adm_AUC": round(d["AUC_admission"], 4),
            "Adm_Brier": round(d["Brier_admission"], 4),
            "Delta_AUC": round(d["Delta_AUC"], 4),
            "Note": "GroupKFold(5) by 病案号 — no patient leakage",
        })
    d_m = summary_df[summary_df.Cohort == "MIMIC"].iloc[0]
    comp_rows.append({
        "Cohort": "MIMIC",
        "v11_N_patients": int(d_m["N_patients"]),
        "v11_AUC": round(float(d_m["AUC_patient_v11"]), 4),
        "v11_Brier": round(float(d_m["Brier_patient_v11"]), 4),
        "Adm_N_encounters": int(d_m["N_rows"]),
        "Adm_AUC": round(d_m["AUC_admission"], 4),
        "Adm_Brier": round(d_m["Brier_admission"], 4),
        "Delta_AUC": round(float(d_m["Delta_AUC"]), 4),
        "Note": "MIMIC: admission-level vs patient-level (median agg), StratifiedKFold(5)",
    })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT_DIR / "Table_Admission_vs_Patient_Comparison.csv", index=False)
    print("Saved Table_Admission_vs_Patient_Comparison.csv")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    fig1_roc(results)
    fig2_dca(results)
    fig3_calibration(results)
    fig4_feature_importance(fi_store)
    fig5_comparison_bar(summary_df)
    fig6_temporal_gradient({
        "SCH":   (OUT_DIR / "predictions_admission_SCH.csv",   "病案号"),
        "WHU":   (OUT_DIR / "predictions_admission_WHU.csv",   "病案号"),
        "MIMIC": (OUT_DIR / "predictions_admission_MIMIC.csv", "SUBJECT_ID"),
    })

    print("\n" + "=" * 70)
    print("All outputs saved to:", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Use checkpoint CSV/NPY files to regenerate figures without retraining")
    args = parser.parse_args()

    if args.plot_only:
        plot_only_from_checkpoints()
    else:
        main()
