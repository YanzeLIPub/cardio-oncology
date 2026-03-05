"""
05_pretrain_transfer.py
─────────────────────────────────────────────────────────────────────────────
Pre-train the ensemble model on SCH + WHU (full raw lab features, English
schema), then transfer to MIMIC-III and evaluate.

Strategy:
  1. Load SCH raw features (usecols to handle 993 MB file)  →  aggregate per
     patient (median), merge with SCH MACE labels
  2. Load WHU raw features  →  aggregate, merge with WHU MACE labels
  3. Align both to the same 43 English feature schema
  4. Train EnsembleModel (XGBoost + RF) on combined SCH+WHU
  5. Isotonic calibration on held-out fold
  6. Transfer to MIMIC-III (features_model.csv + mace_labels.csv)
  7. Evaluate and output results + figures
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                              roc_curve)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.impute import SimpleImputer

# ── paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCH_RAW   = r'D:\cardio-oncology\SCH_processed_data.csv'
WHU_RAW   = r'D:\cardio-oncology\files\WHU_Cancer_processed_data.csv'
# Use real_model_predictions_v2.csv — contains ALL 38,509 SCH + 2,562 WHU
# patients with correct MACE labels (SCH 3.44% / WHU 10.97%)
ALL_MACE  = r'D:\cardio-oncology\outputs\ensemble_model\real_model_predictions_v2.csv'
MIMIC_FT  = os.path.join(ROOT, 'MIMIC-3', 'data', 'features_model.csv')
MIMIC_LB  = os.path.join(ROOT, 'MIMIC-3', 'data', 'mace_labels.csv')
OUT_DIR   = os.path.join(ROOT, 'MIMIC-3', 'results')
FIG_DIR   = os.path.join(OUT_DIR, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
from models.ensemble import EnsembleModel

warnings.filterwarnings('ignore')

# ── Chinese → English feature map ────────────────────────────────────────────
CN_TO_EN = {
    '年龄':             'Age',
    '性别':             'Sex_M',
    '载脂蛋白B':        'Apolipoprotein B',
    '低密度脂蛋白胆固醇':'LDL Cholesterol',
    '总胆固醇':         'Total Cholesterol',
    '胆碱酯酶':         'Cholinesterase',
    '白蛋白':           'Albumin',
    '总胆汁酸':         'Total Bile Acids',
    '前白蛋白':         'Prealbumin',
    'D-二聚体':         'D-Dimer',
    '细胞角蛋白19片段':  'CYFRA 21-1',
    '胱抑素C':          'Cystatin C',
    '甘油三脂':         'Triglycerides',
    '腺苷脱氨酶':       'Adenosine Deaminase',
    '球蛋白':           'Globulin',
    '血小板压积':       'Plateletcrit',
    '鳞状细胞癌相关抗原':'SCC Antigen',
    '直接胆红素':       'Direct Bilirubin',
    '医生数(人)':       'No. of Physicians',
    '医院、卫生院床位数(张)': 'Hospital Beds',
    '葡萄糖':           'Glucose',
    '高密度脂蛋白胆固醇':'HDL Cholesterol',
    '镁':               'Magnesium',
    'α-羟丁酸脱氢酶':   'α-HBDH',
    '丙氨酸氨基转移酶': 'ALT',
    '人均地区生产总值(元)': 'GDP per Capita',
    '医院、卫生院数(个)': 'No. of Hospitals',
    '乙肝核心抗体':     'HBcAb',
    '乳酸脱氢酶':       'LDH',
    '磷':               'Phosphorus',
    '天门冬氨酸氨基转移酶': 'AST',
    '尿酸':             'Uric Acid',
    '甲胎蛋白':         'Alpha-Fetoprotein',
    '总蛋白':           'Total Protein',
    '糖化血红蛋白':     'HbA1c',
    '肌酐':             'Creatinine',
    '肌酸激酶':         'Creatine Kinase',
    '职工平均工资(元)': 'Avg. Employee Wage',
    '钾':               'Potassium',
    '钠':               'Sodium',
    '载脂蛋白AⅠ':      'Apolipoprotein AI',
    '铁蛋白':           'Ferritin',
    '淀粉酶':           'Amylase',
}

FEATS_CN = list(CN_TO_EN.keys())        # 43 Chinese names
FEATS_EN = list(CN_TO_EN.values())      # 43 English names


# ═══════════════════════════════════════════════════════════════════════════
# 1. Helper: load raw feature CSV, aggregate per patient, encode 性别
# ═══════════════════════════════════════════════════════════════════════════
def load_raw_features(raw_path: str, label: str = '') -> pd.DataFrame:
    """
    Load a wide-format CSV with Chinese lab column names.
    Returns one row per 病案号 with the 43 feature columns
    renamed to English, plus 'hadm_id' (= 病案号).
    """
    print(f'\n[{label}] Loading {os.path.basename(raw_path)} ...')
    # Only read the columns we need (huge speedup for 993 MB SCH file)
    available_header = pd.read_csv(raw_path, nrows=0).columns.tolist()
    needed = ['病案号'] + [c for c in FEATS_CN if c in available_header]
    missing_cn = [c for c in FEATS_CN if c not in available_header]
    if missing_cn:
        print(f'  WARNING: {len(missing_cn)} features absent in raw file: {missing_cn[:5]}...')

    print(f'  Reading {len(needed)} columns ...')
    df = pd.read_csv(raw_path, usecols=needed, low_memory=False)
    print(f'  Raw shape: {df.shape}')

    # Encode 性别 → Sex_M: 男=1, 女=0
    if '性别' in df.columns:
        df['性别'] = df['性别'].map(
            lambda x: 1 if str(x).strip() in ('男', '1', '1.0', 'M', 'male') else
                      (0 if str(x).strip() in ('女', '0', '0.0', 'F', 'female') else np.nan)
        )

    # Force numeric for all feature columns
    for c in FEATS_CN:
        if c in df.columns and c != '性别':
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Aggregate per patient: median of numeric, mode of 性别
    print(f'  Aggregating by 病案号 ...')
    agg_dict = {}
    for c in FEATS_CN:
        if c in df.columns:
            if c == '性别':
                agg_dict[c] = lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan
            else:
                agg_dict[c] = 'median'
    
    df_agg = df.groupby('病案号').agg(agg_dict).reset_index()
    print(f'  Aggregated shape (unique patients): {df_agg.shape}')

    # Rename Chinese → English
    rename_map = {cn: en for cn, en in CN_TO_EN.items() if cn in df_agg.columns}
    df_agg = df_agg.rename(columns=rename_map)
    df_agg = df_agg.rename(columns={'病案号': 'hadm_id'})

    # Fill missing English columns with NaN
    for en in FEATS_EN:
        if en not in df_agg.columns:
            df_agg[en] = np.nan

    return df_agg[['hadm_id'] + FEATS_EN]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Load MACE labels from the full predictions file (all patients)
# ═══════════════════════════════════════════════════════════════════════════
def load_mace_center(path: str, center: str) -> pd.DataFrame:
    """
    Load MACE labels for one center from real_model_predictions_v2.csv.
    SCH: 38,509 patients  3.44%
    WHU:  2,562 patients 10.97%
    """
    df = pd.read_csv(path)[['病案号', 'Center', 'MACE']]
    df = df[df['Center'] == center][['病案号', 'MACE']].copy()
    df = df.groupby('病案号', as_index=False)['MACE'].max()
    df = df.rename(columns={'病案号': 'hadm_id', 'MACE': 'label'})
    df['label'] = df['label'].astype(int)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. Load and merge cohort
# ═══════════════════════════════════════════════════════════════════════════
print('=' * 70)
print('STEP 1-2: Loading SCH + WHU data')
print('=' * 70)

sch_feat  = load_raw_features(SCH_RAW, 'SCH')
sch_mace  = load_mace_center(ALL_MACE, 'SCH')   # 38,509 patients, 3.44% MACE
whu_feat  = load_raw_features(WHU_RAW, 'WHU')
whu_mace  = load_mace_center(ALL_MACE, 'WHU')   # 2,562 patients, 10.97% MACE

sch = sch_feat.merge(sch_mace, on='hadm_id').copy()
whu = whu_feat.merge(whu_mace, on='hadm_id').copy()

sch['source'] = 'SCH'
whu['source'] = 'WHU'

print(f'\nSCH merged: {sch.shape}  MACE rate: {sch.label.mean():.3f}')
print(f'WHU merged: {whu.shape}  MACE rate: {whu.label.mean():.3f}')

combined = pd.concat([sch, whu], ignore_index=True)
print(f'Combined:   {combined.shape}  MACE rate: {combined.label.mean():.3f}')

X_train = combined[FEATS_EN].values.astype(float)
y_train = combined['label'].values.astype(int)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Within-cohort 5-fold CV for performance reporting + OOF probabilities
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('STEP 3: Within-cohort 5-fold CV (SCH+WHU)')
print('=' * 70)

imputer = SimpleImputer(strategy='median')
X_imp   = imputer.fit_transform(X_train)

# fit_cv returns (oof_probs, fold_aucs) and stores CV fold models internally
cv_model  = EnsembleModel(random_state=42)
oof_proba, fold_aucs = cv_model.fit_cv(X_imp, y_train, verbose=True)

cv_auc   = fold_aucs
cv_brier = [brier_score_loss(y_train[vl_idx], oof_proba[vl_idx])
            for _, vl_idx in StratifiedKFold(n_splits=5, shuffle=True,
                                             random_state=42).split(X_imp, y_train)]
print(f'\n  CV AUC  : {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}')
print(f'  CV Brier: {np.mean(cv_brier):.4f} ± {np.std(cv_brier):.4f}')


# ═══════════════════════════════════════════════════════════════════════════
# 5. Train FINAL model for Transfer + isotonic calibration on OOF preds
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('STEP 4: Train final model on all SCH+WHU (for feature importance)')
print('=' * 70)

final_model = EnsembleModel(random_state=42)
final_model.fit(X_imp, y_train)     # populates _xgb_full, _rf_full

# Isotonic calibration using OOF probabilities
iso_cal = IsotonicRegression(out_of_bounds='clip')
iso_cal.fit(oof_proba, y_train)

# For Transfer we use the CV-averaged model (cv_model) for more robust preds
print('  Final model trained. Calibrator fitted on OOF predictions.')


# ═══════════════════════════════════════════════════════════════════════════
# 6. Transfer to MIMIC-III
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('STEP 5: Transfer to MIMIC-III')
print('=' * 70)

mimic_ft = pd.read_csv(MIMIC_FT)
mimic_lb = pd.read_csv(MIMIC_LB)

# Align feature columns to training schema
for col in FEATS_EN:
    if col not in mimic_ft.columns:
        mimic_ft[col] = np.nan

X_mimic_raw = mimic_ft[FEATS_EN].values.astype(float)

# Impute MIMIC with training medians (from fitted imputer)
X_mimic = imputer.transform(X_mimic_raw)

mimic_prob_raw  = cv_model.predict(X_mimic)   # averaged over 5 CV fold models
mimic_prob_cal  = iso_cal.predict(mimic_prob_raw)

# Merge with labels
mimic_ft = mimic_ft.reset_index(drop=True)
mimic_lb_merged = mimic_lb.reset_index(drop=True)
hadm_col = [c for c in mimic_ft.columns if 'hadm' in c.lower() or 'id' in c.lower()][0]
lb_col   = [c for c in mimic_lb_merged.columns if 'mace' in c.lower() or 'label' in c.lower()][0]

# Try to merge on hadm_id
if hadm_col in mimic_lb_merged.columns:
    mimic_merged = mimic_ft[[hadm_col]].merge(mimic_lb_merged, on=hadm_col, how='inner')
else:
    # Fall back to row-by-row alignment
    mimic_merged = pd.DataFrame({'hadm_id': range(len(mimic_ft)), lb_col: mimic_lb_merged[lb_col].values[:len(mimic_ft)]})

# Re-align prediction arrays to merged index
def parse_mace_label(s):
    """Convert MACE column which may be 0/1/int or event strings → binary int."""
    if pd.isna(s) or s == '' or s == 'None' or s == 0 or s == '0':
        return 0
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 1  # non-empty string like 'Stroke|Cardiac_Death' → MACE positive

if hadm_col in mimic_lb_merged.columns:
    idx_keep = mimic_merged.index.tolist()
    keep_mask = mimic_ft[hadm_col].isin(mimic_merged[hadm_col])
    y_mimic       = mimic_merged[lb_col].apply(parse_mace_label).values
    prob_raw_keep = mimic_prob_raw[keep_mask]
    prob_cal_keep = mimic_prob_cal[keep_mask]
else:
    y_mimic       = mimic_merged[lb_col].apply(parse_mace_label).values
    n = len(y_mimic)
    prob_raw_keep = mimic_prob_raw[:n]
    prob_cal_keep = mimic_prob_cal[:n]

auc_raw = roc_auc_score(y_mimic, prob_raw_keep)
auc_cal = roc_auc_score(y_mimic, prob_cal_keep)
bri_raw = brier_score_loss(y_mimic, prob_raw_keep)
bri_cal = brier_score_loss(y_mimic, prob_cal_keep)

print(f'\n  MIMIC MACE rate: {y_mimic.mean():.3f}  (n={len(y_mimic)})')
print(f'  Transfer AUC (raw):       {auc_raw:.4f}')
print(f'  Transfer AUC (calibrated):{auc_cal:.4f}')
print(f'  Transfer Brier (raw):     {bri_raw:.4f}')
print(f'  Transfer Brier (cal):     {bri_cal:.4f}')

# Compare with previous naive transfer (from 04_predict_mace.py)
prev_auc = 0.561
print(f'\n  Improvement vs previous naive transfer: {auc_cal - prev_auc:+.4f} AUC')


# ═══════════════════════════════════════════════════════════════════════════
# 7. Save predictions and metrics
# ═══════════════════════════════════════════════════════════════════════════
pred_df = pd.DataFrame({
    'hadm_id':          mimic_merged.get(hadm_col, range(len(y_mimic))).values if hasattr(mimic_merged.get(hadm_col, None), 'values') else range(len(y_mimic)),
    'mace_label':       y_mimic,
    'prob_pretrained_raw': prob_raw_keep,
    'prob_pretrained_cal': prob_cal_keep,
})
pred_df.to_csv(os.path.join(OUT_DIR, 'transfer_pretrained_predictions.csv'), index=False)

metrics_df = pd.DataFrame([
    {'metric': 'Within-SCH+WHU CV AUC',   'value': np.mean(cv_auc),   'std': np.std(cv_auc)},
    {'metric': 'Within-SCH+WHU CV Brier', 'value': np.mean(cv_brier), 'std': np.std(cv_brier)},
    {'metric': 'MIMIC Transfer AUC (raw)',       'value': auc_raw, 'std': np.nan},
    {'metric': 'MIMIC Transfer AUC (calibrated)','value': auc_cal, 'std': np.nan},
    {'metric': 'MIMIC Transfer Brier (raw)',      'value': bri_raw, 'std': np.nan},
    {'metric': 'MIMIC Transfer Brier (cal)',      'value': bri_cal, 'std': np.nan},
    {'metric': 'Previous naive transfer AUC',     'value': prev_auc, 'std': np.nan},
    {'metric': 'AUC improvement',                 'value': auc_cal - prev_auc, 'std': np.nan},
    {'metric': 'N training (SCH+WHU)',            'value': len(y_train), 'std': np.nan},
    {'metric': 'N MIMIC test',                    'value': len(y_mimic), 'std': np.nan},
    {'metric': 'SCH MACE rate',                   'value': sch.label.mean(), 'std': np.nan},
    {'metric': 'WHU MACE rate',                   'value': whu.label.mean(), 'std': np.nan},
    {'metric': 'MIMIC MACE rate',                 'value': y_mimic.mean(),   'std': np.nan},
])
metrics_df.to_csv(os.path.join(OUT_DIR, 'transfer_evaluation_metrics.csv'), index=False)
print(f'\n  Saved predictions → transfer_pretrained_predictions.csv')
print(f'  Saved metrics    → transfer_evaluation_metrics.csv')


# ═══════════════════════════════════════════════════════════════════════════
# 8. Figures
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('STEP 6: Generating figures')
print('=' * 70)

MM = 1/25.4
fig, axes = plt.subplots(1, 3, figsize=(170*MM, 60*MM))
fig.patch.set_facecolor('white')

# ── A: ROC curve ─────────────────────────────────────────────────────────
ax = axes[0]
fpr, tpr, _ = roc_curve(y_mimic, prob_cal_keep)
ax.plot(fpr, tpr, color='#2166AC', lw=1.5,
        label=f'Pre-trained transfer\nAUC = {auc_cal:.3f}')
ax.plot([0,1],[0,1],'--', color='gray', lw=0.8)
ax.set_xlabel('1 − Specificity', fontsize=7)
ax.set_ylabel('Sensitivity', fontsize=7)
ax.set_title('A  ROC Curve (MIMIC-III)', fontsize=7, fontweight='bold')
ax.legend(fontsize=6, frameon=False)
ax.tick_params(labelsize=6)
ax.set_xlim(0,1); ax.set_ylim(0,1)

# ── B: Calibration curve ─────────────────────────────────────────────────
ax = axes[1]
frac_pos, mean_pred = calibration_curve(y_mimic, prob_cal_keep, n_bins=10)
ax.plot(mean_pred, frac_pos, 'o-', color='#D6604D', lw=1.5, ms=3,
        label='Pre-trained (cal)')
frac_pos_r, mean_pred_r = calibration_curve(y_mimic, prob_raw_keep, n_bins=10)
ax.plot(mean_pred_r, frac_pos_r, 's--', color='#4393C3', lw=1.0, ms=2,
        label='Pre-trained (raw)')
ax.plot([0,1],[0,1],'--', color='gray', lw=0.8, label='Ideal')
ax.set_xlabel('Mean predicted probability', fontsize=7)
ax.set_ylabel('Fraction of positives', fontsize=7)
ax.set_title('B  Calibration (MIMIC-III)', fontsize=7, fontweight='bold')
ax.legend(fontsize=6, frameon=False)
ax.tick_params(labelsize=6)

# ── C: AUC comparison bar ─────────────────────────────────────────────────
ax = axes[2]
labels_bar = ['Previous\nnaive', 'Pre-trained\n(raw)', 'Pre-trained\n(cal)']
vals_bar   = [prev_auc, auc_raw, auc_cal]
colors_bar = ['#AAAAAA', '#4393C3', '#2166AC']
bars = ax.bar(labels_bar, vals_bar, color=colors_bar, width=0.5, edgecolor='none')
for bar, v in zip(bars, vals_bar):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{v:.3f}', ha='center', va='bottom', fontsize=6)
ax.set_ylim(0.4, 0.85)
ax.set_ylabel('AUC-ROC', fontsize=7)
ax.set_title('C  Transfer AUC Comparison', fontsize=7, fontweight='bold')
ax.tick_params(labelsize=6)
ax.axhline(0.5, color='gray', lw=0.8, ls='--')

plt.tight_layout(pad=0.5)
fig_path = os.path.join(FIG_DIR, 'Fig_Transfer_Pretrained')
for ext in ('pdf', 'png'):
    plt.savefig(fig_path + '.' + ext, dpi=300, bbox_inches='tight')
plt.close()
print(f'  Saved figure → {fig_path}.pdf / .png')

# ── Figure 2: Cohort comparison ───────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(90*MM, 60*MM))
cohorts  = ['SCH\n(train)', 'WHU\n(train)', 'MIMIC-III\n(test)']
mace_rt  = [sch.label.mean(), whu.label.mean(), y_mimic.mean()]
sizes    = [len(sch), len(whu), len(y_mimic)]
cols2    = ['#2166AC', '#4DAC26', '#D6604D']
bars2    = ax2.bar(cohorts, mace_rt, color=cols2, width=0.5, edgecolor='none')
for bar, v, n in zip(bars2, mace_rt, sizes):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{v:.1%}\n(n={n:,})', ha='center', va='bottom', fontsize=6)
ax2.set_ylim(0, 1.0)
ax2.set_ylabel('MACE Rate', fontsize=7)
ax2.set_title('Cohort Comparison', fontsize=7, fontweight='bold')
ax2.tick_params(labelsize=6)
plt.tight_layout()
fig2_path = os.path.join(FIG_DIR, 'Fig_Transfer_Cohorts')
for ext in ('pdf', 'png'):
    plt.savefig(fig2_path + '.' + ext, dpi=300, bbox_inches='tight')
plt.close()
print(f'  Saved figure → {fig2_path}.pdf / .png')


# ── Feature importance from final_model._xgb_full ───────────────────────
try:
    fi_dict = final_model.feature_importances(FEATS_EN, top_n=20)
    fi = pd.DataFrame(list(fi_dict.items()), columns=['feature', 'importance'])
    fi = fi.sort_values('importance', ascending=False)
    fi.to_csv(os.path.join(OUT_DIR, 'transfer_feature_importance.csv'), index=False)

    fig3, ax3 = plt.subplots(figsize=(90*MM, 80*MM))
    ax3.barh(fi['feature'][::-1].values, fi['importance'][::-1].values,
             color='#2166AC', edgecolor='none')
    ax3.set_xlabel('XGBoost Feature Importance (Gain)', fontsize=7)
    ax3.set_title('Top Feature Importance (Pre-trained Model)', fontsize=7, fontweight='bold')
    ax3.tick_params(labelsize=6)
    plt.tight_layout()
    fig3_path = os.path.join(FIG_DIR, 'Fig_Transfer_FeatureImportance')
    for ext in ('pdf', 'png'):
        plt.savefig(fig3_path + '.' + ext, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved figure → {fig3_path}.pdf / .png')
except Exception as e:
    print(f'  (Feature importance skipped: {e})')


# ═══════════════════════════════════════════════════════════════════════════
# 9. Summary
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'  Training cohort  : SCH ({len(sch):,}) + WHU ({len(whu):,}) = {len(combined):,} patients')
print(f'  Training MACE    : {y_train.mean():.1%}')
print(f'  Test cohort      : MIMIC-III ({len(y_mimic):,} admissions, {y_mimic.mean():.1%} MACE)')
print()
print(f'  Within SCH+WHU CV AUC : {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}')
print(f'  MIMIC Transfer AUC    : {auc_cal:.4f}  (prev: {prev_auc:.4f}, Δ={auc_cal-prev_auc:+.4f})')
print(f'  MIMIC Brier score     : {bri_cal:.4f}')
print()
print('  Output files:')
print(f'    {OUT_DIR}/transfer_pretrained_predictions.csv')
print(f'    {OUT_DIR}/transfer_evaluation_metrics.csv')
print(f'    {FIG_DIR}/Fig_Transfer_Pretrained.pdf/.png')
print(f'    {FIG_DIR}/Fig_Transfer_Cohorts.pdf/.png')
print('\nDone.')
