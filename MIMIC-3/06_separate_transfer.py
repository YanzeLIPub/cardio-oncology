"""
06_separate_transfer.py
────────────────────────────────────────────────────────────────────────────
Three experiments:

  A. SCH-only  →  Train on SCH (38,509 pts)        → Transfer to MIMIC-III
  B. WHU-only  →  Train on WHU  (2,562 pts)         → Transfer to MIMIC-III
  C. Full MIMIC validation:
       Expand features_raw.csv (long-format, 23 available MIMIC features)
       → pivot to wide per admission → include ALL 11,622 admissions
         (with NaN imputation for those missing lab data)
       → within-MIMIC 5-fold CV using all 23 available features

Comparison table across:
  • SCH → MIMIC transfer     (43-feature aligned schema)
  • WHU → MIMIC transfer     (43-feature aligned schema)
  • SCH+WHU → MIMIC transfer (from 05_pretrain_transfer.py, AUC 0.552)
  • MIMIC within-CV          (all 23 MIMIC-native features)

Outputs (all in MIMIC-3/results/):
  separate_transfer_metrics.csv
  separate_transfer_predictions.csv   (SCH-pred, WHU-pred, MIMIC-cv-pred per admission)
  figures/Fig_Separate_Transfer.pdf/.png
  figures/Fig_MIMIC_FullFeatures.pdf/.png
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer

# ── paths ─────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCH_RAW   = r'D:\cardio-oncology\SCH_processed_data.csv'
WHU_RAW   = r'D:\cardio-oncology\files\WHU_Cancer_processed_data.csv'
ALL_MACE  = r'D:\cardio-oncology\outputs\ensemble_model\real_model_predictions_v2.csv'
MIMIC_FT  = os.path.join(ROOT, 'MIMIC-3', 'data', 'features_model.csv')
MIMIC_LB  = os.path.join(ROOT, 'MIMIC-3', 'data', 'mace_labels.csv')
MIMIC_RAW = os.path.join(ROOT, 'MIMIC-3', 'data', 'features_raw.csv')
OUT_DIR   = os.path.join(ROOT, 'MIMIC-3', 'results')
FIG_DIR   = os.path.join(OUT_DIR, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
from models.ensemble import EnsembleModel

warnings.filterwarnings('ignore')

# ── Chinese → English feature map (same 43 features as 05_) ──────────────
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
FEATS_CN = list(CN_TO_EN.keys())
FEATS_EN = list(CN_TO_EN.values())


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers (same as 05_pretrain_transfer.py)
# ═══════════════════════════════════════════════════════════════════════════
def load_raw_features(raw_path: str, label: str = '') -> pd.DataFrame:
    print(f'\n[{label}] Loading {os.path.basename(raw_path)} ...')
    available_header = pd.read_csv(raw_path, nrows=0).columns.tolist()
    needed = ['病案号'] + [c for c in FEATS_CN if c in available_header]
    missing_cn = [c for c in FEATS_CN if c not in available_header]
    if missing_cn:
        print(f'  WARNING: {len(missing_cn)} features absent: {missing_cn[:5]}...')
    print(f'  Reading {len(needed)} columns ...')
    df = pd.read_csv(raw_path, usecols=needed, low_memory=False)
    print(f'  Raw shape: {df.shape}')

    if '性别' in df.columns:
        df['性别'] = df['性别'].map(
            lambda x: 1 if str(x).strip() in ('男','1','1.0','M','male') else
                      (0 if str(x).strip() in ('女','0','0.0','F','female') else np.nan)
        )
    for c in FEATS_CN:
        if c in df.columns and c != '性别':
            df[c] = pd.to_numeric(df[c], errors='coerce')

    print(f'  Aggregating by 病案号 ...')
    agg_dict = {}
    for c in FEATS_CN:
        if c in df.columns:
            agg_dict[c] = (lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan) \
                          if c == '性别' else 'median'
    df_agg = df.groupby('病案号').agg(agg_dict).reset_index()
    print(f'  Aggregated shape: {df_agg.shape}')

    rename_map = {cn: en for cn, en in CN_TO_EN.items() if cn in df_agg.columns}
    df_agg = df_agg.rename(columns=rename_map)
    df_agg = df_agg.rename(columns={'病案号': 'hadm_id'})
    for en in FEATS_EN:
        if en not in df_agg.columns:
            df_agg[en] = np.nan
    return df_agg[['hadm_id'] + FEATS_EN]


def load_mace_center(path: str, center: str) -> pd.DataFrame:
    df = pd.read_csv(path)[['病案号', 'Center', 'MACE']]
    df = df[df['Center'] == center][['病案号','MACE']].copy()
    df = df.groupby('病案号', as_index=False)['MACE'].max()
    df = df.rename(columns={'病案号': 'hadm_id', 'MACE': 'label'})
    df['label'] = df['label'].astype(int)
    return df


def parse_mace(s):
    if pd.isna(s) or s in ('', 'None', 0, '0'):
        return 0
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 1


def run_cv(X_imp, y, n_splits=5, label=''):
    """5-fold CV → returns (oof_proba, fold_aucs, fold_briers)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    aucs, briers = [], []
    for fold, (tr, vl) in enumerate(skf.split(X_imp, y)):
        m = EnsembleModel(random_state=fold)
        m.fit(X_imp[tr], y[tr])
        oof[vl] = m.predict(X_imp[vl])
        aucs.append(roc_auc_score(y[vl], oof[vl]))
        briers.append(brier_score_loss(y[vl], oof[vl]))
        print(f'    [{label}] Fold {fold+1}: AUC={aucs[-1]:.4f}')
    return oof, aucs, briers


def train_full_with_iso(X_imp, y, oof_proba):
    """Train final model + isotonic calibrator on OOF preds."""
    m = EnsembleModel(random_state=42)
    m.fit(X_imp, y)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(oof_proba, y)
    return m, iso


# ═══════════════════════════════════════════════════════════════════════════
# PART 0 — Load MIMIC test set (aligned schema, all 11,622 admissions)
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('PART 0  Loading MIMIC-III test data (aligned 43-feature schema)')
print('=' * 70)

mimic_ft = pd.read_csv(MIMIC_FT)
mimic_lb = pd.read_csv(MIMIC_LB)

for col in FEATS_EN:
    if col not in mimic_ft.columns:
        mimic_ft[col] = np.nan

# Merge to get MACE label aligned to feature rows
lb_col = [c for c in mimic_lb.columns if 'mace' in c.lower()][0]
mimic_align = mimic_ft.merge(mimic_lb[['HADM_ID', lb_col]], on='HADM_ID', how='inner')
mimic_align[lb_col] = mimic_align[lb_col].apply(parse_mace)

X_mimic_raw = mimic_align[FEATS_EN].values.astype(float)
y_mimic     = mimic_align[lb_col].values.astype(int)

print(f'  MIMIC test: {len(y_mimic):,} admissions, MACE rate {y_mimic.mean():.3%}')


# ═══════════════════════════════════════════════════════════════════════════
# PART A — SCH only → MIMIC transfer
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('PART A  SCH-only model → MIMIC transfer')
print('=' * 70)

sch_feat = load_raw_features(SCH_RAW, 'SCH')
sch_mace = load_mace_center(ALL_MACE, 'SCH')
sch = sch_feat.merge(sch_mace, on='hadm_id')
print(f'\n  SCH: {len(sch):,} patients, MACE {sch.label.mean():.3%}')

X_sch = sch[FEATS_EN].values.astype(float)
y_sch = sch['label'].values.astype(int)

imp_sch = SimpleImputer(strategy='median')
X_sch_imp = imp_sch.fit_transform(X_sch)

print('\n  Running 5-fold CV on SCH ...')
oof_sch, aucs_sch, briers_sch = run_cv(X_sch_imp, y_sch, label='SCH')

print('\n  Training final SCH model ...')
final_sch, iso_sch = train_full_with_iso(X_sch_imp, y_sch, oof_sch)

# Transfer to MIMIC
X_mimic_sch = imp_sch.transform(X_mimic_raw)     # impute with SCH medians
prob_sch_raw = final_sch.predict(X_mimic_sch)
prob_sch_cal = iso_sch.predict(prob_sch_raw)

auc_sch_raw = roc_auc_score(y_mimic, prob_sch_raw)
auc_sch_cal = roc_auc_score(y_mimic, prob_sch_cal)
bri_sch_raw = brier_score_loss(y_mimic, prob_sch_raw)
bri_sch_cal = brier_score_loss(y_mimic, prob_sch_cal)

print(f'\n  SCH CV AUC       : {np.mean(aucs_sch):.4f} ± {np.std(aucs_sch):.4f}')
print(f'  SCH→MIMIC AUC raw: {auc_sch_raw:.4f}')
print(f'  SCH→MIMIC AUC cal: {auc_sch_cal:.4f}')
print(f'  SCH→MIMIC Brier  : {bri_sch_raw:.4f}')


# ═══════════════════════════════════════════════════════════════════════════
# PART B — WHU only → MIMIC transfer
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('PART B  WHU-only model → MIMIC transfer')
print('=' * 70)

whu_feat = load_raw_features(WHU_RAW, 'WHU')
whu_mace = load_mace_center(ALL_MACE, 'WHU')
whu = whu_feat.merge(whu_mace, on='hadm_id')
print(f'\n  WHU: {len(whu):,} patients, MACE {whu.label.mean():.3%}')

X_whu = whu[FEATS_EN].values.astype(float)
y_whu = whu['label'].values.astype(int)

imp_whu = SimpleImputer(strategy='median')
X_whu_imp = imp_whu.fit_transform(X_whu)

print('\n  Running 5-fold CV on WHU ...')
oof_whu, aucs_whu, briers_whu = run_cv(X_whu_imp, y_whu, label='WHU')

print('\n  Training final WHU model ...')
final_whu, iso_whu = train_full_with_iso(X_whu_imp, y_whu, oof_whu)

# Transfer to MIMIC
X_mimic_whu = imp_whu.transform(X_mimic_raw)     # impute with WHU medians
prob_whu_raw = final_whu.predict(X_mimic_whu)
prob_whu_cal = iso_whu.predict(prob_whu_raw)

auc_whu_raw = roc_auc_score(y_mimic, prob_whu_raw)
auc_whu_cal = roc_auc_score(y_mimic, prob_whu_cal)
bri_whu_raw = brier_score_loss(y_mimic, prob_whu_raw)
bri_whu_cal = brier_score_loss(y_mimic, prob_whu_cal)

print(f'\n  WHU CV AUC       : {np.mean(aucs_whu):.4f} ± {np.std(aucs_whu):.4f}')
print(f'  WHU→MIMIC AUC raw: {auc_whu_raw:.4f}')
print(f'  WHU→MIMIC AUC cal: {auc_whu_cal:.4f}')
print(f'  WHU→MIMIC Brier  : {bri_whu_raw:.4f}')


# ═══════════════════════════════════════════════════════════════════════════
# PART C — Full MIMIC: all features × all patients, within-CV
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('PART C  Full MIMIC-III: all available features × all admissions')
print('=' * 70)

print('\n  Pivoting features_raw.csv (long → wide) ...')
raw = pd.read_csv(MIMIC_RAW)

# Aggregate to one value per admission × feature (use last charted value per admission)
raw_sorted = raw.sort_values(['HADM_ID', 'Feature', 'CHARTTIME'])
raw_agg = (raw_sorted
           .groupby(['HADM_ID', 'Feature'], as_index=False)['VALUENUM']
           .median())

# Pivot to wide format
mimic_wide = raw_agg.pivot(index='HADM_ID', columns='Feature', values='VALUENUM')
mimic_wide.columns.name = None
mimic_wide = mimic_wide.reset_index()
print(f'  Pivoted wide shape: {mimic_wide.shape}')

# Merge with mace_labels using all 11,622 admissions
mimic_full = mimic_lb[['HADM_ID', lb_col]].copy()
mimic_full[lb_col] = mimic_full[lb_col].apply(parse_mace)
mimic_full = mimic_full.merge(mimic_wide, on='HADM_ID', how='left')

MIMIC_FEATS = [c for c in mimic_wide.columns if c != 'HADM_ID']
print(f'  All MIMIC features ({len(MIMIC_FEATS)}): {MIMIC_FEATS}')

# Add Age and Sex_M from features_model.csv if available
demo_cols = ['HADM_ID', 'Age', 'Sex_M']
demo = mimic_ft[[c for c in demo_cols if c in mimic_ft.columns]]
mimic_full = mimic_full.merge(demo, on='HADM_ID', how='left')
for dc in ['Age', 'Sex_M']:
    if dc not in MIMIC_FEATS and dc in mimic_full.columns:
        MIMIC_FEATS.append(dc)

# Ensure all feature columns are numeric
for col in MIMIC_FEATS:
    mimic_full[col] = pd.to_numeric(mimic_full[col], errors='coerce')

X_full = mimic_full[MIMIC_FEATS].values.astype(float)
y_full = mimic_full[lb_col].values.astype(int)

print(f'\n  Full MIMIC matrix: {X_full.shape}  MACE rate {y_full.mean():.3%}')
print(f'  Missing rate: {np.isnan(X_full).mean():.1%}')

imp_full = SimpleImputer(strategy='median')
X_full_imp = imp_full.fit_transform(X_full)

print('\n  Running 5-fold CV on full MIMIC (all features × all admissions) ...')
oof_full, aucs_full, briers_full = run_cv(X_full_imp, y_full, label='MIMIC-full')

print(f'\n  Full-MIMIC within-CV AUC  : {np.mean(aucs_full):.4f} ± {np.std(aucs_full):.4f}')
print(f'  Full-MIMIC within-CV Brier: {np.mean(briers_full):.4f} ± {np.std(briers_full):.4f}')


# ═══════════════════════════════════════════════════════════════════════════
# Save predictions
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Saving outputs ...')
print('=' * 70)

# Transfer predictions (aligned schema, 11,622 rows)
pred_df = mimic_align[['HADM_ID', lb_col]].copy().rename(columns={lb_col: 'mace_label'})
pred_df['prob_SCH_raw'] = prob_sch_raw
pred_df['prob_SCH_cal'] = prob_sch_cal
pred_df['prob_WHU_raw'] = prob_whu_raw
pred_df['prob_WHU_cal'] = prob_whu_cal
pred_df.to_csv(os.path.join(OUT_DIR, 'separate_transfer_predictions.csv'), index=False)
print('  Saved → separate_transfer_predictions.csv')

# Full-MIMIC within-CV predictions (all admissions)
full_pred_df = mimic_full[['HADM_ID', lb_col]].copy().rename(columns={lb_col: 'mace_label'})
full_pred_df['prob_MIMIC_fullCV'] = oof_full
full_pred_df.to_csv(os.path.join(OUT_DIR, 'mimic_full_cv_predictions.csv'), index=False)
print('  Saved → mimic_full_cv_predictions.csv')

# Metrics summary
rows = [
    # Within-cohort CV
    {'experiment': 'SCH within-CV',         'N_train': len(y_sch),   'MACE_rate_train': y_sch.mean(),
     'N_test': len(y_sch),   'MACE_rate_test': y_sch.mean(),
     'AUC_raw': np.mean(aucs_sch),  'AUC_std': np.std(aucs_sch),
     'Brier': np.mean(briers_sch), 'note': 'within-cohort CV'},
    {'experiment': 'WHU within-CV',         'N_train': len(y_whu),   'MACE_rate_train': y_whu.mean(),
     'N_test': len(y_whu),   'MACE_rate_test': y_whu.mean(),
     'AUC_raw': np.mean(aucs_whu),  'AUC_std': np.std(aucs_whu),
     'Brier': np.mean(briers_whu), 'note': 'within-cohort CV'},
    # Transfer
    {'experiment': 'SCH → MIMIC (raw)',     'N_train': len(y_sch),   'MACE_rate_train': y_sch.mean(),
     'N_test': len(y_mimic), 'MACE_rate_test': y_mimic.mean(),
     'AUC_raw': auc_sch_raw, 'AUC_std': np.nan,
     'Brier': bri_sch_raw,   'note': 'transfer, raw probs'},
    {'experiment': 'SCH → MIMIC (cal)',     'N_train': len(y_sch),   'MACE_rate_train': y_sch.mean(),
     'N_test': len(y_mimic), 'MACE_rate_test': y_mimic.mean(),
     'AUC_raw': auc_sch_cal, 'AUC_std': np.nan,
     'Brier': bri_sch_cal,   'note': 'transfer, isotonic calibrated'},
    {'experiment': 'WHU → MIMIC (raw)',     'N_train': len(y_whu),   'MACE_rate_train': y_whu.mean(),
     'N_test': len(y_mimic), 'MACE_rate_test': y_mimic.mean(),
     'AUC_raw': auc_whu_raw, 'AUC_std': np.nan,
     'Brier': bri_whu_raw,   'note': 'transfer, raw probs'},
    {'experiment': 'WHU → MIMIC (cal)',     'N_train': len(y_whu),   'MACE_rate_train': y_whu.mean(),
     'N_test': len(y_mimic), 'MACE_rate_test': y_mimic.mean(),
     'AUC_raw': auc_whu_cal, 'AUC_std': np.nan,
     'Brier': bri_whu_cal,   'note': 'transfer, isotonic calibrated'},
    {'experiment': 'SCH+WHU → MIMIC (ref)', 'N_train': len(y_sch)+len(y_whu), 'MACE_rate_train': (y_sch.sum()+y_whu.sum())/(len(y_sch)+len(y_whu)),
     'N_test': len(y_mimic), 'MACE_rate_test': y_mimic.mean(),
     'AUC_raw': 0.5520, 'AUC_std': np.nan,
     'Brier': np.nan,   'note': 'from 05_pretrain_transfer.py'},
    {'experiment': 'MIMIC within-CV (all 23 feats)', 'N_train': len(y_full), 'MACE_rate_train': y_full.mean(),
     'N_test': len(y_full),  'MACE_rate_test': y_full.mean(),
     'AUC_raw': np.mean(aucs_full), 'AUC_std': np.std(aucs_full),
     'Brier': np.mean(briers_full), 'note': 'within-MIMIC CV, all available features'},
]
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(os.path.join(OUT_DIR, 'separate_transfer_metrics.csv'), index=False)
print('  Saved → separate_transfer_metrics.csv')


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════
MM = 1/25.4

# ── Figure 1: ROC curves for SCH→MIMIC and WHU→MIMIC ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(180*MM, 65*MM))
fig.patch.set_facecolor('white')

# A: ROC SCH → MIMIC
ax = axes[0]
fpr_s, tpr_s, _ = roc_curve(y_mimic, prob_sch_raw)
fpr_w, tpr_w, _ = roc_curve(y_mimic, prob_whu_raw)
ax.plot(fpr_s, tpr_s, color='#2166AC', lw=1.5, label=f'SCH→MIMIC  AUC={auc_sch_raw:.3f}')
ax.plot(fpr_w, tpr_w, color='#4DAC26', lw=1.5, label=f'WHU→MIMIC  AUC={auc_whu_raw:.3f}')
ax.plot([0,1],[0,1],'--', color='gray', lw=0.8)
ax.set_xlabel('1 − Specificity', fontsize=7); ax.set_ylabel('Sensitivity', fontsize=7)
ax.set_title('A  Transfer ROC (MIMIC-III)', fontsize=7, fontweight='bold')
ax.legend(fontsize=6, frameon=False); ax.tick_params(labelsize=6)
ax.set_xlim(0,1); ax.set_ylim(0,1)

# B: Calibration curves SCH → MIMIC and WHU → MIMIC (raw)
ax = axes[1]
for probs, lbl, col in [(prob_sch_raw, 'SCH→MIMIC', '#2166AC'),
                         (prob_whu_raw, 'WHU→MIMIC', '#4DAC26')]:
    try:
        fp, mp = calibration_curve(y_mimic, probs, n_bins=10)
        ax.plot(mp, fp, 'o-', color=col, lw=1.2, ms=3, label=lbl)
    except Exception:
        pass
ax.plot([0,1],[0,1],'--', color='gray', lw=0.8, label='Ideal')
ax.set_xlabel('Mean predicted probability', fontsize=7)
ax.set_ylabel('Fraction of positives', fontsize=7)
ax.set_title('B  Calibration (MIMIC-III)', fontsize=7, fontweight='bold')
ax.legend(fontsize=6, frameon=False); ax.tick_params(labelsize=6)

# C: AUC bar comparison
ax = axes[2]
bar_labels = ['SCH→MIMIC\n(raw)', 'WHU→MIMIC\n(raw)', 'SCH+WHU→MIMIC\n(ref)']
bar_vals   = [auc_sch_raw, auc_whu_raw, 0.552]
bar_cols   = ['#2166AC', '#4DAC26', '#636363']
bars = ax.bar(bar_labels, bar_vals, color=bar_cols, width=0.55, edgecolor='none')
for bar, v in zip(bars, bar_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{v:.3f}', ha='center', va='bottom', fontsize=6)
ax.set_ylim(0.4, 0.75); ax.set_ylabel('AUC-ROC', fontsize=7)
ax.set_title('C  Transfer AUC Comparison', fontsize=7, fontweight='bold')
ax.axhline(0.5, color='gray', lw=0.8, ls='--')
ax.tick_params(labelsize=6)

plt.tight_layout(pad=0.5)
for ext in ('pdf','png'):
    plt.savefig(os.path.join(FIG_DIR, f'Fig_Separate_Transfer.{ext}'),
                dpi=300, bbox_inches='tight')
plt.close()
print('  Saved → figures/Fig_Separate_Transfer.pdf/.png')


# ── Figure 2: Full MIMIC-III (all features, all patients) ─────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(130*MM, 65*MM))
fig2.patch.set_facecolor('white')

# A: within-CV ROC
ax = axes2[0]
from sklearn.metrics import roc_curve as _roc
fpr_f, tpr_f, _ = _roc(y_full, oof_full)
ax.plot(fpr_f, tpr_f, color='#D6604D', lw=1.5,
        label=f'Within-CV AUC={np.mean(aucs_full):.3f}±{np.std(aucs_full):.3f}')
ax.plot([0,1],[0,1],'--', color='gray', lw=0.8)
ax.set_xlabel('1 − Specificity', fontsize=7); ax.set_ylabel('Sensitivity', fontsize=7)
ax.set_title('A  MIMIC Full-Feature ROC', fontsize=7, fontweight='bold')
ax.legend(fontsize=6, frameon=False); ax.tick_params(labelsize=6)
ax.set_xlim(0,1); ax.set_ylim(0,1)

# B: AUC summary across all experiments
ax = axes2[1]
exp_labels = ['SCH\nwithin-CV', 'WHU\nwithin-CV',
              'SCH→MIMIC', 'WHU→MIMIC',
              'SCH+WHU\n→MIMIC (ref)',
              'MIMIC\nfull-CV']
exp_aucs = [np.mean(aucs_sch), np.mean(aucs_whu),
            auc_sch_raw, auc_whu_raw,
            0.552,
            np.mean(aucs_full)]
exp_stds = [np.std(aucs_sch), np.std(aucs_whu),
            np.nan, np.nan,
            np.nan,
            np.std(aucs_full)]
exp_cols = ['#2166AC', '#4DAC26', '#92C5DE', '#A1D99B', '#636363', '#D6604D']
bars2 = ax.bar(exp_labels, exp_aucs, color=exp_cols, width=0.6, edgecolor='none')
for bar, v, s in zip(bars2, exp_aucs, exp_stds):
    label = f'{v:.3f}' if np.isnan(s) else f'{v:.3f}\n±{s:.3f}'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            label, ha='center', va='bottom', fontsize=5)
ax.set_ylim(0.4, 0.95); ax.set_ylabel('AUC-ROC', fontsize=7)
ax.set_title('B  AUC Summary All Experiments', fontsize=7, fontweight='bold')
ax.axhline(0.5, color='gray', lw=0.8, ls='--')
ax.tick_params(labelsize=5)

plt.tight_layout(pad=0.5)
for ext in ('pdf','png'):
    plt.savefig(os.path.join(FIG_DIR, f'Fig_MIMIC_FullFeatures.{ext}'),
                dpi=300, bbox_inches='tight')
plt.close()
print('  Saved → figures/Fig_MIMIC_FullFeatures.pdf/.png')


# ═══════════════════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'  SCH-only model ({len(y_sch):,} pts trained):')
print(f'    Within-SCH CV AUC : {np.mean(aucs_sch):.4f} ± {np.std(aucs_sch):.4f}')
print(f'    SCH→MIMIC AUC     : {auc_sch_raw:.4f}  (cal: {auc_sch_cal:.4f})')
print()
print(f'  WHU-only model ({len(y_whu):,} pts trained):')
print(f'    Within-WHU CV AUC : {np.mean(aucs_whu):.4f} ± {np.std(aucs_whu):.4f}')
print(f'    WHU→MIMIC AUC     : {auc_whu_raw:.4f}  (cal: {auc_whu_cal:.4f})')
print()
print(f'  SCH+WHU combined → MIMIC (from 05_pretrain_transfer.py):')
print(f'    AUC = 0.5520')
print()
print(f'  Full MIMIC within-CV ({len(y_full):,} admissions, {len(MIMIC_FEATS)} features):')
print(f'    AUC : {np.mean(aucs_full):.4f} ± {np.std(aucs_full):.4f}')
print(f'    Brier: {np.mean(briers_full):.4f}')
print()
print('  Output files:')
print(f'    {OUT_DIR}/separate_transfer_metrics.csv')
print(f'    {OUT_DIR}/separate_transfer_predictions.csv')
print(f'    {OUT_DIR}/mimic_full_cv_predictions.csv')
print(f'    {FIG_DIR}/Fig_Separate_Transfer.pdf/.png')
print(f'    {FIG_DIR}/Fig_MIMIC_FullFeatures.pdf/.png')
print('\nDone.')
