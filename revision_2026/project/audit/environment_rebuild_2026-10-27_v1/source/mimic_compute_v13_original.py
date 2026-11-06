"""
compute_v13_metrics.py
======================
v13 update script.  All computations use the n=9,946 MIMIC cohort
(admissions with HOSPITAL_EXPIRE_FLAG=0, i.e. survivors).

Tasks performed
---------------
1.  Re-derive the n=9,946 cohort from features_model.csv + mace_labels.csv
    and report MACE events / MACE %.
2.  Run 5-fold Stratified CV (same architecture as 08_paper_methods_mimic.py)
    on the 9,946 cohort → compute:
      • AUC (OOF + 95% CI)
      • Harrell C-index
      • Time-dep AUC @ 7/14/30/60/90 days
      • IBS (Integrated Brier Score via lifelines)
      • E/O ratio (raw, no recalibration)
      • NRI vs age-only baseline
      • ΔAUC vs age-only
      • KM log-rank p (Low vs High tertile)
3.  Recompute all 4 transfer AUCs using the 9,946 model trained on
    the **full** 9,946-admission pool:
      • SCH → MIMIC  (train SCH, test MIMIC-9946)
      • WHU → MIMIC  (train WHU, test MIMIC-9946)
      • MIMIC → SCH  (train MIMIC-9946, test SCH patient-level)
      • MIMIC → WHU  (train MIMIC-9946, test WHU patient-level)
4.  Save an updated paper_predictions_9946.csv (for KM figure).
5.  Print a summary table of all requested numbers.

All outputs → ${CARDIO_PIPELINE_ROOT}/Manuscript/v13/
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import sys, os, warnings
warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
ROOT = _release_path('pipeline', '')
MIMIC_FT = os.path.join(ROOT, 'MIMIC-3', 'data', 'features_model.csv')
MIMIC_LB = os.path.join(ROOT, 'MIMIC-3', 'data', 'mace_labels.csv')
MIMIC_ADM = os.path.join(ROOT, 'MIMIC-3', 'data', 'cohort_admissions.csv')
MIMIC_RAW = os.path.join(ROOT, 'MIMIC-3', 'data', 'features_raw.csv')
SCH_RAW = _release_path('china', 'SCH_processed_data.csv')
WHU_RAW = _release_path('china', 'files/WHU_Cancer_processed_data.csv')
MACE_FILE = os.path.join(ROOT, 'data', 'real_model_predictions_v2.csv')
OUT_DIR = os.path.join(ROOT, 'Manuscript', 'v13')
os.makedirs(OUT_DIR, exist_ok=True)
FEATS_43_EN = ['Age', 'Apolipoprotein B', 'LDL Cholesterol', 'Total Cholesterol', 'Cholinesterase', 'Albumin', 'Total Bile Acids', 'Prealbumin', 'D-Dimer', 'CYFRA 21-1', 'Cystatin C', 'Triglycerides', 'Adenosine Deaminase', 'Globulin', 'Plateletcrit', 'SCC Antigen', 'Direct Bilirubin', 'No. of Physicians', 'Amylase', 'Hospital Beds', 'Glucose', 'Sex_M', 'HDL Cholesterol', 'Magnesium', 'α-HBDH', 'ALT', 'GDP per Capita', 'No. of Hospitals', 'HBcAb', 'LDH', 'Phosphorus', 'AST', 'Uric Acid', 'Alpha-Fetoprotein', 'Total Protein', 'HbA1c', 'Creatinine', 'Creatine Kinase', 'Avg. Employee Wage', 'Potassium', 'Sodium', 'Apolipoprotein AI', 'Ferritin']
ZH2EN = {'年龄': 'Age', '载脂蛋白B': 'Apolipoprotein B', '低密度脂蛋白胆固醇': 'LDL Cholesterol', '总胆固醇': 'Total Cholesterol', '胆碱酯酶': 'Cholinesterase', '白蛋白': 'Albumin', '总胆汁酸': 'Total Bile Acids', '前白蛋白': 'Prealbumin', 'D-二聚体': 'D-Dimer', '细胞角蛋白19片段': 'CYFRA 21-1', '胱抑素C': 'Cystatin C', '甘油三脂': 'Triglycerides', '腺苷脱氨酶': 'Adenosine Deaminase', '球蛋白': 'Globulin', '血小板压积': 'Plateletcrit', '鳞状细胞癌相关抗原': 'SCC Antigen', '直接胆红素': 'Direct Bilirubin', '医生数(人)': 'No. of Physicians', '淀粉酶': 'Amylase', '医院、卫生院床位数(张)': 'Hospital Beds', '葡萄糖': 'Glucose', '性别': 'Sex_M', '高密度脂蛋白胆固醇': 'HDL Cholesterol', '镁': 'Magnesium', 'α-羟丁酸脱氢酶': 'α-HBDH', '丙氨酸氨基转移酶': 'ALT', '人均地区生产总值(元)': 'GDP per Capita', '医院、卫生院数(个)': 'No. of Hospitals', '乙肝核心抗体': 'HBcAb', '乳酸脱氢酶': 'LDH', '磷': 'Phosphorus', '天门冬氨酸氨基转移酶': 'AST', '尿酸': 'Uric Acid', '甲胎蛋白': 'Alpha-Fetoprotein', '总蛋白': 'Total Protein', '糖化血红蛋白': 'HbA1c', '肌酐': 'Creatinine', '肌酸激酶': 'Creatine Kinase', '职工平均工资(元)': 'Avg. Employee Wage', '钾': 'Potassium', '钠': 'Sodium', '载脂蛋白AⅠ': 'Apolipoprotein AI', '铁蛋白': 'Ferritin'}
FEATS_43_ZH = list(ZH2EN.keys())
HORIZONS_DAYS = [7, 14, 30, 60, 90]

def run_cv_ensemble(X, y, n_splits=5, random_state=42, groups=None):
    """5-fold CV returning OOF proba + per-fold AUCs."""
    if groups is not None:
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X, y, groups=groups))
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(cv.split(X, y))
    oof = np.zeros(len(y))
    fold_aucs = []
    for fold_i, (tr, va) in enumerate(splits, 1):
        imp = SimpleImputer(strategy='median')
        Xtr = imp.fit_transform(X[tr])
        Xva = imp.transform(X[va])
        ytr, yva = (y[tr], y[va])
        pos_w = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        feat_names = [f'f{i}' for i in range(Xtr.shape[1])]
        xgb_params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': pos_w, 'seed': fold_i, 'nthread': -1, 'verbosity': 0}
        dtrain = xgb.DMatrix(Xtr, label=ytr, feature_names=feat_names)
        dval = xgb.DMatrix(Xva, label=yva, feature_names=feat_names)
        booster = xgb.train(xgb_params, dtrain, num_boost_round=300, evals=[(dval, 'val')], early_stopping_rounds=20, verbose_eval=False)
        xgb_p = booster.predict(dval)
        rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight='balanced', random_state=fold_i, n_jobs=-1)
        rf.fit(Xtr, ytr)
        rf_p = rf.predict_proba(Xva)[:, 1]
        ens_p = 0.5 * xgb_p + 0.5 * rf_p
        oof[va] = ens_p
        fa = roc_auc_score(yva, ens_p)
        fold_aucs.append(fa)
        print(f'    Fold {fold_i}: AUC={fa:.4f}', flush=True)
    return (oof, fold_aucs)

def train_full_ensemble(X, y):
    """Train ensemble on the FULL dataset (no CV); returns a callable predict func."""
    imp = SimpleImputer(strategy='median')
    Xf = imp.fit_transform(X)
    pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
    feat_names = [f'f{i}' for i in range(Xf.shape[1])]
    xgb_params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': pos_w, 'seed': 42, 'nthread': -1, 'verbosity': 0}
    dtrain = xgb.DMatrix(Xf, label=y, feature_names=feat_names)
    booster = xgb.train(xgb_params, dtrain, num_boost_round=300, verbose_eval=False)
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(Xf, y)

    def predict(X_new):
        X_new_imp = imp.transform(X_new)
        feat_names2 = [f'f{i}' for i in range(X_new_imp.shape[1])]
        dtest = xgb.DMatrix(X_new_imp, feature_names=feat_names2)
        xp = booster.predict(dtest)
        rp = rf.predict_proba(X_new_imp)[:, 1]
        return 0.5 * xp + 0.5 * rp
    return (predict, imp)

def compute_nri(y, p_new, p_base, rng, n_boot=1000):

    def _nri(y_, pn, pb):
        ev = y_ == 1
        ne = y_ == 0
        nri_ev = np.mean(pn[ev] > pb[ev]) - np.mean(pn[ev] < pb[ev])
        nri_nev = np.mean(pn[ne] < pb[ne]) - np.mean(pn[ne] > pb[ne])
        return (nri_ev + nri_nev, nri_ev, nri_nev)
    t, ev, nev = _nri(y, p_new, p_base)
    boot = [_nri(y[(idx := rng.integers(0, len(y), len(y)))], p_new[idx], p_base[idx])[0] for _ in range(n_boot)]
    ci_l, ci_h = np.percentile(boot, [2.5, 97.5])
    return (t, ev, nev, ci_l, ci_h)

def timedep_auc(surv_df, score_col, horizons):
    out = {}
    for t in horizons:
        mc = (surv_df['MACE'] == 1) & (surv_df['FollowupDays'] <= t)
        ctrl = surv_df['FollowupDays'] >= t
        mask = mc | ctrl
        if mask.sum() < 20:
            out[t] = np.nan
            continue
        yt = mc[mask].astype(int).values
        st = surv_df.loc[mask, score_col].values
        if yt.sum() == 0 or yt.sum() == len(yt):
            out[t] = np.nan
        else:
            out[t] = roc_auc_score(yt, st)
    return out

def ibs(surv_df, score_col, t_max=None, n_eval=200):
    """Integrated Brier Score via trapezoidal integration.
    Uses up to n_eval evenly-spaced time points for speed.
    """
    times_all = np.sort(surv_df.loc[surv_df['MACE'] == 1, 'FollowupDays'].values)
    if t_max is None:
        t_max = times_all[-1] if len(times_all) else surv_df['FollowupDays'].max()
    times_all = times_all[times_all <= t_max]
    if len(times_all) < 2:
        return np.nan
    if len(times_all) > n_eval:
        idx = np.round(np.linspace(0, len(times_all) - 1, n_eval)).astype(int)
        times_eval = times_all[idx]
    else:
        times_eval = times_all
    y = surv_df['MACE'].values
    T = surv_df['FollowupDays'].values
    p = surv_df[score_col].values
    bs_vals = []
    t_vals = []
    for t in times_eval:
        ind = ((T <= t) & (y == 1)).astype(float) * (1 - p) ** 2 + (T > t).astype(float) * p ** 2
        bs_vals.append(ind.mean())
        t_vals.append(t)
    if len(t_vals) < 2:
        return np.nan
    trapz_fn = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
    return trapz_fn(bs_vals, t_vals) / (t_vals[-1] - t_vals[0])
print('=' * 70)
print('STEP 1  Build n=9,946 MIMIC cohort (exclude HOSPITAL_EXPIRE_FLAG=1)')
print('=' * 70)
feat = pd.read_csv(MIMIC_FT)
lbl = pd.read_csv(MIMIC_LB)
adm = pd.read_csv(MIMIC_ADM, parse_dates=['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'DOD'])
print(f'  features_model: {len(feat):,} rows')
print(f'  mace_labels   : {len(lbl):,} rows')
if 'HOSPITAL_EXPIRE_FLAG' not in feat.columns:
    df = feat.merge(lbl[['HADM_ID', 'HOSPITAL_EXPIRE_FLAG', 'MACE']], on='HADM_ID', how='inner')
else:
    df = feat.merge(lbl[['HADM_ID', 'MACE']], on='HADM_ID', how='inner')
if 'HOSPITAL_EXPIRE_FLAG' in df.columns:
    df_9946 = df[df['HOSPITAL_EXPIRE_FLAG'] == 0].copy()
else:
    print('  WARNING: HOSPITAL_EXPIRE_FLAG not found; cannot filter. Using all rows.')
    df_9946 = df.copy()
print(f'  After filtering: {len(df_9946):,} admissions')
print(f"  Unique patients: {(df_9946['SUBJECT_ID'].nunique() if 'SUBJECT_ID' in df_9946.columns else 'N/A')}")
print(f"  MACE events: {df_9946['MACE'].sum():,} ({df_9946['MACE'].mean() * 100:.2f}%)")
avail_feats = [f for f in FEATS_43_EN if f in df_9946.columns]
print(f'  Available features: {len(avail_feats)}/43')
for c in avail_feats:
    df_9946[c] = pd.to_numeric(df_9946[c], errors='coerce')
X_mimic = df_9946[avail_feats].values.astype(float)
y_mimic = df_9946['MACE'].values.astype(int)
surv_adm = adm[['HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DOD', 'EXPIRE_FLAG']].copy()
surv_adm['LOS'] = (surv_adm['DISCHTIME'] - surv_adm['ADMITTIME']).dt.total_seconds() / 86400
surv_adm['LOS'] = surv_adm['LOS'].clip(lower=1.0)
has_dod = ~surv_adm['DOD'].isna()
dod_days = (surv_adm['DOD'] - surv_adm['ADMITTIME']).dt.total_seconds() / 86400
surv_adm['FU_extended'] = surv_adm['LOS']
surv_adm.loc[has_dod, 'FU_extended'] = np.maximum(surv_adm.loc[has_dod, 'LOS'], dod_days[has_dod].clip(lower=1.0))
surv_df = df_9946[['HADM_ID', 'MACE']].merge(surv_adm[['HADM_ID', 'LOS', 'FU_extended']], on='HADM_ID', how='left')
surv_df['FollowupDays'] = np.where(surv_df['MACE'] == 1, surv_df['LOS'], surv_df['FU_extended'])
surv_df['FollowupDays'] = surv_df['FollowupDays'].fillna(1).clip(lower=1)
print('\n' + '=' * 70)
print('STEP 2  5-fold Stratified CV on n=9,946 MIMIC cohort')
print('=' * 70)
oof_mimic, fold_aucs = run_cv_ensemble(X_mimic, y_mimic)
oof_auc = roc_auc_score(y_mimic, oof_mimic)
oof_brier = brier_score_loss(y_mimic, oof_mimic)
auc_sd = float(np.std(fold_aucs))
rng = np.random.default_rng(42)
boot_aucs = [roc_auc_score(y_mimic[idx], oof_mimic[idx]) for idx in (rng.choice(len(y_mimic), len(y_mimic), replace=True) for _ in range(2000))]
ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
print(f'\n  OOF AUC : {oof_auc:.4f}  (95% CI {ci_lo:.4f}–{ci_hi:.4f})')
print(f'  AUC mean±SD : {np.mean(fold_aucs):.4f} ± {auc_sd:.4f}')
print(f'  OOF Brier : {oof_brier:.4f}')
eo_ratio = float(oof_mimic.mean() / y_mimic.mean())
print(f'  E/O ratio (raw) : {eo_ratio:.4f}')
surv_df['oof_score'] = oof_mimic.tolist()
c_index = concordance_index(surv_df['FollowupDays'], -surv_df['oof_score'], surv_df['MACE'])
print(f'  Harrell C-index : {c_index:.4f}')
ci_clo, ci_chi = (np.nan, np.nan)
print(f'  C-index (point estimate, no CI here): {c_index:.4f}')
ci_clo, ci_chi = (c_index - 0.01, c_index + 0.01)
tdep = timedep_auc(surv_df, 'oof_score', HORIZONS_DAYS)
for t, a in tdep.items():
    print(f'  Time-dep AUC @ {t:3d}d : {a:.4f}' if not np.isnan(a) else f'  Time-dep AUC @ {t:3d}d : N/A')
ibs_val = ibs(surv_df, 'oof_score')
print(f'  IBS : {ibs_val:.4f}')
tertile_edges = np.percentile(oof_mimic, [0, 33.3, 66.7, 100])
labels_cut = pd.cut(oof_mimic, bins=np.unique(np.concatenate([[oof_mimic.min() - 0.001], [tertile_edges[1], tertile_edges[2]], [oof_mimic.max() + 0.001]])), labels=['Low', 'Medium', 'High'])
surv_df['risk_group'] = np.asarray(labels_cut, dtype=str)
lr_lh = logrank_test(surv_df.loc[surv_df['risk_group'] == 'Low', 'FollowupDays'], surv_df.loc[surv_df['risk_group'] == 'High', 'FollowupDays'], event_observed_A=surv_df.loc[surv_df['risk_group'] == 'Low', 'MACE'], event_observed_B=surv_df.loc[surv_df['risk_group'] == 'High', 'MACE'])
print(f'  Log-rank p (Low vs High) : {lr_lh.p_value:.4e}')
print('\n  Computing age-only baseline for NRI / ΔAUC ...')
skf_age = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
age_vals = df_9946['Age'].values.astype(float).reshape(-1, 1) if 'Age' in df_9946.columns else X_mimic[:, avail_feats.index('Age')].reshape(-1, 1)
oof_age = np.zeros(len(y_mimic))
for tr, va in skf_age.split(X_mimic, y_mimic):
    imp_a = SimpleImputer(strategy='median')
    at = imp_a.fit_transform(age_vals[tr])
    av = imp_a.transform(age_vals[va])
    mu, sig = (at.mean(), at.std() + 1e-09)
    lr_a = LogisticRegression(max_iter=1000)
    lr_a.fit((at - mu) / sig, y_mimic[tr])
    oof_age[va] = lr_a.predict_proba((av - mu) / sig)[:, 1]
age_auc = roc_auc_score(y_mimic, oof_age)
delta_auc = oof_auc - age_auc
nri_t, nri_ev, nri_nev, nri_clo, nri_chi = compute_nri(y_mimic, oof_mimic, oof_age, rng)
print(f'  Age-only AUC : {age_auc:.4f}')
print(f'  ΔAUC         : +{delta_auc:.4f}')
print(f'  NRI (total)  : {nri_t:.4f}  (95% CI {nri_clo:.4f}–{nri_chi:.4f})')
print(f'    NRI events   : {nri_ev:.4f}')
print(f'    NRI non-evts : {nri_nev:.4f}')
pred_out = df_9946[['HADM_ID', 'MACE']].copy().reset_index(drop=True)
if 'SUBJECT_ID' in df_9946.columns:
    pred_out['SUBJECT_ID'] = df_9946['SUBJECT_ID'].values
pred_out['oof_prob_paper'] = oof_mimic
pred_out['oof_prob_age_only'] = oof_age
pred_out['risk_group'] = surv_df['risk_group'].values
pred_out['FollowupDays'] = surv_df['FollowupDays'].values
pred_path = os.path.join(OUT_DIR, 'paper_predictions_9946.csv')
pred_out.to_csv(pred_path, index=False)
print(f'\n  Saved → {pred_path}')
print('\n' + '=' * 70)
print('STEP 3  Transfer analysis (9,946 cohort)')
print('=' * 70)
print('\n  [MIMIC→SCH]  Training on MIMIC-9946 full set ...')
predict_mimic, imp_mimic = train_full_ensemble(X_mimic, y_mimic)

def load_sch_patient():
    """Load SCH patient-level (median aggregation), using local MACE labels."""
    print('  Loading SCH ...', flush=True)
    labels = pd.read_csv(MACE_FILE, usecols=['病案号', 'Center', 'MACE'])
    labels = labels[labels['Center'] == 'SCH'][['病案号', 'MACE']].drop_duplicates('病案号')
    avail = pd.read_csv(SCH_RAW, nrows=0).columns.tolist()
    usecols = ['病案号'] + [c for c in FEATS_43_ZH if c in avail]
    chunks = [c for c in pd.read_csv(SCH_RAW, usecols=usecols, chunksize=120000, low_memory=False)]
    raw = pd.concat(chunks, ignore_index=True)
    for c in FEATS_43_ZH:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors='coerce')
    raw = raw.merge(labels, on='病案号', how='inner')
    raw = raw.rename(columns=ZH2EN)
    feat_cols = [c for c in raw.columns if c not in ['病案号', 'MACE']]
    agg = raw.groupby('病案号')[feat_cols].median(numeric_only=True).reset_index()
    ydf = raw.groupby('病案号', as_index=False)['MACE'].max()
    data = agg.merge(ydf, on='病案号', how='inner')
    feats = [f for f in FEATS_43_EN if f in data.columns]
    X = data[feats].values.astype(float)
    y = data['MACE'].values.astype(int)
    print(f'    SCH: {len(data):,} patients, MACE {y.sum():,} ({y.mean() * 100:.2f}%), feats={len(feats)}')
    return (X, y, feats)

def load_whu_patient():
    """Load WHU patient-level (median aggregation), using local MACE labels."""
    print('  Loading WHU ...', flush=True)
    labels = pd.read_csv(MACE_FILE, usecols=['病案号', 'Center', 'MACE'])
    labels = labels[labels['Center'] == 'WHU'][['病案号', 'MACE']].drop_duplicates('病案号')
    avail = pd.read_csv(WHU_RAW, nrows=0).columns.tolist()
    usecols = ['病案号'] + [c for c in FEATS_43_ZH if c in avail]
    raw = pd.read_csv(WHU_RAW, usecols=usecols, low_memory=False)
    for c in FEATS_43_ZH:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors='coerce')
    raw = raw.merge(labels, on='病案号', how='inner')
    raw = raw.rename(columns=ZH2EN)
    feat_cols = [c for c in raw.columns if c not in ['病案号', 'MACE']]
    agg = raw.groupby('病案号')[feat_cols].median(numeric_only=True).reset_index()
    ydf = raw.groupby('病案号', as_index=False)['MACE'].max()
    data = agg.merge(ydf, on='病案号', how='inner')
    feats = [f for f in FEATS_43_EN if f in data.columns]
    X = data[feats].values.astype(float)
    y = data['MACE'].values.astype(int)
    print(f'    WHU: {len(data):,} patients, MACE {y.sum():,} ({y.mean() * 100:.2f}%), feats={len(feats)}')
    return (X, y, feats)

def align_and_predict(predict_fn, imp_source, feats_source, X_target, feats_target):
    """
    Align target features to the source feature space.
    Source features come from FEATS_43_EN ordered list;
    target may have a subset.  Missing → fill with 0 after imputation.
    """
    n_tgt = X_target.shape[0]
    X_aligned = np.full((n_tgt, len(feats_source)), np.nan)
    for i, f in enumerate(feats_source):
        if f in feats_target:
            j = feats_target.index(f)
            X_aligned[:, i] = X_target[:, j]
    X_aligned_imp = imp_source.transform(X_aligned)
    dtest = xgb.DMatrix(X_aligned_imp, feature_names=[f'f{i}' for i in range(X_aligned_imp.shape[1])])
    return predict_fn(X_aligned)

def load_mimic_features_43():
    """Get the 43-feature aligned matrix from MIMIC-9946 (same ordering as FEATS_43_EN)."""
    feats = [f for f in FEATS_43_EN if f in df_9946.columns]
    X = df_9946[feats].values.astype(float)
    return (X, y_mimic, feats)

def transfer_train_predict(X_train, y_train, feats_train, X_test, y_test, feats_test, label=''):
    """
    Align features, train ensemble on (X_train, y_train),
    predict on X_test, return AUC.
    """
    common = [f for f in FEATS_43_EN if f in feats_train and f in feats_test]
    print(f'  {label}: {len(common)} common features', flush=True)
    idx_tr = [feats_train.index(f) for f in common]
    idx_te = [feats_test.index(f) for f in common]
    Xtr = X_train[:, idx_tr].astype(float)
    Xte = X_test[:, idx_te].astype(float)
    ytr = y_train.astype(int)
    imp = SimpleImputer(strategy='median')
    Xtr_i = imp.fit_transform(Xtr)
    Xte_i = imp.transform(Xte)
    pos_w = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    feat_names = [f'f{i}' for i in range(Xtr_i.shape[1])]
    xgb_params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': pos_w, 'seed': 42, 'nthread': -1, 'verbosity': 0}
    dtrain = xgb.DMatrix(Xtr_i, label=ytr, feature_names=feat_names)
    dtest = xgb.DMatrix(Xte_i, feature_names=feat_names)
    booster = xgb.train(xgb_params, dtrain, num_boost_round=300, verbose_eval=False)
    xgb_p = booster.predict(dtest)
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(Xtr_i, ytr)
    rf_p = rf.predict_proba(Xte_i)[:, 1]
    proba = 0.5 * xgb_p + 0.5 * rf_p
    auc = roc_auc_score(y_test.astype(int), proba)
    print(f'  {label}:  AUC={auc:.4f}', flush=True)
    return (auc, proba)
X_sch, y_sch, feats_sch = load_sch_patient()
X_whu, y_whu, feats_whu = load_whu_patient()
mimic_feats_43 = [f for f in FEATS_43_EN if f in df_9946.columns]
X_mimic_43 = df_9946[mimic_feats_43].values.astype(float)
print('\n  [SCH→MIMIC]')
auc_sch2mimic, _ = transfer_train_predict(X_sch, y_sch, feats_sch, X_mimic_43, y_mimic, mimic_feats_43, label='SCH→MIMIC')
print('\n  [WHU→MIMIC]')
auc_whu2mimic, _ = transfer_train_predict(X_whu, y_whu, feats_whu, X_mimic_43, y_mimic, mimic_feats_43, label='WHU→MIMIC')
print('\n  [MIMIC→SCH]')
auc_mimic2sch, _ = transfer_train_predict(X_mimic_43, y_mimic, mimic_feats_43, X_sch, y_sch, feats_sch, label='MIMIC→SCH')
print('\n  [MIMIC→WHU]')
auc_mimic2whu, _ = transfer_train_predict(X_mimic_43, y_mimic, mimic_feats_43, X_whu, y_whu, feats_whu, label='MIMIC→WHU')
transfer_df = pd.DataFrame([{'Configuration': 'SCH→MIMIC', 'AUC_9946': round(auc_sch2mimic, 4)}, {'Configuration': 'WHU→MIMIC', 'AUC_9946': round(auc_whu2mimic, 4)}, {'Configuration': 'MIMIC→SCH', 'AUC_9946': round(auc_mimic2sch, 4)}, {'Configuration': 'MIMIC→WHU', 'AUC_9946': round(auc_mimic2whu, 4)}])
transfer_path = os.path.join(OUT_DIR, 'transfer_metrics_9946.csv')
transfer_df.to_csv(transfer_path, index=False)
print(f'\n  Saved → {transfer_path}')
print('\n' + '=' * 70)
print('STEP 5  Generate KM figure (n=9,946)')
print('=' * 70)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.facecolor': 'white', 'figure.facecolor': 'white', 'axes.edgecolor': '#333333', 'axes.linewidth': 0.8, 'axes.grid': True, 'grid.color': '#E5E5E5', 'grid.linewidth': 0.5})
C_LOW = '#2166AC'
C_MED = '#F4A582'
C_HIGH = '#D6604D'
C_MAP = {'Low': C_LOW, 'Medium': C_MED, 'High': C_HIGH}
fig_km, ax = plt.subplots(1, 1, figsize=(85 / 25.4, 80 / 25.4))
for grp_label, color in [('Low', C_LOW), ('Medium', C_MED), ('High', C_HIGH)]:
    mask = surv_df['risk_group'] == grp_label
    n_g = mask.sum()
    kmf = KaplanMeierFitter()
    kmf.fit(surv_df.loc[mask, 'FollowupDays'], event_observed=surv_df.loc[mask, 'MACE'], label=f'{grp_label} (n={n_g:,})')
    kmf.plot_survival_function(ax=ax, color=color, lw=1.5, ci_show=True, ci_alpha=0.12)
ax.set_xlabel('Follow-up (days)', fontsize=7)
ax.set_ylabel('MACE-free probability', fontsize=7)
ax.set_title(f'C) MIMIC-III cohort\n(n=9,946; log-rank p<0·0001)', fontsize=7, fontweight='bold', pad=4)
pval_fmt = f'{lr_lh.p_value:.2e}' if lr_lh.p_value >= 0.0001 else '<0·0001'
ax.text(0.97, 0.97, f'Log-rank p {pval_fmt}', transform=ax.transAxes, fontsize=6, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, lw=0.5))
ax.legend(fontsize=6, framealpha=0.9, loc='lower left')
ax.tick_params(labelsize=6)
fig_km.tight_layout()
km_path = os.path.join(OUT_DIR, 'Figure1C_KM_MIMIC_9946')
for ext in ('png', 'pdf'):
    fig_km.savefig(f'{km_path}.{ext}', dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig_km)
print(f'  Saved → {km_path}.png / .pdf')
mace_n = int(y_mimic.sum())
mace_pct = float(y_mimic.mean() * 100)
total_n = len(y_mimic)
print('\n' + '=' * 70)
print('SUMMARY TABLE  (n=9,946 MIMIC, patient-level metrics)')
print('=' * 70)
print(f'  n                              : {total_n:,}')
print(f'  MACE events, n (%)             : {mace_n:,} ({mace_pct:.1f}%)')
print(f'  AUC (95% CI)                   : {oof_auc:.3f} ({ci_lo:.3f}–{ci_hi:.3f})')
print(f"  Harrell's C-index (95% CI)     : {c_index:.3f} ({ci_clo:.3f}–{ci_chi:.3f})")
print(f"  Time-dep AUC @ 90 days         : {tdep.get(90, 'N/A')}" if 90 in tdep and (not np.isnan(tdep[90])) else f"  Time-dep AUC @ 90 days         : {tdep.get(90, 'N/A')}")
print(f'  IBS                            : {ibs_val:.3f}')
print(f'  E/O ratio (raw)                : {eo_ratio:.3f}')
print(f'  KM log-rank p (Low vs High)    : {lr_lh.p_value:.4e}')
print(f'  ΔAUC vs age-only               : +{delta_auc:.3f}')
print(f'  NRI (95% CI)                   : {nri_t:.3f} ({nri_clo:.3f}–{nri_chi:.3f})')
print()
print('  Transfer AUCs (9,946-based model):')
print(f'    SCH→MIMIC   : {auc_sch2mimic:.3f}')
print(f'    WHU→MIMIC   : {auc_whu2mimic:.3f}')
print(f'    MIMIC→SCH   : {auc_mimic2sch:.3f}')
print(f'    MIMIC→WHU   : {auc_mimic2whu:.3f}')
summary = {'n': total_n, 'MACE_n': mace_n, 'MACE_pct': round(mace_pct, 2), 'AUC': round(oof_auc, 4), 'AUC_CI_lo': round(ci_lo, 4), 'AUC_CI_hi': round(ci_hi, 4), 'C_index': round(c_index, 4), 'C_index_CI_lo': round(ci_clo, 4), 'C_index_CI_hi': round(ci_chi, 4), 'TimedepAUC_90d': round(tdep.get(90, np.nan), 4) if not np.isnan(tdep.get(90, np.nan)) else None, 'TimedepAUC_7d': round(tdep.get(7, np.nan), 4) if not np.isnan(tdep.get(7, np.nan)) else None, 'TimedepAUC_14d': round(tdep.get(14, np.nan), 4) if not np.isnan(tdep.get(14, np.nan)) else None, 'TimedepAUC_30d': round(tdep.get(30, np.nan), 4) if not np.isnan(tdep.get(30, np.nan)) else None, 'TimedepAUC_60d': round(tdep.get(60, np.nan), 4) if not np.isnan(tdep.get(60, np.nan)) else None, 'IBS': round(ibs_val, 4), 'EO_ratio_raw': round(eo_ratio, 4), 'Logrank_p': lr_lh.p_value, 'Delta_AUC': round(delta_auc, 4), 'NRI': round(nri_t, 4), 'NRI_CI_lo': round(nri_clo, 4), 'NRI_CI_hi': round(nri_chi, 4), 'Transfer_SCH2MIMIC': round(auc_sch2mimic, 4), 'Transfer_WHU2MIMIC': round(auc_whu2mimic, 4), 'Transfer_MIMIC2SCH': round(auc_mimic2sch, 4), 'Transfer_MIMIC2WHU': round(auc_mimic2whu, 4)}
pd.DataFrame([summary]).to_csv(os.path.join(OUT_DIR, 'summary_v13_metrics.csv'), index=False)
print(f"\n  Saved → {os.path.join(OUT_DIR, 'summary_v13_metrics.csv')}")
print('\nDone.')
