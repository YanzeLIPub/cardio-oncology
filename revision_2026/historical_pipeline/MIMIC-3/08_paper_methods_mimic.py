"""
08_paper_methods_mimic.py
══════════════════════════════════════════════════════════════════════════════
Apply the EXACT methodology from Methods_LancetDigitalHealth.docx to MIMIC-III
cardio-oncology cohort.

Key methodology fidelity points (vs earlier scripts):
  ① Feature aggregation: 4 stats (mean, most-recent, max, min) per lab → 92 lab
     features + Age, Sex_M, Admission_has_cancer, Admission_has_cardiac = 96 total
  ② Model: XGBoost + Random Forest 50:50 (EnsembleModel; paper exact config)
  ③ CV: StratifiedKFold(n_splits=5), no patient grouping (consistent with paper)
  ④ Full evaluation suite:
       • Discrimination  : OOF AUC + 95% CI (bootstrap), per-fold AUC + SD
       • Calibration     : calibration curve, slope/intercept, E/O ratio,
                           Hosmer-Lemeshow chi² (8 df)
       • DCA             : net benefit curves vs treat-all / treat-none
       • Survival        : C-index, Cox PH (HR+CI), time-dep AUC @ 7/14/30d,
                           Kaplan-Meier by risk tertile + log-rank
       • NRI             : vs age-only logistic regression (1 000 bootstrap CIs)
       • Feature imp.    : XGB gain + RF Gini, normalised average across folds
  ⑤ Cross-cohort transfer: reload SCH→MIMIC / WHU→MIMIC predictions from
     06_separate_transfer.py and apply the same full metrics suite

Outputs → MIMIC-3/results/
  paper_metrics_within_mimic.csv
  paper_metrics_transfer.csv
  paper_feature_importance.csv
  paper_predictions.csv
  figures/Fig_Paper_Main.pdf/.png          (ROC + Calibration + DCA + KM)
  figures/Fig_Paper_Features.pdf/.png     (Feature importance)
  figures/Fig_Paper_NRI.pdf/.png          (NRI + Transfer comparison)
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os, sys, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'MIMIC-3', 'data')
RES_DIR = os.path.join(ROOT, 'MIMIC-3', 'results')
FIG_DIR = os.path.join(RES_DIR, 'figures')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
sys.path.insert(0, ROOT)
from models.ensemble import EnsembleModel
PAL = {'mimic': '#1a6faf', 'sch': '#e07b39', 'whu': '#3cb371', 'age': '#9b59b6', 'treat_all': '#888', 'treat_none': '#ccc'}
HORIZONS_DAYS = [7, 14, 30, 60]
print('=' * 70)
print('STEP 1  Building 4-stat feature matrix  (mean / last / max / min)')
print('=' * 70)
raw = pd.read_csv(os.path.join(DATA_DIR, 'features_raw.csv'))
labs = pd.read_csv(os.path.join(DATA_DIR, 'mace_labels.csv'))
adm = pd.read_csv(os.path.join(DATA_DIR, 'cohort_admissions.csv'), parse_dates=['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'DOD'])
print(f"  features_raw: {len(raw):,} lab events, {raw['HADM_ID'].nunique():,} admissions, {raw['Feature'].nunique()} features")
feats_list = sorted(raw['Feature'].unique())
raw = raw.sort_values('CHARTTIME')
print('  Computing 4-stat aggregation (vectorised) ...')
grp = raw.groupby(['HADM_ID', 'Feature'])['VALUENUM']
agg_mean = grp.mean().rename('mean')
agg_last = grp.last().rename('last')
agg_max = grp.max().rename('max')
agg_min = grp.min().rename('min')
agg_long = pd.concat([agg_mean, agg_last, agg_max, agg_min], axis=1).reset_index()
agg_long_melted = agg_long.melt(id_vars=['HADM_ID', 'Feature'], value_vars=['mean', 'last', 'max', 'min'], var_name='stat', value_name='value')
agg_long_melted['col'] = agg_long_melted['Feature'] + '__' + agg_long_melted['stat']
lab_wide = agg_long_melted.pivot_table(index='HADM_ID', columns='col', values='value', aggfunc='first').reset_index()
lab_wide.columns.name = None
print(f'  Lab wide matrix: {lab_wide.shape}')
demo_cols = ['HADM_ID', 'Age', 'Sex_M', 'Admission_has_cancer', 'Admission_has_cardiac']
demo = adm[demo_cols].copy()
feat_df = labs[['HADM_ID', 'MACE']].merge(lab_wide, on='HADM_ID', how='left')
feat_df = feat_df.merge(demo, on='HADM_ID', how='left')
FEATURE_COLS = [c for c in feat_df.columns if c not in ('HADM_ID', 'MACE')]
print(f'  Total features: {len(FEATURE_COLS)}  ({len(feats_list)} labs × 4 stats + 4 demo)')
print(f"  Admissions: {len(feat_df):,}  |  MACE+: {feat_df['MACE'].sum():,} ({feat_df['MACE'].mean() * 100:.1f}%)")
X_raw = feat_df[FEATURE_COLS].values.astype(float)
y_all = feat_df['MACE'].values.astype(int)
print('\n' + '=' * 70)
print('STEP 2  5-fold Stratified CV  (XGBoost + Random Forest 50:50)')
print('=' * 70)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = np.zeros(len(y_all))
fold_aucs = []
fold_briers = []
xgb_gain_accum = np.zeros(len(FEATURE_COLS))
rf_gini_accum = np.zeros(len(FEATURE_COLS))
for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_raw, y_all), 1):
    X_tr, X_vl = (X_raw[tr_idx], X_raw[vl_idx])
    y_tr, y_vl = (y_all[tr_idx], y_all[vl_idx])
    imp = SimpleImputer(strategy='median')
    X_tr_imp = imp.fit_transform(X_tr)
    X_vl_imp = imp.transform(X_vl)
    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    xgb_params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': pos_w, 'seed': fold, 'nthread': -1, 'verbosity': 0}
    dtrain = xgb.DMatrix(X_tr_imp, label=y_tr, feature_names=FEATURE_COLS)
    dval = xgb.DMatrix(X_vl_imp, label=y_vl, feature_names=FEATURE_COLS)
    booster = xgb.train(xgb_params, dtrain, num_boost_round=300, evals=[(dval, 'val')], early_stopping_rounds=20, verbose_eval=False)
    xgb_pred = booster.predict(dval)
    xgb_feat_names = [f.replace(' ', '_').replace('-', '_').replace('/', '_') for f in FEATURE_COLS]
    dtrain = xgb.DMatrix(X_tr_imp, label=y_tr, feature_names=xgb_feat_names)
    dval = xgb.DMatrix(X_vl_imp, label=y_vl, feature_names=xgb_feat_names)
    booster = xgb.train(xgb_params, dtrain, num_boost_round=300, evals=[(dval, 'val')], early_stopping_rounds=20, verbose_eval=False)
    xgb_pred = booster.predict(dval)
    fscore = booster.get_score(importance_type='gain')
    xgb_imp = np.zeros(len(FEATURE_COLS))
    feat_idx_map = {f: i for i, f in enumerate(xgb_feat_names)}
    for fname, score in fscore.items():
        if fname in feat_idx_map:
            xgb_imp[feat_idx_map[fname]] = score
    xgb_total = xgb_imp.sum()
    if xgb_total > 0:
        xgb_gain_accum += xgb_imp / xgb_total
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight='balanced', random_state=fold, n_jobs=-1)
    rf.fit(X_tr_imp, y_tr)
    rf_pred = rf.predict_proba(X_vl_imp)[:, 1]
    rf_imp = rf.feature_importances_
    rf_total = rf_imp.sum()
    if rf_total > 0:
        rf_gini_accum += rf_imp / rf_total
    ens_pred = 0.5 * xgb_pred + 0.5 * rf_pred
    oof_proba[vl_idx] = ens_pred
    fold_aucs.append(roc_auc_score(y_vl, ens_pred))
    fold_briers.append(brier_score_loss(y_vl, ens_pred))
    print(f'  Fold {fold}: AUC={fold_aucs[-1]:.4f}  Brier={fold_briers[-1]:.4f}')
xgb_gain_avg = xgb_gain_accum / 5
rf_gini_avg = rf_gini_accum / 5
avg_imp = (xgb_gain_avg + rf_gini_avg) / 2
oof_auc = roc_auc_score(y_all, oof_proba)
oof_brier = brier_score_loss(y_all, oof_proba)
auc_sd = np.std(fold_aucs)
rng = np.random.default_rng(42)
boot_aucs = [roc_auc_score(y_all[idx], oof_proba[idx]) for idx in (rng.choice(len(y_all), len(y_all), replace=True) for _ in range(2000))]
ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
print(f'\n  OOF AUC : {oof_auc:.4f}  (95% CI {ci_lo:.4f}–{ci_hi:.4f})')
print(f'  AUC mean±SD across folds : {np.mean(fold_aucs):.4f} ± {auc_sd:.4f}')
print(f'  OOF Brier : {oof_brier:.4f}')
ckpt_path = os.path.join(RES_DIR, '_ckpt_step2.npz')
np.savez(ckpt_path, oof_proba=oof_proba, y_all=y_all, xgb_gain_accum=xgb_gain_accum, rf_gini_accum=rf_gini_accum, oof_age=np.zeros(len(y_all)))
print('\n' + '=' * 70)
print('STEP 3  Calibration analysis')
print('=' * 70)
frac_obs, mean_pred = calibration_curve(y_all, oof_proba, n_bins=10, strategy='quantile')
eps = 1e-06
logit_pred = np.log(np.clip(oof_proba, eps, 1 - eps) / (1 - np.clip(oof_proba, eps, 1 - eps)))
lr_cal = LogisticRegression(fit_intercept=True, max_iter=1000)
lr_cal.fit(logit_pred.reshape(-1, 1), y_all)
cal_slope = float(lr_cal.coef_[0][0])
cal_intercept = float(lr_cal.intercept_[0])
eo_ratio = oof_proba.mean() / y_all.mean()
n_bins = 10
bin_edges = np.percentile(oof_proba, np.linspace(0, 100, n_bins + 1))
bin_edges[0] -= 1e-08
bin_edges[-1] += 1e-08
bins = np.digitize(oof_proba, bin_edges) - 1
bins = np.clip(bins, 0, n_bins - 1)
hl_chi2 = 0.0
for b in range(n_bins):
    mask = bins == b
    n_b = mask.sum()
    o_b = y_all[mask].sum()
    e_b = oof_proba[mask].sum()
    if e_b > 0 and n_b - e_b > 0:
        hl_chi2 += (o_b - e_b) ** 2 / (e_b * (1 - e_b / n_b) + 1e-09)
hl_p = 1 - stats.chi2.cdf(hl_chi2, df=8)
print(f'  Calibration slope    : {cal_slope:.3f}   (ideal = 1.0)')
print(f'  Calibration intercept: {cal_intercept:.3f}  (ideal = 0.0)')
print(f'  E/O ratio            : {eo_ratio:.3f}  (ideal = 1.0)')
print(f'  Hosmer-Lemeshow χ²   : {hl_chi2:.2f}  p={hl_p:.4f}  (df=8)')
print('\n' + '=' * 70)
print('STEP 4  Decision Curve Analysis')
print('=' * 70)
thresholds = np.arange(0.01, 0.8, 0.01)
n = len(y_all)
prev = y_all.mean()
nb_model = []
nb_treat_all = []
nb_treat_non = []
for t in thresholds:
    tp = ((oof_proba >= t) & (y_all == 1)).sum()
    fp = ((oof_proba >= t) & (y_all == 0)).sum()
    nb_model.append((tp - fp * t / (1 - t)) / n)
    nb_treat_all.append(prev - (1 - prev) * t / (1 - t))
    nb_treat_non.append(0.0)
nb_model = np.array(nb_model)
nb_treat_all = np.array(nb_treat_all)
fpr_arr, tpr_arr, thresh_arr = roc_curve(y_all, oof_proba)
youden = tpr_arr - fpr_arr
opt_idx = np.argmax(youden)
opt_thresh = thresh_arr[opt_idx]
opt_sens = tpr_arr[opt_idx]
opt_spec = 1 - fpr_arr[opt_idx]
print(f'  Optimal threshold (Youden): {opt_thresh:.3f}  Sens={opt_sens:.3f}  Spec={opt_spec:.3f}')
print('\n' + '=' * 70)
print('STEP 5  Survival analysis')
print('=' * 70)
surv = adm[['HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DOD', 'EXPIRE_FLAG']].copy()
surv['LOS'] = (surv['DISCHTIME'] - surv['ADMITTIME']).dt.total_seconds() / 86400
surv['LOS'] = surv['LOS'].clip(lower=1.0)
surv['FU_extended'] = surv['LOS']
has_dod = ~surv['DOD'].isna()
dod_days = (surv['DOD'] - surv['ADMITTIME']).dt.total_seconds() / 86400
surv.loc[has_dod, 'FU_extended'] = np.maximum(surv.loc[has_dod, 'LOS'], dod_days[has_dod].clip(lower=1.0))
surv_df = feat_df[['HADM_ID', 'MACE']].merge(surv[['HADM_ID', 'LOS', 'FU_extended']], on='HADM_ID', how='left')
surv_df['oof_score'] = oof_proba
surv_df['FollowupDays'] = np.where(surv_df['MACE'] == 1, surv_df['LOS'], surv_df['FU_extended'])
surv_df['FollowupDays'] = surv_df['FollowupDays'].fillna(1).clip(lower=1)
c_index = concordance_index(surv_df['FollowupDays'], -surv_df['oof_score'], surv_df['MACE'])
print(f'  Harrell C-index: {c_index:.4f}')
score_std = (surv_df['oof_score'] - surv_df['oof_score'].mean()) / surv_df['oof_score'].std()
cox_df = pd.DataFrame({'T': surv_df['FollowupDays'], 'E': surv_df['MACE'], 'score': score_std})
cox_df = cox_df.dropna()
cph = CoxPHFitter()
cph.fit(cox_df, duration_col='T', event_col='E', formula='score')
hr_val = float(np.exp(cph.params_['score']))
hr_lo = float(np.exp(cph.confidence_intervals_.loc['score', '95% lower-bound']))
hr_hi = float(np.exp(cph.confidence_intervals_.loc['score', '95% upper-bound']))
cox_p = float(cph.summary['p']['score'])
print(f'  Cox HR (per 1 SD of score): {hr_val:.3f} (95% CI {hr_lo:.3f}–{hr_hi:.3f})  p={cox_p:.4e}')
timedep_aucs = {}
for t_days in HORIZONS_DAYS:
    mask_case = (surv_df['MACE'] == 1) & (surv_df['FollowupDays'] <= t_days)
    mask_ctrl = surv_df['FollowupDays'] >= t_days
    mask = mask_case | mask_ctrl
    if mask.sum() < 20:
        timedep_aucs[t_days] = np.nan
        continue
    y_t = mask_case[mask].astype(int).values
    s_t = surv_df.loc[mask, 'oof_score'].values
    if y_t.sum() == 0 or y_t.sum() == len(y_t):
        timedep_aucs[t_days] = np.nan
    else:
        timedep_aucs[t_days] = roc_auc_score(y_t, s_t)
for t, a in timedep_aucs.items():
    print(f'  Time-dep AUC @ {t:2d}d : {a:.4f}' if a is not np.nan else f'  Time-dep AUC @ {t:2d}d : N/A')
tertile_edges = np.percentile(oof_proba, [0, 33.3, 66.7, 100])
tertile_labels = pd.cut(oof_proba, bins=np.unique(np.concatenate([[oof_proba.min() - 0.001], [tertile_edges[1], tertile_edges[2]], [oof_proba.max() + 0.001]])), labels=['Low', 'Intermediate', 'High'])
surv_df['risk_group'] = np.asarray(tertile_labels, dtype=str)
kmf_dict = {}
logrank_results = {}
groups = ['Low', 'Intermediate', 'High']
for g in groups:
    mask = surv_df['risk_group'] == g
    kmf = KaplanMeierFitter()
    kmf.fit(surv_df.loc[mask, 'FollowupDays'], event_observed=surv_df.loc[mask, 'MACE'], label=f'{g} risk')
    kmf_dict[g] = kmf
lr_lh = logrank_test(surv_df.loc[surv_df['risk_group'] == 'Low', 'FollowupDays'], surv_df.loc[surv_df['risk_group'] == 'High', 'FollowupDays'], event_observed_A=surv_df.loc[surv_df['risk_group'] == 'Low', 'MACE'], event_observed_B=surv_df.loc[surv_df['risk_group'] == 'High', 'MACE'])
print(f'  Log-rank p (Low vs High): {lr_lh.p_value:.4e}')
print('\n' + '=' * 70)
print('STEP 6  Net Reclassification Index  (vs age-only baseline)')
print('=' * 70)
age_vals = feat_df['Age'].values.astype(float).reshape(-1, 1)
oof_age = np.zeros(len(y_all))
for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_raw, y_all)):
    imp_age = SimpleImputer(strategy='median')
    age_tr = imp_age.fit_transform(age_vals[tr_idx])
    age_vl = imp_age.transform(age_vals[vl_idx])
    mu, sig = (age_tr.mean(), age_tr.std() + 1e-09)
    age_tr_std = (age_tr - mu) / sig
    age_vl_std = (age_vl - mu) / sig
    lr_age = LogisticRegression(fit_intercept=True, max_iter=1000)
    lr_age.fit(age_tr_std, y_all[tr_idx])
    oof_age[vl_idx] = lr_age.predict_proba(age_vl_std)[:, 1]
age_auc = roc_auc_score(y_all, oof_age)
print(f'  Age-only OOF AUC: {age_auc:.4f}')

def compute_nri(y, p_new, p_base):
    """Continuous NRI."""
    events = y == 1
    nonevents = y == 0
    nri_events = np.mean(p_new[events] > p_base[events]) - np.mean(p_new[events] < p_base[events])
    nri_nonevents = np.mean(p_new[nonevents] < p_base[nonevents]) - np.mean(p_new[nonevents] > p_base[nonevents])
    return (nri_events + nri_nonevents, nri_events, nri_nonevents)
nri_total, nri_ev, nri_nev = compute_nri(y_all, oof_proba, oof_age)
boot_nri = []
for _ in range(1000):
    idx = rng.integers(0, len(y_all), len(y_all))
    bt, _, _ = compute_nri(y_all[idx], oof_proba[idx], oof_age[idx])
    boot_nri.append(bt)
nri_ci_lo, nri_ci_hi = np.percentile(boot_nri, [2.5, 97.5])
print(f'  NRI (continuous)      : {nri_total:.4f}  (95% CI {nri_ci_lo:.4f}–{nri_ci_hi:.4f})')
print(f'    NRI (events)        : {nri_ev:.4f}')
print(f'    NRI (non-events)    : {nri_nev:.4f}')
print(f'  Delta AUC vs age-only : +{oof_auc - age_auc:.4f}')
print('\n' + '=' * 70)
print('STEP 7  Feature importance  (XGB gain + RF Gini, normalised average)')
print('=' * 70)
fi_df = pd.DataFrame({'Feature': FEATURE_COLS, 'XGB_gain': xgb_gain_avg, 'RF_gini': rf_gini_avg, 'Avg_imp': avg_imp}).sort_values('Avg_imp', ascending=False).reset_index(drop=True)
print('  Top 20 features:')
for _, row in fi_df.head(20).iterrows():
    print(f"    {row['Feature']:<45s}  XGB={row['XGB_gain']:.4f}  RF={row['RF_gini']:.4f}  Avg={row['Avg_imp']:.4f}")
print('\n' + '=' * 70)
print('STEP 8  Cross-cohort transfer full metrics  (SCH→MIMIC / WHU→MIMIC)')
print('=' * 70)
pred06_path = os.path.join(RES_DIR, 'separate_transfer_predictions.csv')
transfer_metrics = {}
if os.path.exists(pred06_path):
    pred06 = pd.read_csv(pred06_path)
    y_t = pred06['mace_label'].values.astype(int)
    for tag, col in [('SCH→MIMIC', 'prob_SCH_raw'), ('WHU→MIMIC', 'prob_WHU_raw')]:
        p_t = pred06[col].values.astype(float)
        nan_mask = ~np.isnan(p_t)
        if nan_mask.sum() < 100:
            print(f'  {tag}: insufficient predictions, skipped')
            continue
        p_t = p_t[nan_mask]
        y_ = y_t[nan_mask]
        t_auc = roc_auc_score(y_, p_t)
        t_brier = brier_score_loss(y_, p_t)
        logit_t = np.log(np.clip(p_t, eps, 1 - eps) / (1 - np.clip(p_t, eps, 1 - eps)))
        lr_ct = LogisticRegression(fit_intercept=True, max_iter=1000)
        lr_ct.fit(logit_t.reshape(-1, 1), y_)
        t_slope = float(lr_ct.coef_[0][0])
        t_int = float(lr_ct.intercept_[0])
        t_eo = p_t.mean() / y_.mean()
        transfer_metrics[tag] = {'AUC': t_auc, 'Brier': t_brier, 'Cal_slope': t_slope, 'Cal_intercept': t_int, 'EO_ratio': t_eo}
        print(f'  {tag}:  AUC={t_auc:.4f}  Brier={t_brier:.4f}  Slope={t_slope:.3f}  Intercept={t_int:.3f}  E/O={t_eo:.3f}')
else:
    print('  06_separate_transfer_predictions.csv not found; skipping transfer metrics.')
print('\n' + '=' * 70)
print('STEP 9  Saving CSV outputs')
print('=' * 70)
within_metrics = pd.DataFrame([{'Model': 'MIMIC within-CV (paper config)', 'Features': f'{len(FEATURE_COLS)} (4-stat × 23 labs + 4 demo)', 'CV': '5-fold StratifiedKF', 'OOF_AUC': round(oof_auc, 4), 'AUC_CI_lo': round(ci_lo, 4), 'AUC_CI_hi': round(ci_hi, 4), 'AUC_mean_folds': round(np.mean(fold_aucs), 4), 'AUC_SD_folds': round(auc_sd, 4), 'Brier': round(oof_brier, 4), 'Cal_slope': round(cal_slope, 4), 'Cal_intercept': round(cal_intercept, 4), 'EO_ratio': round(eo_ratio, 4), 'HL_chi2': round(hl_chi2, 3), 'HL_p': round(hl_p, 4), 'C_index': round(c_index, 4), 'Cox_HR': round(hr_val, 3), 'Cox_HR_CI_lo': round(hr_lo, 3), 'Cox_HR_CI_hi': round(hr_hi, 3), 'Cox_p': cox_p, 'Logrank_Low_vs_High': round(lr_lh.p_value, 6), 'NRI': round(nri_total, 4), 'NRI_CI_lo': round(nri_ci_lo, 4), 'NRI_CI_hi': round(nri_ci_hi, 4), 'Age_baseline_AUC': round(age_auc, 4), 'Delta_AUC': round(oof_auc - age_auc, 4), **{f'TimedepAUC_{t}d': round(v, 4) if v is not None and (not np.isnan(v)) else None for t, v in timedep_aucs.items()}}])
within_metrics.to_csv(os.path.join(RES_DIR, 'paper_metrics_within_mimic.csv'), index=False)
print('  Saved → paper_metrics_within_mimic.csv')
if transfer_metrics:
    t_rows = [{'Configuration': k, **v} for k, v in transfer_metrics.items()]
    pd.DataFrame(t_rows).to_csv(os.path.join(RES_DIR, 'paper_metrics_transfer.csv'), index=False)
    print('  Saved → paper_metrics_transfer.csv')
fi_df.to_csv(os.path.join(RES_DIR, 'paper_feature_importance.csv'), index=False)
print('  Saved → paper_feature_importance.csv')
pred_out = feat_df[['HADM_ID', 'MACE']].copy()
pred_out['oof_prob_paper'] = oof_proba
pred_out['oof_prob_age_only'] = oof_age
pred_out['risk_group'] = surv_df['risk_group'].values
pred_out['FollowupDays'] = surv_df['FollowupDays'].values
pred_out.to_csv(os.path.join(RES_DIR, 'paper_predictions.csv'), index=False)
print('  Saved → paper_predictions.csv')
print('\n' + '=' * 70)
print('STEP 10  Figures')
print('=' * 70)
fig1, axes = plt.subplots(2, 2, figsize=(13, 11))
fig1.suptitle('MIMIC-III Cardio-Oncology MACE Prediction\n(Paper Methodology: XGBoost + RF 50:50, 4-stat features, 5-fold CV)', fontsize=12, fontweight='bold', y=0.98)
ax = axes[0, 0]
fpr_arr, tpr_arr2, _ = roc_curve(y_all, oof_proba)
ax.plot(fpr_arr, tpr_arr2, color=PAL['mimic'], lw=2, label=f'Ensemble (AUC={oof_auc:.3f}, 95%CI {ci_lo:.3f}–{ci_hi:.3f})')
fpr_a, tpr_a, _ = roc_curve(y_all, oof_age)
ax.plot(fpr_a, tpr_a, color=PAL['age'], lw=1.5, ls='--', label=f'Age-only (AUC={age_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k:', lw=0.8)
if transfer_metrics:
    for (tag, vm), color in zip(transfer_metrics.items(), [PAL['sch'], PAL['whu']]):
        short = tag.split('→')[0]
        ax.plot([], [], lw=1.5, color=color, label=f"{tag} (AUC={vm['AUC']:.3f})")
ax.set_xlabel('1 − Specificity', fontsize=10)
ax.set_ylabel('Sensitivity', fontsize=10)
ax.set_title('A  Discrimination (ROC)', fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
ax.fill_between(fpr_arr, tpr_arr2, alpha=0.06, color=PAL['mimic'])
ax = axes[0, 1]
ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Perfect calibration')
ax.plot(mean_pred, frac_obs, 'o-', color=PAL['mimic'], lw=1.5, ms=5, label=f'Slope={cal_slope:.2f}, Intercept={cal_intercept:.3f}\nE/O={eo_ratio:.2f}, HL χ² p={hl_p:.3f}')
ax.set_xlabel('Mean predicted probability', fontsize=10)
ax.set_ylabel('Observed event fraction', fontsize=10)
ax.set_title('B  Calibration', fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax = axes[1, 0]
ax.plot(thresholds, nb_model, color=PAL['mimic'], lw=2, label='Ensemble model')
ax.plot(thresholds, nb_treat_all, color=PAL['treat_all'], lw=1.5, ls='--', label='Treat all')
ax.axhline(0, color=PAL['treat_none'], lw=1.5, ls=':', label='Treat none')
ax.set_xlabel('Threshold probability', fontsize=10)
ax.set_ylabel('Net benefit', fontsize=10)
ax.set_title('C  Decision Curve Analysis', fontweight='bold')
ax.legend(fontsize=8)
ax.set_xlim([0.0, 0.8])
ax.set_ylim([-0.05, max(nb_treat_all) * 1.1])
ax.axvline(opt_thresh, color=PAL['mimic'], lw=0.8, ls=':', alpha=0.6)
ax.text(opt_thresh + 0.005, max(nb_model) * 0.5, f'Youden\nθ={opt_thresh:.2f}', fontsize=7, color=PAL['mimic'])
ax = axes[1, 1]
colors_km = {'Low': '#2196F3', 'Intermediate': '#FF9800', 'High': '#F44336'}
for g in groups:
    kmf_dict[g].plot_survival_function(ax=ax, color=colors_km[g], lw=2, ci_show=True, ci_alpha=0.1)
ax.set_xlabel('Follow-up (days, within admission)', fontsize=10)
ax.set_ylabel('MACE-free probability', fontsize=10)
ax.set_title('D  Kaplan-Meier by Risk Tertile', fontweight='bold')
ax.set_xlim([0, ax.get_xlim()[1]])
ax.text(0.5, 0.12, f'Log-rank p (Low vs High): {lr_lh.p_value:.3e}', transform=ax.transAxes, fontsize=8, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
fig1.tight_layout(rect=[0, 0, 1, 0.96])
for fmt in ('pdf', 'png'):
    fig1.savefig(os.path.join(FIG_DIR, f'Fig_Paper_Main.{fmt}'), dpi=300, bbox_inches='tight')
plt.close(fig1)
print('  Saved → Fig_Paper_Main.pdf/.png')
top_n = 25
fi_top = fi_df.head(top_n).iloc[::-1]

def feat_color(name):
    if '__mean' in name or '__last' in name or '__max' in name or ('__min' in name):
        lab_name = name.split('__')[0]
        return '#1a6faf'
    return '#e07b39'
fig2, ax2 = plt.subplots(figsize=(9, 8))
colors_fi = [feat_color(f) for f in fi_top['Feature']]
bars = ax2.barh(range(top_n), fi_top['Avg_imp'], color=colors_fi, edgecolor='none')
from matplotlib.patches import Patch
legend_els = [Patch(facecolor='#1a6faf', label='Lab stat (mean/last/max/min)'), Patch(facecolor='#e07b39', label='Demographic / metadata')]
ax2.legend(handles=legend_els, loc='lower right', fontsize=9)
ax2.set_yticks(range(top_n))
ax2.set_yticklabels(fi_top['Feature'], fontsize=8)
ax2.set_xlabel('Average normalised importance\n(XGB gain + RF Gini)', fontsize=10)
ax2.set_title(f'Feature Importance – Top {top_n}\n(Paper method: averaged across 5 CV folds)', fontweight='bold')
ax2.spines[['top', 'right']].set_visible(False)
fig2.tight_layout()
for fmt in ('pdf', 'png'):
    fig2.savefig(os.path.join(FIG_DIR, f'Fig_Paper_Features.{fmt}'), dpi=300, bbox_inches='tight')
plt.close(fig2)
print('  Saved → Fig_Paper_Features.pdf/.png')
fig3, axes3 = plt.subplots(1, 2, figsize=(11, 5))
ax = axes3[0]
nri_components = {'NRI (events)': nri_ev, 'NRI (non-events)': nri_nev, 'Total NRI': nri_total}
xpos = np.arange(3)
cols = ['#1a6faf' if v >= 0 else '#e07b39' for v in nri_components.values()]
bars_n = ax.bar(xpos, list(nri_components.values()), color=cols, width=0.5, edgecolor='white', linewidth=0.5)
ax.axhline(0, color='black', lw=0.8)
for bar, val in zip(bars_n, nri_components.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005 * np.sign(val), f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
ax.errorbar(2, nri_total, yerr=[[nri_total - nri_ci_lo], [nri_ci_hi - nri_total]], fmt='none', color='black', capsize=5, lw=1.5)
ax.set_xticks(xpos)
ax.set_xticklabels(list(nri_components.keys()), fontsize=9)
ax.set_ylabel('NRI value', fontsize=10)
ax.set_title(f'A  NRI vs Age-only\n(Age-only AUC={age_auc:.3f}, Ensemble AUC={oof_auc:.3f})', fontweight='bold', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax = axes3[1]
models_cmp = ['Age-only\nbaseline', 'MIMIC\nwithin-CV']
aucs_cmp = [age_auc, oof_auc]
yerr_lo = [0, oof_auc - ci_lo]
yerr_hi = [0, ci_hi - oof_auc]
colors_cmp = [PAL['age'], PAL['mimic']]
if transfer_metrics:
    for tag, vm in transfer_metrics.items():
        models_cmp.append(tag)
        aucs_cmp.append(vm['AUC'])
        yerr_lo.append(0)
        yerr_hi.append(0)
        colors_cmp.append(PAL['sch'] if 'SCH' in tag else PAL['whu'])
xpos3 = np.arange(len(models_cmp))
bars3 = ax.bar(xpos3, aucs_cmp, color=colors_cmp, width=0.55, edgecolor='white')
ax.errorbar(xpos3, aucs_cmp, yerr=[yerr_lo, yerr_hi], fmt='none', color='black', capsize=4, lw=1.2)
for bar, val in zip(bars3, aucs_cmp):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003, f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(0.5, color='grey', lw=0.8, ls='--', label='No discrimination')
ax.set_xticks(xpos3)
ax.set_xticklabels(models_cmp, fontsize=8)
ax.set_ylabel('AUROC', fontsize=10)
ax.set_ylim([0.45, min(1.0, max(aucs_cmp) + 0.06)])
ax.set_title('B  Model Comparison (AUC)\n(Paper methodology)', fontweight='bold', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
fig3.tight_layout()
for fmt in ('pdf', 'png'):
    fig3.savefig(os.path.join(FIG_DIR, f'Fig_Paper_NRI.{fmt}'), dpi=300, bbox_inches='tight')
plt.close(fig3)
print('  Saved → Fig_Paper_NRI.pdf/.png')
print('\n' + '=' * 70)
print('SUMMARY — Paper Methodology Applied to MIMIC-III')
print('=' * 70)
print(f'  Features   : {len(FEATURE_COLS)}  (23 labs × 4 stats [mean/last/max/min] + 4 demographic)')
print(f'  Admissions : {len(feat_df):,}  |  MACE+: {y_all.sum():,} ({y_all.mean() * 100:.1f}%)')
print()
print(f'  ┌────────────────────────────────────────────────────────────┐')
print(f'  │  Within-MIMIC (5-fold StratKF, XGB+RF 50:50)              │')
print(f'  │    AUC     : {oof_auc:.4f}  (95% CI {ci_lo:.4f}–{ci_hi:.4f})       │')
print(f'  │    Brier   : {oof_brier:.4f}                                       │')
print(f'  │    Cal.slope : {cal_slope:.3f}   E/O: {eo_ratio:.3f}               │')
print(f'  │    C-index   : {c_index:.4f}                                      │')
print(f'  │    Cox HR    : {hr_val:.3f} (95%CI {hr_lo:.3f}–{hr_hi:.3f})  p={cox_p:.2e}│')
print(f'  │    NRI vs age: {nri_total:.4f} (95%CI {nri_ci_lo:.3f}–{nri_ci_hi:.3f})    │')
if transfer_metrics:
    for tag, vm in transfer_metrics.items():
        print(f"  │  {tag}: AUC={vm['AUC']:.4f}  Slope={vm['Cal_slope']:.3f}  E/O={vm['EO_ratio']:.3f}│")
print(f'  └────────────────────────────────────────────────────────────┘')
print()
print(f'  Output files:')
print(f'    {RES_DIR}/paper_metrics_within_mimic.csv')
print(f'    {RES_DIR}/paper_metrics_transfer.csv')
print(f'    {RES_DIR}/paper_feature_importance.csv')
print(f'    {RES_DIR}/paper_predictions.csv')
print(f'    {FIG_DIR}/Fig_Paper_Main.pdf/.png')
print(f'    {FIG_DIR}/Fig_Paper_Features.pdf/.png')
print(f'    {FIG_DIR}/Fig_Paper_NRI.pdf/.png')
print('\nDone.')
