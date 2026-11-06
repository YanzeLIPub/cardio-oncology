"""
compute_v13_fast.py  --  fast, minimal, checkpointed version
Saves intermediate results so each section can be re-run independently.
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
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
ROOT = _release_path('pipeline', '')
MIMIC_FT = _release_path('pipeline', 'MIMIC-3/data/features_model.csv')
MIMIC_LB = _release_path('pipeline', 'MIMIC-3/data/mace_labels.csv')
MIMIC_ADM = _release_path('pipeline', 'MIMIC-3/data/cohort_admissions.csv')
SCH_RAW = _release_path('china', 'SCH_processed_data.csv')
WHU_RAW = _release_path('china', 'files/WHU_Cancer_processed_data.csv')
MACE_FILE = _release_path('pipeline', 'data/real_model_predictions_v2.csv')
OUT_DIR = _release_path('pipeline', 'Manuscript/v13')
CKPT = os.path.join(OUT_DIR, '_ckpt_v13.npz')
FEATS_43_EN = ['Age', 'Apolipoprotein B', 'LDL Cholesterol', 'Total Cholesterol', 'Cholinesterase', 'Albumin', 'Total Bile Acids', 'Prealbumin', 'D-Dimer', 'CYFRA 21-1', 'Cystatin C', 'Triglycerides', 'Adenosine Deaminase', 'Globulin', 'Plateletcrit', 'SCC Antigen', 'Direct Bilirubin', 'No. of Physicians', 'Amylase', 'Hospital Beds', 'Glucose', 'Sex_M', 'HDL Cholesterol', 'Magnesium', 'α-HBDH', 'ALT', 'GDP per Capita', 'No. of Hospitals', 'HBcAb', 'LDH', 'Phosphorus', 'AST', 'Uric Acid', 'Alpha-Fetoprotein', 'Total Protein', 'HbA1c', 'Creatinine', 'Creatine Kinase', 'Avg. Employee Wage', 'Potassium', 'Sodium', 'Apolipoprotein AI', 'Ferritin']
ZH2EN = {'年龄': 'Age', '载脂蛋白B': 'Apolipoprotein B', '低密度脂蛋白胆固醇': 'LDL Cholesterol', '总胆固醇': 'Total Cholesterol', '胆碱酯酶': 'Cholinesterase', '白蛋白': 'Albumin', '总胆汁酸': 'Total Bile Acids', '前白蛋白': 'Prealbumin', 'D-二聚体': 'D-Dimer', '细胞角蛋白19片段': 'CYFRA 21-1', '胱抑素C': 'Cystatin C', '甘油三脂': 'Triglycerides', '腺苷脱氨酶': 'Adenosine Deaminase', '球蛋白': 'Globulin', '血小板压积': 'Plateletcrit', '鳞状细胞癌相关抗原': 'SCC Antigen', '直接胆红素': 'Direct Bilirubin', '医生数(人)': 'No. of Physicians', '淀粉酶': 'Amylase', '医院、卫生院床位数(张)': 'Hospital Beds', '葡萄糖': 'Glucose', '性别': 'Sex_M', '高密度脂蛋白胆固醇': 'HDL Cholesterol', '镁': 'Magnesium', 'α-羟丁酸脱氢酶': 'α-HBDH', '丙氨酸氨基转移酶': 'ALT', '人均地区生产总值(元)': 'GDP per Capita', '医院、卫生院数(个)': 'No. of Hospitals', '乙肝核心抗体': 'HBcAb', '乳酸脱氢酶': 'LDH', '磷': 'Phosphorus', '天门冬氨酸氨基转移酶': 'AST', '尿酸': 'Uric Acid', '甲胎蛋白': 'Alpha-Fetoprotein', '总蛋白': 'Total Protein', '糖化血红蛋白': 'HbA1c', '肌酐': 'Creatinine', '肌酸激酶': 'Creatine Kinase', '职工平均工资(元)': 'Avg. Employee Wage', '钾': 'Potassium', '钠': 'Sodium', '载脂蛋白AⅠ': 'Apolipoprotein AI', '铁蛋白': 'Ferritin'}
FEATS_43_ZH = list(ZH2EN.keys())

def ens_cv(X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    aucs = []
    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        imp = SimpleImputer(strategy='median')
        Xtr = imp.fit_transform(X[tr])
        Xva = imp.transform(X[va])
        ytr = y[tr]
        pw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        fn = [f'f{i}' for i in range(Xtr.shape[1])]
        bst = xgb.train({'objective': 'binary:logistic', 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': pw, 'seed': k, 'nthread': -1, 'verbosity': 0}, xgb.DMatrix(Xtr, label=ytr, feature_names=fn), 300, evals=[(xgb.DMatrix(Xva, label=y[va], feature_names=fn), 'v')], early_stopping_rounds=20, verbose_eval=False)
        xp = bst.predict(xgb.DMatrix(Xva, feature_names=fn))
        rf = RandomForestClassifier(300, max_depth=8, min_samples_leaf=5, class_weight='balanced', random_state=k, n_jobs=-1)
        rf.fit(Xtr, ytr)
        rp = rf.predict_proba(Xva)[:, 1]
        oof[va] = 0.5 * xp + 0.5 * rp
        a = roc_auc_score(y[va], oof[va])
        aucs.append(a)
        print(f'  fold {k}: AUC={a:.4f}', flush=True)
    return (oof, aucs)

def ens_full(X, y):
    imp = SimpleImputer(strategy='median')
    Xf = imp.fit_transform(X)
    pw = (y == 0).sum() / max((y == 1).sum(), 1)
    fn = [f'f{i}' for i in range(Xf.shape[1])]
    bst = xgb.train({'objective': 'binary:logistic', 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': pw, 'seed': 42, 'nthread': -1, 'verbosity': 0}, xgb.DMatrix(Xf, label=y, feature_names=fn), 300, verbose_eval=False)
    rf = RandomForestClassifier(300, max_depth=8, min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(Xf, y)

    def predict(Xnew):
        Xi = imp.transform(Xnew)
        fn2 = [f'f{i}' for i in range(Xi.shape[1])]
        return 0.5 * bst.predict(xgb.DMatrix(Xi, feature_names=fn2)) + 0.5 * rf.predict_proba(Xi)[:, 1]
    return (predict, imp)

def transfer(Xtr, ytr, ftr, Xte, yte, fte, tag):
    common = [f for f in FEATS_43_EN if f in ftr and f in fte]
    print(f'  {tag}: {len(common)} common feats', flush=True)
    itr = [ftr.index(f) for f in common]
    ite = [fte.index(f) for f in common]
    Xt = Xtr[:, itr].astype(float)
    Xe = Xte[:, ite].astype(float)
    imp = SimpleImputer(strategy='median')
    Xt = imp.fit_transform(Xt)
    Xe = imp.transform(Xe)
    pw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    fn = [f'f{i}' for i in range(Xt.shape[1])]
    bst = xgb.train({'objective': 'binary:logistic', 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': pw, 'seed': 42, 'nthread': -1, 'verbosity': 0}, xgb.DMatrix(Xt, label=ytr, feature_names=fn), 300, verbose_eval=False)
    xp = bst.predict(xgb.DMatrix(Xe, feature_names=fn))
    rf = RandomForestClassifier(300, max_depth=8, min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(Xt, ytr)
    rp = rf.predict_proba(Xe)[:, 1]
    p = 0.5 * xp + 0.5 * rp
    a = roc_auc_score(yte.astype(int), p)
    print(f'  {tag}: AUC={a:.4f}', flush=True)
    return a
print('=' * 60)
print('STEP 1  Build n=9,946 cohort')
print('=' * 60)
feat = pd.read_csv(MIMIC_FT)
lbl = pd.read_csv(MIMIC_LB)
adm = pd.read_csv(MIMIC_ADM, parse_dates=['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'DOD'])
if 'HOSPITAL_EXPIRE_FLAG' not in feat.columns:
    lbl2 = pd.read_csv(MIMIC_LB, usecols=['HADM_ID', 'HOSPITAL_EXPIRE_FLAG', 'MACE'])
    df = feat.merge(lbl2, on='HADM_ID', how='inner')
else:
    df = feat.merge(lbl[['HADM_ID', 'MACE']], on='HADM_ID', how='inner')
df9 = df[df['HOSPITAL_EXPIRE_FLAG'] == 0].copy() if 'HOSPITAL_EXPIRE_FLAG' in df.columns else df.copy()
print(f"  n={len(df9):,}, MACE={df9['MACE'].sum()}({df9['MACE'].mean() * 100:.1f}%)")
avf = [f for f in FEATS_43_EN if f in df9.columns]
for c in avf:
    df9[c] = pd.to_numeric(df9[c], errors='coerce')
X9 = df9[avf].values.astype(float)
y9 = df9['MACE'].values.astype(int)
s = adm[['HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DOD', 'EXPIRE_FLAG']].copy()
s['LOS'] = ((s['DISCHTIME'] - s['ADMITTIME']).dt.total_seconds() / 86400).clip(lower=1)
hd = ~s['DOD'].isna()
dd = (s['DOD'] - s['ADMITTIME']).dt.total_seconds() / 86400
s['FU'] = s['LOS'].copy()
s.loc[hd, 'FU'] = np.maximum(s.loc[hd, 'LOS'], dd[hd].clip(lower=1))
sv = df9[['HADM_ID', 'MACE']].merge(s[['HADM_ID', 'LOS', 'FU']], on='HADM_ID', how='left')
sv['T'] = np.where(sv['MACE'] == 1, sv['LOS'], sv['FU'])
sv['T'] = sv['T'].fillna(1).clip(lower=1)
print('=' * 60)
print('STEP 2  5-fold CV')
print('=' * 60)
if os.path.exists(CKPT):
    ck = np.load(CKPT, allow_pickle=True)
    oof9 = ck['oof9']
    aucs9 = list(ck['aucs9'])
    oof_age = ck['oof_age']
    print(f'  Loaded checkpoint. OOF AUC={roc_auc_score(y9, oof9):.4f}')
else:
    oof9, aucs9 = ens_cv(X9, y9)
    Age_col = df9['Age'].values.astype(float).reshape(-1, 1) if 'Age' in df9.columns else X9[:, avf.index('Age')].reshape(-1, 1)
    oof_age = np.zeros(len(y9))
    skf2 = StratifiedKFold(5, shuffle=True, random_state=42)
    for tr, va in skf2.split(X9, y9):
        ia = SimpleImputer(strategy='median')
        at = ia.fit_transform(Age_col[tr])
        av = ia.transform(Age_col[va])
        mu, sg = (at.mean(), at.std() + 1e-09)
        lr = LogisticRegression(max_iter=1000)
        lr.fit((at - mu) / sg, y9[tr])
        oof_age[va] = lr.predict_proba((av - mu) / sg)[:, 1]
    np.savez(CKPT, oof9=oof9, aucs9=np.array(aucs9), oof_age=oof_age)
    print('  Checkpoint saved.')
sv['score'] = oof9
auc = roc_auc_score(y9, oof9)
br = brier_score_loss(y9, oof9)
rng = np.random.default_rng(42)
boot = [roc_auc_score(y9[i], oof9[i]) for i in (rng.choice(len(y9), len(y9), replace=True) for _ in range(2000))]
cilo, cihi = np.percentile(boot, [2.5, 97.5])
eo = oof9.mean() / y9.mean()
ci = concordance_index(sv['T'].values, -sv['score'].values, sv['MACE'].values)
print(f'\n  AUC={auc:.4f} (CI {cilo:.4f}-{cihi:.4f})')
print(f'  mean fold AUC={np.mean(aucs9):.4f} SD={np.std(aucs9):.4f}')
print(f'  Brier={br:.4f}  E/O={eo:.4f}  C-index={ci:.4f}')
for t in [7, 14, 30, 60, 90]:
    mc = (sv['MACE'] == 1) & (sv['T'] <= t)
    ct = sv['T'] >= t
    mask = mc | ct
    if mask.sum() < 20 or mc[mask].sum() == 0 or mc[mask].sum() == mask.sum():
        print(f'  td-AUC@{t:3d}d: N/A')
        continue
    print(f"  td-AUC@{t:3d}d: {roc_auc_score(mc[mask].astype(int).values, sv.loc[mask, 'score'].values):.4f}")
evt_times = np.sort(sv.loc[sv['MACE'] == 1, 'T'].values)
if len(evt_times) > 200:
    idx2 = np.round(np.linspace(0, len(evt_times) - 1, 200)).astype(int)
    evt_times = evt_times[idx2]
T_arr = sv['T'].values
y_arr = sv['MACE'].values
p_arr = sv['score'].values
bs_vals = [((T_arr <= t) & (y_arr == 1)).astype(float) * (1 - p_arr) ** 2 + (T_arr > t).astype(float) * p_arr ** 2 for t in evt_times]
bs_vals = [x.mean() for x in bs_vals]
trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
ibs_v = trapz(bs_vals, evt_times) / (evt_times[-1] - evt_times[0]) if len(evt_times) > 1 else np.nan
print(f'  IBS={ibs_v:.4f}')
age_auc = roc_auc_score(y9, oof_age)
dauc = auc - age_auc

def nri(y, pn, pb):
    ev = y == 1
    ne = y == 0
    return np.mean(pn[ev] > pb[ev]) - np.mean(pn[ev] < pb[ev]) + np.mean(pn[ne] < pb[ne]) - np.mean(pn[ne] > pb[ne])
nt = nri(y9, oof9, oof_age)
bnri = [nri(y9[(i := rng.integers(0, len(y9), len(y9)))], oof9[i], oof_age[i]) for _ in range(1000)]
nc, nh = np.percentile(bnri, [2.5, 97.5])
print(f'  dAUC=+{dauc:.4f}  NRI={nt:.4f} (CI {nc:.4f}-{nh:.4f})')
terq = np.percentile(oof9, [33.3, 66.7])
sv['grp'] = pd.cut(oof9, bins=np.unique([oof9.min() - 0.001, terq[0], terq[1], oof9.max() + 0.001]), labels=['Low', 'Medium', 'High'])
sv['grp'] = sv['grp'].astype(str)
lr = logrank_test(sv.loc[sv['grp'] == 'Low', 'T'], sv.loc[sv['grp'] == 'High', 'T'], event_observed_A=sv.loc[sv['grp'] == 'Low', 'MACE'], event_observed_B=sv.loc[sv['grp'] == 'High', 'MACE'])
print(f'  log-rank p (Low vs High)={lr.p_value:.4e}')
print('=' * 60)
print('STEP 3  Transfer analysis (9946)')
print('=' * 60)

def load_cohort(raw_path, center, is_sch=False):
    lbs = pd.read_csv(MACE_FILE, usecols=['病案号', 'Center', 'MACE'])
    lbs = lbs[lbs['Center'] == center][['病案号', 'MACE']].drop_duplicates('病案号')
    av = pd.read_csv(raw_path, nrows=0).columns.tolist()
    uc = ['病案号'] + [c for c in FEATS_43_ZH if c in av]
    if is_sch:
        raw = pd.concat([c for c in pd.read_csv(raw_path, usecols=uc, chunksize=120000, low_memory=False)])
    else:
        raw = pd.read_csv(raw_path, usecols=uc, low_memory=False)
    for c in FEATS_43_ZH:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors='coerce')
    raw = raw.merge(lbs, on='病案号', how='inner').rename(columns=ZH2EN)
    fc = [c for c in raw.columns if c not in ['病案号', 'MACE']]
    agg = raw.groupby('病案号')[fc].median(numeric_only=True).reset_index()
    yd = raw.groupby('病案号', as_index=False)['MACE'].max()
    data = agg.merge(yd, on='病案号', how='inner')
    feats = [f for f in FEATS_43_EN if f in data.columns]
    X = data[feats].values.astype(float)
    y = data['MACE'].values.astype(int)
    print(f'  {center}: {len(data):,} pts, MACE {y.sum()} ({y.mean() * 100:.1f}%), feats={len(feats)}')
    return (X, y, feats)
X_sch, y_sch, f_sch = load_cohort(SCH_RAW, 'SCH', True)
X_whu, y_whu, f_whu = load_cohort(WHU_RAW, 'WHU', False)
f9 = [f for f in FEATS_43_EN if f in df9.columns]
X9_43 = df9[f9].values.astype(float)
a_s2m = transfer(X_sch, y_sch, f_sch, X9_43, y9, f9, 'SCH→MIMIC')
a_w2m = transfer(X_whu, y_whu, f_whu, X9_43, y9, f9, 'WHU→MIMIC')
a_m2s = transfer(X9_43, y9, f9, X_sch, y_sch, f_sch, 'MIMIC→SCH')
a_m2w = transfer(X9_43, y9, f9, X_whu, y_whu, f_whu, 'MIMIC→WHU')
print('=' * 60)
print('STEP 4  KM figure')
print('=' * 60)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.facecolor': 'white', 'figure.facecolor': 'white'})
fig, ax = plt.subplots(figsize=(85 / 25.4, 80 / 25.4))
for g, co in [('Low', '#2166AC'), ('Medium', '#F4A582'), ('High', '#D6604D')]:
    m = sv['grp'] == g
    KaplanMeierFitter().fit(sv.loc[m, 'T'], event_observed=sv.loc[m, 'MACE'], label=f'{g} (n={m.sum():,})').plot_survival_function(ax=ax, color=co, lw=1.5, ci_show=True, ci_alpha=0.12)
ax.set_xlabel('Follow-up (days)', fontsize=7)
ax.set_ylabel('MACE-free probability', fontsize=7)
ax.set_title(f'C) MIMIC-III cohort\n(n=9,946; log-rank p<0·0001)', fontsize=7, fontweight='bold')
ax.legend(fontsize=6, loc='lower left')
ax.tick_params(labelsize=6)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT_DIR, f'Figure1C_KM_MIMIC_9946.{ext}'), dpi=600, bbox_inches='tight')
plt.close(fig)
print(f'  Saved KM figure')
t90 = 90
mc90 = (sv['MACE'] == 1) & (sv['T'] <= t90)
ct90 = sv['T'] >= t90
m90 = mc90 | ct90
tdauc90 = roc_auc_score(mc90[m90].astype(int).values, sv.loc[m90, 'score'].values) if mc90[m90].sum() > 0 else np.nan
summary = {'n': len(y9), 'MACE_n': int(y9.sum()), 'MACE_pct': round(y9.mean() * 100, 2), 'AUC': round(auc, 4), 'AUC_CI_lo': round(cilo, 4), 'AUC_CI_hi': round(cihi, 4), 'C_index': round(ci, 4), 'TimedepAUC_90d': round(tdauc90, 4) if not np.isnan(tdauc90) else None, 'IBS': round(ibs_v, 4) if not np.isnan(ibs_v) else None, 'EO_ratio_raw': round(eo, 4), 'Logrank_Low_vs_High': lr.p_value, 'Delta_AUC': round(dauc, 4), 'NRI': round(nt, 4), 'NRI_CI_lo': round(nc, 4), 'NRI_CI_hi': round(nh, 4), 'Transfer_SCH2MIMIC': round(a_s2m, 4), 'Transfer_WHU2MIMIC': round(a_w2m, 4), 'Transfer_MIMIC2SCH': round(a_m2s, 4), 'Transfer_MIMIC2WHU': round(a_m2w, 4)}
pd.DataFrame([summary]).to_csv(os.path.join(OUT_DIR, 'summary_v13_metrics.csv'), index=False)
pd.DataFrame([{'Configuration': k, 'AUC_9946': v} for k, v in [('SCH→MIMIC', a_s2m), ('WHU→MIMIC', a_w2m), ('MIMIC→SCH', a_m2s), ('MIMIC→WHU', a_m2w)]]).to_csv(os.path.join(OUT_DIR, 'transfer_metrics_9946.csv'), index=False)
print('\n' + '=' * 60)
print('SUMMARY  (n=9,946, patient-level MIMIC)')
print('=' * 60)
for k, v in summary.items():
    print(f'  {k:<28s}: {v}')
print('\nDone.')
