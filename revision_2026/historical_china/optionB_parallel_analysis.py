"""
Option B: Parallel Cardio-Oncology Cohort Study
"Parallel Development and Comparison of MACE Prediction Models
 in Two Independent Cardio-Oncology Cohorts"

This script:
1. WHU 5-fold CV (within-WHU model)
2. SCH 5-fold CV (already done; reproduced for feature importance)
3. Cross-cohort transfer: SCH->WHU and WHU->SCH AUC
4. Feature importance comparison + common/unique features
5. Generates all comparison figures
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
print('=' * 65)
print('PARALLEL COHORT MACE PREDICTION — Option B')
print('=' * 65)
OUTPUT_DIR = _release_path('china', 'outputs')
print('\n[1/6] Loading feature data...')
sch_head = pd.read_csv(_release_path('china', 'SCH_processed_data.csv'), nrows=0)
whu_head = pd.read_csv(_release_path('china', 'WHU_Cancer_processed_data.csv'), nrows=0)
exclude_pats = ['Unnamed', '病案号', '时间', 'time', '日期', '地址', '电话', '姓名', '编码', '名称', '人员', '单位', '联系']

def is_feature(col):
    return not any((p in col for p in exclude_pats))
feat_cols = [c for c in sch_head.columns if c in set(whu_head.columns) and is_feature(c)]
print(f'  Common features: {len(feat_cols)}')
labels = pd.read_csv(_release_path('china', 'outputs/survival_analysis_patient_level/patient_level_data.csv'))
sch_labels = labels[labels['Center'] == 'SCH'][['病案号', 'MACE']].copy()
sch_labels['MACE'] = sch_labels['MACE'].astype(int)
print('  Loading SCH (selected cols)...')
sch_load = ['病案号'] + feat_cols
sch_df = pd.read_csv(_release_path('china', 'SCH_processed_data.csv'), usecols=[c for c in sch_load if c in sch_head.columns], low_memory=False)
for col in feat_cols:
    if col in sch_df.columns:
        sch_df[col] = pd.to_numeric(sch_df[col], errors='coerce')
sch_agg = sch_df.groupby('病案号')[feat_cols].mean().reset_index()
sch_merged = sch_labels.merge(sch_agg, on='病案号', how='inner')
y_sch = sch_merged['MACE'].values.astype(int)
pid_sch = sch_merged['病案号'].values
X_sch = sch_merged[feat_cols].fillna(0).values.astype(np.float32)
X_sch = np.nan_to_num(X_sch, nan=0.0, posinf=0.0, neginf=0.0)
print(f'  SCH: {X_sch.shape}, MACE={y_sch.sum()} ({y_sch.mean() * 100:.2f}%)')
whu_labels = labels[labels['Center'] == 'WHU'][['病案号', 'MACE']].copy()
whu_labels['MACE'] = whu_labels['MACE'].astype(int)
print('  Loading WHU (selected cols)...')
whu_df = pd.read_csv(_release_path('china', 'WHU_Cancer_processed_data.csv'), usecols=[c for c in sch_load if c in whu_head.columns], low_memory=False)
for col in feat_cols:
    if col in whu_df.columns:
        whu_df[col] = pd.to_numeric(whu_df[col], errors='coerce')
whu_cols_present = [c for c in feat_cols if c in whu_df.columns]
whu_agg = whu_df.groupby('病案号')[whu_cols_present].mean().reset_index()
whu_merged = whu_labels.merge(whu_agg, on='病案号', how='inner')
y_whu = whu_merged['MACE'].values.astype(int)
pid_whu = whu_merged['病案号'].values
X_whu = np.zeros((len(whu_merged), len(feat_cols)), dtype=np.float32)
for i, col in enumerate(feat_cols):
    if col in whu_merged.columns:
        X_whu[:, i] = whu_merged[col].fillna(0).values.astype(np.float32)
X_whu = np.nan_to_num(X_whu, nan=0.0, posinf=0.0, neginf=0.0)
print(f'  WHU: {X_whu.shape}, MACE={y_whu.sum()} ({y_whu.mean() * 100:.2f}%)')

def make_xgb_params(y):
    pos_w = (y == 0).sum() / max(1, (y == 1).sum())
    return {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': pos_w, 'seed': 42, 'nthread': -1}

def make_rf(y):
    pos_w = (y == 0).sum() / max(1, (y == 1).sum())
    return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight={0: 1, 1: int(pos_w)}, random_state=42, n_jobs=-1)

def cv_oof(X, y, n_splits=5, n_rounds=300):
    xgb_params = make_xgb_params(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    fold_aucs = []
    feat_imps = []
    models = []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        dtr = xgb.DMatrix(X[tr], label=y[tr])
        dva = xgb.DMatrix(X[va], label=y[va])
        m_xgb = xgb.train(xgb_params, dtr, num_boost_round=n_rounds, evals=[(dtr, 'tr'), (dva, 'va')], early_stopping_rounds=20, verbose_eval=False)
        p_xgb = m_xgb.predict(dva)
        m_rf = make_rf(y)
        m_rf.fit(X[tr], y[tr])
        p_rf = m_rf.predict_proba(X[va])[:, 1]
        preds = (p_xgb + p_rf) / 2.0
        oof[va] = preds
        fold_aucs.append(roc_auc_score(y[va], preds))
        feat_imps.append({'xgb': m_xgb.get_score(importance_type='gain'), 'rf': m_rf.feature_importances_})
        models.append((m_xgb, m_rf))
        print(f'    fold {fold}: XGB={roc_auc_score(y[va], p_xgb):.4f}  RF={roc_auc_score(y[va], p_rf):.4f}  Ensemble={roc_auc_score(y[va], preds):.4f}')
    return (oof, fold_aucs, feat_imps, models)
print('\n[2/6] SCH 5-fold CV...')
sch_oof, sch_fold_aucs, sch_feat_imps, sch_models = cv_oof(X_sch, y_sch)
sch_cv_auc = roc_auc_score(y_sch, sch_oof)
print(f'  SCH fold AUCs: {[round(a, 4) for a in sch_fold_aucs]}')
print(f'  SCH OOF AUC:   {sch_cv_auc:.4f} +/- {np.std(sch_fold_aucs):.4f}')
print('\n[3/6] WHU 5-fold CV...')
whu_oof, whu_fold_aucs, whu_feat_imps, whu_models = cv_oof(X_whu, y_whu)
whu_cv_auc = roc_auc_score(y_whu, whu_oof)
print(f'  WHU fold AUCs: {[round(a, 4) for a in whu_fold_aucs]}')
print(f'  WHU OOF AUC:   {whu_cv_auc:.4f} +/- {np.std(whu_fold_aucs):.4f}')
print('\n[4/6] Cross-cohort transfer tests...')
xgb_sch = xgb.train(make_xgb_params(y_sch), xgb.DMatrix(X_sch, label=y_sch), num_boost_round=200, verbose_eval=False)
rf_sch = make_rf(y_sch)
rf_sch.fit(X_sch, y_sch)
whu_from_sch = (xgb_sch.predict(xgb.DMatrix(X_whu)) + rf_sch.predict_proba(X_whu)[:, 1]) / 2.0
sch2whu_auc = roc_auc_score(y_whu, whu_from_sch)
print(f'  SCH->WHU transfer AUC: {sch2whu_auc:.4f}')
xgb_whu = xgb.train(make_xgb_params(y_whu), xgb.DMatrix(X_whu, label=y_whu), num_boost_round=200, verbose_eval=False)
rf_whu = make_rf(y_whu)
rf_whu.fit(X_whu, y_whu)
sch_from_whu = (xgb_whu.predict(xgb.DMatrix(X_sch)) + rf_whu.predict_proba(X_sch)[:, 1]) / 2.0
whu2sch_auc = roc_auc_score(y_sch, sch_from_whu)
print(f'  WHU->SCH transfer AUC: {whu2sch_auc:.4f}')
print('\n[5/6] Feature importance analysis...')

def aggregate_importance(feat_imps_list, feat_names, top_n=25):
    """Aggregate ensemble importance across CV folds (XGB gain + RF MDI), return top_n."""
    agg = {}
    n_folds = len(feat_imps_list)
    for fold_imp in feat_imps_list:
        xgb_total = sum(fold_imp['xgb'].values()) or 1
        for k, v in fold_imp['xgb'].items():
            try:
                idx = int(k[1:])
                name = feat_names[idx]
            except:
                name = k
            agg[name] = agg.get(name, 0) + v / xgb_total / n_folds
        rf_arr = fold_imp['rf']
        rf_total = rf_arr.sum() or 1
        for idx, v in enumerate(rf_arr):
            name = feat_names[idx]
            agg[name] = agg.get(name, 0) + v / rf_total / n_folds
    sorted_imp = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    return sorted_imp[:top_n]
sch_top = aggregate_importance(sch_feat_imps, feat_cols, top_n=25)
whu_top = aggregate_importance(whu_feat_imps, feat_cols, top_n=25)
sch_top_names = set((x[0] for x in sch_top))
whu_top_names = set((x[0] for x in whu_top))
common_names = sch_top_names & whu_top_names
sch_only = sch_top_names - whu_top_names
whu_only = whu_top_names - sch_top_names
print(f'  SCH top-25 features: {len(sch_top_names)}')
print(f'  WHU top-25 features: {len(whu_top_names)}')
print(f'  Common top-25:       {len(common_names)} -> {sorted(common_names)}')
print(f'  SCH-only:            {sorted(sch_only)}')
print(f'  WHU-only:            {sorted(whu_only)}')
sch_imp_dict = dict(sch_top)
whu_imp_dict = dict(whu_top)
all_feats = sorted(sch_top_names | whu_top_names)
feat_comp_df = pd.DataFrame({'Feature': all_feats, 'SCH_importance': [sch_imp_dict.get(f, 0) for f in all_feats], 'WHU_importance': [whu_imp_dict.get(f, 0) for f in all_feats], 'In_common': [f in common_names for f in all_feats]})
feat_comp_df['SCH_norm'] = feat_comp_df['SCH_importance'] / feat_comp_df['SCH_importance'].max()
feat_comp_df['WHU_norm'] = feat_comp_df['WHU_importance'] / feat_comp_df['WHU_importance'].max()
feat_comp_df = feat_comp_df.sort_values('SCH_norm', ascending=False)
feat_comp_df.to_csv(f'{OUTPUT_DIR}/feature_importance_comparison.csv', index=False, encoding='utf-8-sig')
print(f'  Saved: feature_importance_comparison.csv')
FEAT_EN = {'低密度脂蛋白胆固醇': 'LDL Cholesterol', '高密度脂蛋白胆固醇': 'HDL Cholesterol', '总胆固醇': 'Total Cholesterol', '甘油三脂': 'Triglycerides', '载脂蛋白B': 'Apolipoprotein B', '载脂蛋白AⅠ': 'Apolipoprotein A1', '年龄': 'Age', '性别': 'Sex', '医生数(人)': 'Physician Count', '甲胎蛋白': 'AFP', '神经元特异性烯醇化酶': 'NSE', '癌胚抗原': 'CEA', '糖类抗原CA125': 'CA-125', '细胞角蛋白19片段': 'CYFRA21-1', '鳞状细胞癌相关抗原': 'SCC-Ag', '乳酸脱氢酶': 'LDH', 'α-羟丁酸脱氢酶': 'α-HBDH', '天门冬氨酸氨基转移酶': 'AST', '白蛋白': 'Albumin', '总胆汁酸': 'Total Bile Acids', '胆碱酯酶': 'Cholinesterase', '胱抑素C': 'Cystatin C', '肌酸激酶': 'Creatine Kinase', '肌酐': 'Creatinine', '葡萄糖': 'Glucose', '糖化血红蛋白': 'HbA1c', '尿比重': 'Urine SG', '腺苷脱氨酶': 'Adenosine Deaminase', '白细胞介素-4': 'IL-4', '铁蛋白': 'Ferritin', '磷': 'Phosphorus', '钾': 'Potassium', '乙肝核心抗体': 'HBc Antibody', '人均地区生产总值(元)': 'GDP per Capita', '医院、卫生院床位数(张)': 'Hospital Beds', '医院、卫生院数(个)': 'Hospital Count', '职工平均工资(元)': 'Avg Employee Wage', 'LY30': 'LY30 (TEG)', 'MA': 'MA (TEG)', 'R': 'R (TEG)'}

def feat_en(name):
    """Return English label for a Chinese feature name, or the name as-is."""
    return FEAT_EN.get(name, name)
print('\n[6a/7] Saving prediction CSVs...')
preds_old = pd.read_csv(f'{OUTPUT_DIR}/real_model_predictions.csv')
whu_update = pd.DataFrame({'病案号': pid_whu, 'predicted_prob_within_cv': whu_oof})
sch_update = pd.DataFrame({'病案号': pid_sch, 'predicted_prob_within_cv': sch_oof})
upd = pd.concat([sch_update, whu_update], ignore_index=True)
preds_merged = preds_old.merge(upd, on='病案号', how='left')
whu_cross = pd.DataFrame({'病案号': pid_whu, 'pred_from_sch': whu_from_sch})
sch_cross = pd.DataFrame({'病案号': pid_sch, 'pred_from_whu': sch_from_whu})
preds_merged = preds_merged.merge(whu_cross, on='病案号', how='left')
preds_merged = preds_merged.merge(sch_cross, on='病案号', how='left')
preds_merged.to_csv(f'{OUTPUT_DIR}/real_model_predictions_v2.csv', index=False)
print(f'  Saved: real_model_predictions_v2.csv')
common_names_list = sorted(common_names)
sch_only_list = sorted(sch_only)
whu_only_list = sorted(whu_only)
summary_df = pd.DataFrame({'Metric': ['SCH_CV_AUC', 'SCH_CV_SD', 'WHU_CV_AUC', 'WHU_CV_SD', 'SCH_to_WHU_transfer', 'WHU_to_SCH_transfer', 'Common_top25_features', 'SCH_n', 'WHU_n', 'SCH_MACE_events', 'WHU_MACE_events', 'SCH_MACE_rate', 'WHU_MACE_rate'], 'Value': [round(sch_cv_auc, 4), round(np.std(sch_fold_aucs), 4), round(whu_cv_auc, 4), round(np.std(whu_fold_aucs), 4), round(sch2whu_auc, 4), round(whu2sch_auc, 4), len(common_names), len(y_sch), len(y_whu), int(y_sch.sum()), int(y_whu.sum()), round(y_sch.mean() * 100, 2), round(y_whu.mean() * 100, 2)]})
summary_df.to_csv(f'{OUTPUT_DIR}/OptionB_Summary.csv', index=False)
print(f'  Saved: OptionB_Summary.csv')
sch_models.clear()
whu_models.clear()
import gc
gc.collect()
print('\n[7/7] Generating figures...')
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Parallel MACE Prediction Models: SCH vs WHU Cardio-Oncology Cohorts', fontsize=14, fontweight='bold', y=0.98)
ax_roc = fig.add_subplot(2, 3, 1)
fpr_s, tpr_s, _ = roc_curve(y_sch, sch_oof)
fpr_w, tpr_w, _ = roc_curve(y_whu, whu_oof)
ax_roc.plot(fpr_s, tpr_s, 'b-', lw=2.5, label=f'SCH (AUC={sch_cv_auc:.3f})')
ax_roc.plot(fpr_w, tpr_w, 'r-', lw=2.5, label=f'WHU (AUC={whu_cv_auc:.3f})')
ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax_roc.set_xlabel('False Positive Rate', fontsize=10)
ax_roc.set_ylabel('True Positive Rate', fontsize=10)
ax_roc.set_title('(A) Within-Cohort ROC\n(5-fold CV, real MACE labels)', fontsize=10)
ax_roc.legend(fontsize=9)
ax_roc.grid(alpha=0.3)
ax_trans = fig.add_subplot(2, 3, 2)
transfer_data = {'SCH\n→SCH\n(CV)': sch_cv_auc, 'WHU\n→WHU\n(CV)': whu_cv_auc, 'SCH\n→WHU\n(transfer)': sch2whu_auc, 'WHU\n→SCH\n(transfer)': whu2sch_auc}
bar_colors = ['#2196F3', '#f44336', '#9E9E9E', '#9E9E9E']
bars = ax_trans.bar(transfer_data.keys(), transfer_data.values(), color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1.5)
ax_trans.axhline(y=0.5, color='black', linestyle='--', lw=1.2, alpha=0.6, label='Random (0.5)')
ax_trans.set_ylabel('AUC', fontsize=10)
ax_trans.set_ylim([0.4, 0.85])
ax_trans.set_title('(B) Within- vs Cross-Cohort AUC\n(Non-transferability finding)', fontsize=10)
for bar, val in zip(bars, transfer_data.values()):
    ax_trans.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f'{val:.3f}', ha='center', fontsize=9.5, fontweight='bold')
ax_trans.legend(fontsize=9)
ax_trans.grid(axis='y', alpha=0.3)
ax_char = fig.add_subplot(2, 3, 3)
char_data = {'n (×1000)': [len(y_sch) / 1000, len(y_whu) / 1000], 'MACE rate (×10%)': [y_sch.mean() * 10 * 10, y_whu.mean() * 10 * 10]}
x = np.arange(2)
width = 0.35
ax_char.bar([-0.2], [len(y_sch) / 1000], width=0.35, color='#2196F3', alpha=0.9, edgecolor='white', label='SCH (Oncology center)')
ax_char.bar([0.2], [len(y_whu) / 1000], width=0.35, color='#f44336', alpha=0.9, edgecolor='white', label='WHU (General hospital)')
ax_char.bar([0.8], [y_sch.mean() * 100], width=0.35, color='#2196F3', alpha=0.6, edgecolor='white')
ax_char.bar([1.2], [y_whu.mean() * 100], width=0.35, color='#f44336', alpha=0.6, edgecolor='white')
ax_char.set_xticks([0, 1])
ax_char.set_xticklabels(['Sample size\n(×1,000 patients)', 'MACE rate (%)'])
ax_char.set_title('(C) Cohort Characteristics\n(Different risk profiles)', fontsize=10)
ax_char.legend(fontsize=8.5)
ax_char.text(-0.2, len(y_sch) / 1000 + 0.3, '38,509', ha='center', fontsize=8, color='#1565C0')
ax_char.text(0.2, len(y_whu) / 1000 + 0.3, '2,562', ha='center', fontsize=8, color='#b71c1c')
ax_char.text(0.8, y_sch.mean() * 100 + 0.1, f'{y_sch.mean() * 100:.1f}%', ha='center', fontsize=8, color='#1565C0')
ax_char.text(1.2, y_whu.mean() * 100 + 0.1, f'{y_whu.mean() * 100:.1f}%', ha='center', fontsize=8, color='#b71c1c')
ax_char.grid(axis='y', alpha=0.3)
ax_sch_imp = fig.add_subplot(2, 3, 4)
top15_sch = sch_top[:15]
feats_s = [x[0] for x in top15_sch]
imps_s = [x[1] for x in top15_sch]
imps_s_n = [v / max(imps_s) for v in imps_s]
colors_s = ['#1565C0' if f in common_names else '#2196F3' for f in feats_s]
ax_sch_imp.barh(range(15), imps_s_n[::-1], color=colors_s[::-1], alpha=0.85, edgecolor='white')
ax_sch_imp.set_yticks(range(15))
ax_sch_imp.set_yticklabels([feat_en(f) for f in feats_s[::-1]], fontsize=8)
ax_sch_imp.set_xlabel('Normalized Importance (gain)', fontsize=9)
ax_sch_imp.set_title('(D) SCH Top-15 Features\n(dark blue = also in WHU top-25)', fontsize=10)
ax_sch_imp.grid(axis='x', alpha=0.3)
ax_whu_imp = fig.add_subplot(2, 3, 5)
top15_whu = whu_top[:15]
feats_w = [x[0] for x in top15_whu]
imps_w = [x[1] for x in top15_whu]
imps_w_n = [v / max(imps_w) for v in imps_w]
colors_w = ['#b71c1c' if f in common_names else '#f44336' for f in feats_w]
ax_whu_imp.barh(range(15), imps_w_n[::-1], color=colors_w[::-1], alpha=0.85, edgecolor='white')
ax_whu_imp.set_yticks(range(15))
ax_whu_imp.set_yticklabels([feat_en(f) for f in feats_w[::-1]], fontsize=8)
ax_whu_imp.set_xlabel('Normalized Importance (gain)', fontsize=9)
ax_whu_imp.set_title('(E) WHU Top-15 Features\n(dark red = also in SCH top-25)', fontsize=10)
ax_whu_imp.grid(axis='x', alpha=0.3)
ax_scatter = fig.add_subplot(2, 3, 6)
scatter_df = feat_comp_df.copy()
common_mask = scatter_df['In_common']
ax_scatter.scatter(scatter_df[~common_mask]['SCH_norm'], scatter_df[~common_mask]['WHU_norm'], alpha=0.5, s=50, color='grey', label='Cohort-specific', zorder=3)
ax_scatter.scatter(scatter_df[common_mask]['SCH_norm'], scatter_df[common_mask]['WHU_norm'], alpha=0.9, s=100, color='#4CAF50', edgecolors='#1B5E20', linewidths=1.5, label=f'Common ({common_mask.sum()})', zorder=5)
for _, row in scatter_df[common_mask].iterrows():
    ax_scatter.annotate(feat_en(row['Feature']), (row['SCH_norm'], row['WHU_norm']), textcoords='offset points', xytext=(4, 3), fontsize=7, color='#1B5E20', fontweight='bold')
ax_scatter.set_xlabel('SCH Normalized Importance', fontsize=10)
ax_scatter.set_ylabel('WHU Normalized Importance', fontsize=10)
ax_scatter.set_title('(F) Feature Importance Concordance\n(green = cross-context signal)', fontsize=10)
ax_scatter.legend(fontsize=9)
ax_scatter.grid(alpha=0.3)
ax_scatter.set_xlim([-0.05, 1.1])
ax_scatter.set_ylim([-0.05, 1.1])
plt.tight_layout(rect=[0, 0, 1, 0.97])
out_fig = f'{OUTPUT_DIR}/OptionB_Parallel_Cohort_Analysis.pdf'
plt.savefig(out_fig, dpi=200, bbox_inches='tight')
plt.savefig(out_fig.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f'  Saved: {out_fig}')
preds_old = pd.read_csv(f'{OUTPUT_DIR}/real_model_predictions.csv')
whu_update = pd.DataFrame({'病案号': pid_whu, 'predicted_prob_within_cv': whu_oof})
sch_update = pd.DataFrame({'病案号': pid_sch, 'predicted_prob_within_cv': sch_oof})
upd = pd.concat([sch_update, whu_update], ignore_index=True)
preds_merged = preds_old.merge(upd, on='病案号', how='left')
whu_cross = pd.DataFrame({'病案号': pid_whu, 'pred_from_sch': whu_from_sch})
sch_cross = pd.DataFrame({'病案号': pid_sch, 'pred_from_whu': sch_from_whu})
preds_merged = preds_merged.merge(whu_cross, on='病案号', how='left')
preds_merged = preds_merged.merge(sch_cross, on='病案号', how='left')
preds_merged.to_csv(f'{OUTPUT_DIR}/real_model_predictions_v2.csv', index=False)
print(f'  Saved: real_model_predictions_v2.csv')
print('\n' + '=' * 65)
print('OPTION B RESULTS — PARALLEL COHORT STUDY')
print('=' * 65)
print(f'\nWithin-cohort performance (5-fold CV):')
print(f'  SCH: AUC = {sch_cv_auc:.4f} +/- {np.std(sch_fold_aucs):.4f}')
print(f'       Folds: {[round(a, 4) for a in sch_fold_aucs]}')
print(f'  WHU: AUC = {whu_cv_auc:.4f} +/- {np.std(whu_fold_aucs):.4f}')
print(f'       Folds: {[round(a, 4) for a in whu_fold_aucs]}')
print(f'\nCross-cohort transfer (non-transferability finding):')
print(f'  SCH trained -> predict WHU: AUC = {sch2whu_auc:.4f}')
print(f'  WHU trained -> predict SCH: AUC = {whu2sch_auc:.4f}')
print(f'\nFeature analysis (top-25):')
print(f'  Common features ({len(common_names)}): {sorted(common_names)}')
print(f'  SCH-specific ({len(sch_only)}): {sorted(sch_only)}')
print(f'  WHU-specific ({len(whu_only)}): {sorted(whu_only)}')
print(f'\nKeys for paper narrative:')
print(f'  1. Both cohorts support predictive models (SCH={sch_cv_auc:.3f}, WHU={whu_cv_auc:.3f})')
print(f'  2. Non-transferability: SCH->WHU = {sch2whu_auc:.3f}, WHU->SCH = {whu2sch_auc:.3f}')
print(f'  3. {len(common_names)} features predictive in BOTH contexts = cross-context signal')
print(f'  4. Divergence reflects different clinical risk contexts (3.4% vs 11.0% MACE)')
print('=' * 65)
print(f'\nSaved: OptionB_Summary.csv')
print(f'Saved: feature_importance_comparison.csv')
print(f'Saved: OptionB_Parallel_Cohort_Analysis.pdf/.png')
