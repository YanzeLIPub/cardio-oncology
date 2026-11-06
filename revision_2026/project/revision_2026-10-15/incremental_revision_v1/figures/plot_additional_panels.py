"""Plot frozen original ROC probabilities and audited aggregate-only additions."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent.parent / 'original_v5_independent_validation_v1'
LAND = ROOT.parent.parent / 'landmark_completion_v1'
PRED = Path(_release_path('pipeline', ''))
plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})

def identity(p):
    p = Path(p)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def save(fig, stem):
    fig.savefig(ROOT / (stem + '.pdf'), facecolor='white')
    plt.close(fig)

def figure4():
    cnpath = PRED / 'data/real_model_predictions_v2.csv'
    mpath = PRED / 'Manuscript/v13/paper_predictions_9946.csv'
    cpath = LAND / 'calibration/calibration_curve_intervals.csv'
    cn = pd.read_csv(cnpath, usecols=['Center', 'MACE', 'predicted_prob_within_cv'])
    mm = pd.read_csv(mpath, usecols=['MACE', 'oof_prob_paper'])
    cal = pd.read_csv(cpath)
    fig, axes = plt.subplots(2, 3, figsize=(10.3, 7.4))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.15, top=0.94, wspace=0.3, hspace=0.48)
    aucs = {}
    for j, (cohort, color) in enumerate(zip(['SCH', 'WHU', 'MIMIC'], ['#2166AC', '#EF8A62', '#159447'])):
        data = cn[cn.Center == cohort] if cohort != 'MIMIC' else mm
        prob = data['predicted_prob_within_cv'] if cohort != 'MIMIC' else data['oof_prob_paper']
        fpr, tpr, _ = roc_curve(data.MACE, prob)
        auc = roc_auc_score(data.MACE, prob)
        aucs[cohort] = float(auc)
        assert round(auc, 3) == {'SCH': 0.657, 'WHU': 0.84, 'MIMIC': 0.742}[cohort]
        ax = axes[0, j]
        ax.plot([0, 1], [0, 1], color='.65', linestyle='--', linewidth=0.8)
        ax.plot(fpr, tpr, color=color, linewidth=1.7, label=f'Original model (AUC {auc:.3f})')
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='False-positive rate', ylabel='True-positive rate')
        ax.set_title(cohort if cohort != 'MIMIC' else 'MIMIC-III', fontweight='bold', pad=8)
        ax.text(-0.22, 1.08, chr(65 + j), transform=ax.transAxes, fontweight='bold', fontsize=11)
        ax.legend(loc='lower right', fontsize=7.5, frameon=False)
        ax = axes[1, j]
        ax.plot([0, 1], [0, 1], color='.65', linestyle='--', linewidth=0.8)
        for model, col, label in [('base_raw', '#CB4542', 'Base raw'), ('base_platt', '#287A45', 'Same base + held-out Platt')]:
            rows = cal[(cal.run == cohort + '_original') & (cal.model == model)].sort_values('bin')
            assert len(rows) > 0
            x = rows.mean_probability.to_numpy()
            y = rows.observed_fraction.to_numpy()
            ax.plot(x, y, color=col, linewidth=1.1, label=label)
            for k, row in rows.iterrows():
                sparse = str(row['sparse']).lower() == 'true'
                ax.errorbar(row.mean_probability, row.observed_fraction, yerr=[[row.observed_fraction - row.observed_fraction_ci_low], [row.observed_fraction_ci_high - row.observed_fraction]], fmt='o', ms=3.2, color=col, mfc='white' if sparse else col, mec=col, linewidth=0.7, capsize=1.3)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean probability in fixed bin', ylabel='Observed positive-label fraction')
        ax.set_title('Additional patient-isolated calibration', fontsize=9, pad=9)
        ax.text(-0.22, 1.08, chr(68 + j), transform=ax.transAxes, fontweight='bold', fontsize=11)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.065), ncol=2, frameon=False, fontsize=8)
    fig.text(0.5, 0.045, 'Top: original cached diagnosis-classification ROC; not a new independent-patient result.', ha='center', fontsize=8)
    fig.text(0.5, 0.024, 'Bottom: same base model before/after held-out calibration; pointwise 95% patient-cluster intervals.', ha='center', fontsize=8)
    fig.text(0.5, 0.004, 'Base models use less training data than the new primary full-outer-training ensemble. No future-event risk or clinical-utility claim.', ha='center', fontsize=7.5)
    save(fig, 'Figure4_incremental')
    return {'source_identities': [identity(cnpath), identity(mpath), identity(cpath)], 'original_auc': aucs, 'private_rows_exported': False, 'model_refitting': False, 'roc_source_columns': {'China': 'predicted_prob_within_cv', 'MIMIC': 'oof_prob_paper'}, 'removed_original_panels': ['A-C proxy and old age curves; their methods/input provenance were not sufficiently verified', 'D-F decision curves without validated clinical decisions or independent calibration'], 'new_panels': 'D-F: frozen base_raw versus base_platt, fixed original bins and pointwise intervals'}

def figure_s11():
    cp = OLD / 'evidence/china_preparation.json'
    mp = OLD / 'evidence/mimic_preparation.json'
    china = _release_resolve_config(json.loads(cp.read_text(encoding='utf-8')))
    mimic = _release_resolve_config(json.loads(mp.read_text(encoding='utf-8')))
    assert len(china['feature_names']) == 104 and len(mimic['matrix']['per_feature']) == 43
    arrays = []
    cohorts = {x['center']: x for x in china['cohorts']}
    for c in ['SCH', 'WHU']:
        rows = cohorts[c]['features']
        assert [r['name'] for r in rows] == china['feature_names']
        arrays.append([r['patient_missing_fraction'] * 100 for r in rows])
    mvals = np.array([[r['missing_fraction'] * 100 for r in mimic['matrix']['per_feature']]])
    fig = plt.figure(figsize=(10.3, 5.1))
    gs = fig.add_gridspec(2, 1, left=0.17, right=0.88, bottom=0.24, top=0.89, hspace=0.77, height_ratios=[2, 1])
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(arrays, cmap='YlGnBu', vmin=0, vmax=100, aspect='auto', interpolation='nearest')
    ax.set_yticks([0, 1], ['SCH\n38,509 patients', 'WHU\n2,562 patients'])
    ticks = [0] + list(range(4, 104, 5)) + [103]
    ax.set_xticks(ticks, [str(t + 1) for t in ticks], fontsize=7)
    ax.set_xlabel('Original Chinese input index (104 nominal columns)')
    ax.set_title('A  China: missing after patient-level mean aggregation', loc='left', fontweight='bold', pad=10)
    ax = fig.add_subplot(gs[1])
    ax.imshow(mvals, cmap='YlGnBu', vmin=0, vmax=100, aspect='auto', interpolation='nearest')
    ax.set_yticks([0], ['MIMIC-III\n9,946 admissions'])
    ticks = [0] + list(range(4, 43, 5)) + [42]
    ax.set_xticks(ticks, [str(t + 1) for t in ticks], fontsize=7)
    ax.set_xlabel('Original MIMIC input index (43 nominal slots; 17 entirely missing)')
    ax.set_title('B  MIMIC-III: missing after admission-median aggregation', loc='left', fontweight='bold', pad=10)
    cax = fig.add_axes([0.9, 0.24, 0.016, 0.65])
    fig.colorbar(im, cax=cax, label='Missing before imputation (%)')
    fig.text(0.54, 0.12, 'Indexes refer to the complete source-field inventories; columns are not a shared cross-centre biomarker panel.', ha='center', fontsize=8)
    fig.text(0.54, 0.078, 'Includes laboratory, demographic, regional and constant text-derived slots. Missingness is not the proportion with a result.', ha='center', fontsize=8)
    fig.text(0.54, 0.036, 'No patient rows shown. Entirely missing placeholders are retained in the nominal input count.', ha='center', fontsize=8)
    save(fig, 'FigureS11_incremental')
    inventory = {'China': [{'index': i + 1, 'name': n, 'SCH_missing_percent': arrays[0][i], 'WHU_missing_percent': arrays[1][i]} for i, n in enumerate(china['feature_names'])], 'MIMIC': [{'index': r['index'] + 1, 'name': r['name'], 'missing_percent': r['missing_fraction'] * 100} for r in mimic['matrix']['per_feature']]}
    (ROOT / 'S11_feature_index.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'source_identities': [identity(cp), identity(mp)], 'China_columns': 104, 'MIMIC_columns': 43, 'MIMIC_all_missing_columns': int(np.sum(mvals == 100)), 'definition': 'Fraction missing after original source-specific aggregation and before imputation', 'private_rows_exported': False}
if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    result = {'Figure4': figure4(), 'FigureS11': figure_s11(), 'script': identity(Path(__file__))}
    (ROOT / 'additional_panel_provenance.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'completed': ['Figure4_incremental.pdf', 'FigureS11_incremental.pdf'], 'original_auc': result['Figure4']['original_auc']}))
