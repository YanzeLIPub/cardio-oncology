"""Replot manuscript figures from aggregate results only; no patient data or fitting."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import csv
import hashlib
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'original_v5_independent_validation_v1'
OUT = ROOT / 'figures_main'
OUT.mkdir(exist_ok=True)
METRICS = OLD / 'results/validation_metrics.csv'
CURVES = ROOT / 'calibration/calibration_curve_intervals.csv'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
with METRICS.open(newline='', encoding='utf-8') as f:
    metrics = list(csv.DictReader(f))
with CURVES.open(newline='', encoding='utf-8') as f:
    curves = list(csv.DictReader(f))
RUNS = ['SCH_original', 'WHU_original', 'MIMIC_original']
TITLES = ['SCH\n38,509 patients', 'WHU\n2,562 patients', 'MIMIC-III\n9,946 admissions / 6,783 patients']

def identity(p):
    p = Path(p)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def metric(run, model, kind='auc'):
    rows = [r for r in metrics if r['run'] == run and r['model'] == model and (r['metric'] == kind)]
    assert len(rows) == 1
    return np.array([float(rows[0][k]) for k in ('estimate', 'ci_low', 'ci_high')])

def save(fig, n):
    fig.savefig(OUT / f'Figure_{n}.png', dpi=300, facecolor='white')
    fig.savefig(OUT / f'Figure_{n}.svg', facecolor='white')
    plt.close(fig)
fig, (ax, values_ax) = plt.subplots(1, 2, figsize=(10.5, 4.5), gridspec_kw={'width_ratios': [2.1, 1.25]})
for i, run in enumerate(RUNS):
    val, low, high = metric(run, 'primary_raw')
    ax.errorbar(val, i - 0.09, xerr=[[val - low], [high - val]], fmt='o', color='#007C91', capsize=4, label='New patient-isolated internal validation' if i == 0 else None)
    hist = metric(run, 'original')[0]
    ax.plot(hist, i + 0.09, 'D', mfc='white', mec='#555555', label='Historical analysis point estimate' if i == 0 else None)
    values_ax.text(0, i, f'Historical {hist:.3f}\nNew {val:.3f} ({low:.3f}-{high:.3f})', va='center', fontsize=10)
ax.set(yticks=range(3), yticklabels=TITLES, xlim=(0.5, 0.9), ylim=(2.5, -0.5), xlabel='ROC AUC')
values_ax.set(xlim=(0, 1), ylim=(2.5, -0.5))
values_ax.axis('off')
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax.axvline(0.5, color='#999999', ls='--', lw=0.8)
ax.grid(axis='x', alpha=0.18)
fig.suptitle('Recorded cardiovascular diagnosis discrimination', fontsize=14)
fig.legend(loc='lower center', bbox_to_anchor=(0.56, 0.07), ncol=1, frameon=False, fontsize=9)
fig.text(0.5, 0.018, 'New 95% CIs: 2,000 patient-cluster resamples of fixed OOF predictions. Original labels retained.', ha='center', fontsize=9)
fig.tight_layout(rect=(0, 0.19, 1, 0.94))
save(fig, 1)
matrix = np.array([[0.657, 0.438, 0.5], [0.485, 0.84, 0.494], [0.559, 0.444, 0.742]])
fig, ax = plt.subplots(figsize=(7.7, 5.4))
im = ax.imshow(matrix, vmin=0.4, vmax=0.85, cmap='YlGnBu')
for i in range(3):
    for j in range(3):
        text = f'{matrix[i, j]:.3f}\n' + ('within cohort' if i == j else 'transfer')
        ax.text(j, i, text, ha='center', va='center', fontsize=12, color='white' if matrix[i, j] > 0.7 else 'black')
ax.set(xticks=range(3), xticklabels=['SCH', 'WHU', 'MIMIC-III'], yticks=range(3), yticklabels=['SCH', 'WHU', 'MIMIC-III'], xlabel='Evaluation cohort', ylabel='Training cohort')
ax.tick_params(length=0)
fig.colorbar(im, ax=ax, shrink=0.75, label='Historical ROC AUC')
fig.suptitle('Historical within cohort and transfer results', fontsize=14)
fig.text(0.5, 0.065, 'Original values retained; different units, labels and feature mappings.', ha='center', fontsize=9)
fig.text(0.5, 0.025, 'This comparison does not identify a causal effect of care architecture.', ha='center', fontsize=9)
fig.tight_layout(rect=(0, 0.1, 1, 0.94))
save(fig, 2)
models = ['primary_raw', 'logistic', 'age', 'missingness', 'xgb_full', 'rf_full']
labels = ['XGBoost + RF', 'Logistic', 'Age only', 'Missingness only', 'XGBoost', 'Random forest']
colors = ['#007C91', '#317847', '#777777', '#B44748', '#4F78B5', '#A6761D']
fig, axes = plt.subplots(1, 3, figsize=(11.4, 5.1), sharey=True)
for ax, run, title in zip(axes, RUNS, TITLES):
    for i, (model, color) in enumerate(zip(models, colors)):
        val, low, high = metric(run, model)
        ax.errorbar(val, i, xerr=[[val - low], [high - val]], fmt='o', color=color, capsize=3)
        ax.text(0.985, i, f'{val:.3f}', ha='right', va='center', fontsize=9)
    ax.set(yticks=range(6), yticklabels=labels, xlim=(0.48, 1.0), ylim=(5.65, -0.6), xlabel='ROC AUC', title=title)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.axvline(0.5, color='#999999', ls='--', lw=0.8)
    ax.grid(axis='x', alpha=0.18)
fig.suptitle('Patient isolated comparison on original inputs and labels', fontsize=14)
fig.text(0.5, 0.025, 'All models use the same outer test patients. 95% patient-cluster CIs condition on fixed OOF predictions.', ha='center', fontsize=9)
fig.tight_layout(rect=(0, 0.07, 1, 0.93))
save(fig, 3)
calmodels = ['primary_raw', 'base_raw', 'base_platt']
calcolors = {'primary_raw': '#007C91', 'base_raw': '#B44748', 'base_platt': '#317847'}
markers = {'primary_raw': 'o', 'base_raw': '^', 'base_platt': 's'}
calnames = {'primary_raw': 'Primary raw (full outer train)', 'base_raw': 'Base raw', 'base_platt': 'Same base + held-out Platt'}
fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.7))
handles = []
for ax, run, title in zip(axes, RUNS, ['SCH', 'WHU', 'MIMIC-III']):
    ax.plot([0, 1], [0, 1], color='#777777', ls='--', lw=0.8)
    for model in calmodels:
        rows = sorted([r for r in curves if r['run'] == run and r['model'] == model], key=lambda r: int(r['bin']))
        x = np.array([float(r['mean_probability']) for r in rows])
        y = np.array([float(r['observed_fraction']) for r in rows])
        line, = ax.plot(x, y, color=calcolors[model], lw=1.1)
        ax.vlines(x, [float(r['observed_fraction_ci_low']) for r in rows], [float(r['observed_fraction_ci_high']) for r in rows], color=calcolors[model], lw=1.25, alpha=0.8)
        for sparse in (False, True):
            choose = np.array([(r['sparse'].lower() == 'true') == sparse for r in rows])
            ax.scatter(x[choose], y[choose], s=24, marker=markers[model], facecolors='white' if sparse else calcolors[model], edgecolors=calcolors[model], zorder=3)
        if len(handles) < 3:
            handles.append(line)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), title=title, xlabel='Mean probability in fixed bin', ylabel='Observed recorded-label fraction')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(alpha=0.15)
fig.legend(handles, [calnames[m] for m in calmodels], loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.09), fontsize=9)
fig.text(0.5, 0.055, 'Pointwise 95% patient-cluster intervals; fixed OOF predictions and original bins. Open markers: sparse bins.', ha='center', fontsize=9)
fig.text(0.5, 0.02, 'Primary and base models use different training amounts. Recorded diagnoses, not prospective event risk.', ha='center', fontsize=9)
fig.suptitle('Calibration assessment with conditional uncertainty', fontsize=14)
fig.tight_layout(rect=(0, 0.18, 1, 0.94))
save(fig, 4)
for ext in ('png', 'svg'):
    shutil.copy2(ROOT / f'mimic/figures/MIMIC_measurement_window_AUC.{ext}', OUT / f'Figure_5.{ext}')
manifest = {'aggregate_inputs': [identity(METRICS), identity(CURVES)], 'figures': [identity(OUT / f'Figure_{n}.{ext}') for n in range(1, 6) for ext in ('png', 'svg')], 'no_models_trained': True, 'patient_level_data_read': False, 'historical_transfer_matrix': matrix.tolist(), 'figure_4_logic': 'Same fixed-bin pointwise patient-cluster CIs as calibration source; redrawn solely to provide matching SVG and PNG.'}
(OUT / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps({'figures_created': 10, 'output': OUT.as_posix()}))
