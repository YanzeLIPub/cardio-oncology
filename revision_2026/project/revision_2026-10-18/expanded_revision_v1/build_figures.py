"""Additional scientific panels; never replaces the retained main figures."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
ROOT = Path(__file__).resolve().parent
AN = ROOT.parent / 'four_analysis_v1'
OUT = ROOT / 'figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
COLORS = ['#454545', '#976229', '#007C83', '#6B5B95', '#BE3C4B']

def rows(path):
    with (AN / path).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def one(data, **filters):
    found = [r for r in data if all((r[k] == str(v) for k, v in filters.items()))]
    assert len(found) == 1, filters
    return found[0]

def numbers(r):
    return np.array([float(r[k]) for k in ('estimate', 'ci_low', 'ci_high')])

def save(fig, name, source):
    OUT.mkdir(exist_ok=True, parents=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    paths = [AN / p for p in source]
    provenance = {'inputs': [{'path': p.as_posix()} for p in paths], 'figure': name, 'source': 'Executed aggregate outputs; no simulated data'}
    (OUT / f'{name}_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')

def c1():
    source = 'transfer_adaptation/aggregate/mean_repeat_metrics.csv'
    data = rows(source)
    direct_path = 'transfer_adaptation/aggregate/metrics_by_repeat.csv'
    direct = rows(direct_path)
    models = ['direct_source', 'platt_unconstrained', 'logistic_update_source_transform', 'local_logistic_source_transform', 'local_logistic_target_transform']
    labels = ['Direct source', 'Platt (slope unrestricted)', 'Source score + labs update', 'Local fit / source transform', 'Fully target-local logistic']
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.6))
    for row, (s, t) in enumerate((('MIMIC', 'WHU'), ('WHU', 'MIMIC'))):
        for col, (metric, label) in enumerate((('auc', 'ROC AUC'), ('average_precision', 'Average precision'), ('brier', 'Brier score'))):
            ax = axes[row, col]
            for m, lab, color in zip(models, labels, COLORS):
                if m == 'direct_source':
                    r = one(direct, source=s, target=t, population='all_original_target', method=m, metric=metric)
                    ax.axhline(float(r['estimate']), color=color, label=lab, linewidth=1.3, ls='--')
                    ax.axhspan(float(r['ci_low']), float(r['ci_high']), color=color, alpha=0.08, linewidth=0)
                    continue
                subset = [r for r in data if r['source'] == s and r['target'] == t and (r['population'] == 'all_original_target') and (r['method'] == m) and (r['metric'] == metric)]
                subset.sort(key=lambda r: int(r['sample_size_patients_per_fold']))
                values = np.array([numbers(r) for r in subset])
                x = np.array([int(r['sample_size_patients_per_fold']) for r in subset])
                assert len(values) == 3, (s, t, m, metric)
                ax.plot(x, values[:, 0], 'o-', color=color, label=lab, markersize=4, linewidth=1.3)
                ax.fill_between(x, values[:, 1], values[:, 2], color=color, alpha=0.08, linewidth=0)
            ax.set_xscale('log')
            ax.set_xticks([50, 200, 800], ['50', '200', '800'])
            ax.set_title(f"{'ABCDEF'[row * 3 + col]}  {s} to {t}", loc='left', fontweight='bold')
            ax.set_ylabel(label)
            ax.set_xlabel('Labelled target patients per fold')
            ax.grid(axis='y', color='#DFDFDF', linewidth=0.6)
            if metric == 'auc':
                ax.axhline(0.5, color='#777777', ls=':', lw=0.8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.005), fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.19, hspace=0.42, wspace=0.36)
    save(fig, 'FigureC1_adaptation', [source, direct_path])

def c2():
    source = 'high_coverage/results/high_coverage_metrics.csv'
    delta = 'high_coverage/results/high_coverage_paired_comparisons.csv'
    data, paired = (rows(source), rows(delta))
    models = ['all_labs', 'coverage_80', 'coverage_50']
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.6))
    for col, c in enumerate(('SCH', 'WHU', 'MIMIC')):
        ax = axes[0, col]
        for i, m in enumerate(models):
            val, lo, hi = numbers(one(data, cohort=c, model=m, metric='auc'))
            ax.errorbar(i, val, yerr=[[val - lo], [hi - val]], fmt='o', color=COLORS[i], capsize=3)
        ax.set_xticks(range(3), ['All eligible', '80%+', '50%+'])
        ax.set_title(f"{'ABC'[col]}  {c}", loc='left', fontweight='bold')
        ax.set_ylabel('ROC AUC')
        ax.grid(axis='y', color='#DFDFDF', linewidth=0.6)
        ax = axes[1, col]
        for i, m in enumerate(models[1:]):
            val, lo, hi = numbers(one(paired, cohort=c, first=m, second=models[0], metric='delta_auc'))
            ax.errorbar(i, val, yerr=[[val - lo], [hi - val]], fmt='o', color=COLORS[i + 1], capsize=3)
        ax.axhline(0, color='#777777', lw=0.8, ls=':')
        ax.set_xticks(range(2), ['80%+ minus all', '50%+ minus all'])
        ax.set_title(f"{'DEF'[col]}  {c}", loc='left', fontweight='bold')
        ax.set_ylabel('Paired change in AUC')
        ax.grid(axis='y', color='#DFDFDF', linewidth=0.6)
        ax.margins(x=0.5)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.96, hspace=0.4, wspace=0.35)
    save(fig, 'FigureC2_coverage', [source, delta])

def c4():
    source = 'nested_tuning/results/metrics.csv'
    delta = 'nested_tuning/results/paired_comparisons.csv'
    data, paired = (rows(source), rows(delta))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    labels = ['Matched nested fixed', 'Nested selected', 'Previous validation']
    models = ['nested_fixed_ensemble', 'nested_selected_ensemble', 'previous_ensemble']
    for i, c in enumerate(('SCH', 'WHU', 'MIMIC')):
        for j, (m, lab) in enumerate(zip(models, labels)):
            val, lo, hi = numbers(one(data, cohort=c, model=m, metric='auc'))
            axes[0].errorbar(i + (j - 1) * 0.17, val, yerr=[[val - lo], [hi - val]], fmt='o', color=COLORS[j], capsize=3, label=lab if i == 0 else None)
        val, lo, hi = numbers(one(paired, cohort=c, first='nested_selected_ensemble', second='nested_fixed_ensemble', metric='delta_auc'))
        axes[1].errorbar(i, val, yerr=[[val - lo], [hi - val]], fmt='o', color=COLORS[2], capsize=3)
    for ax in axes:
        ax.set_xticks(range(3), ['SCH', 'WHU', 'MIMIC'])
        ax.grid(axis='y', color='#DFDFDF', linewidth=0.6)
        ax.margins(x=0.2)
    axes[0].set_title('A  Ensemble discrimination', loc='left', fontweight='bold')
    axes[0].set_ylabel('ROC AUC')
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0))
    axes[1].set_title('B  Selected minus matched fixed', loc='left', fontweight='bold')
    axes[1].set_ylabel('Paired change in AUC')
    axes[1].axhline(0, color='#777777', lw=0.8, ls=':')
    fig.tight_layout(w_pad=3, rect=(0, 0.1, 1, 1))
    save(fig, 'FigureC4_nested_tuning', [source, delta])

def c3():
    source = 'shap_seeds/results/seed_rank_stability.csv'
    retain = 'shap_seeds/results/retention.csv'
    summary = 'shap_seeds/results/publication_summary.json'
    data, retention = (rows(source), rows(retain))
    report = _release_resolve_config(json.loads((AN / summary).read_text(encoding='utf-8')))
    seeds = report['seeds']
    fig = plt.figure(figsize=(10.4, 8.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.65], hspace=0.48, wspace=0.34)
    for col, c in enumerate(('SCH', 'WHU')):
        ax = fig.add_subplot(grid[0, col])
        matrix = np.eye(4)
        for r in data:
            if r['cohort'] == c:
                i, j = (seeds.index(int(r['first_seed'])), seeds.index(int(r['second_seed'])))
                matrix[i, j] = matrix[j, i] = float(r['rho_nonzero'])
        ax.imshow(matrix, vmin=0.8, vmax=1, cmap='YlGnBu', aspect='equal')
        ax.set_xticks(range(4), seeds)
        ax.set_yticks(range(4), seeds)
        ax.set_xlabel('Training seed')
        ax.set_ylabel('Training seed')
        ax.set_title(f"{'AB'[col]}  {c} importance rank correlation", loc='left', fontweight='bold')
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center', color='white' if matrix[i, j] > 0.92 else 'black', fontsize=10)
    features = report['original_eight']
    names = ['Uric acid', 'Phosphorus', 'Hemoglobin', 'Apolipoprotein B', 'Ferritin', 'Age', 'Regional hospital / health-centre count', 'Regional physician count']
    expected = ['尿酸', '磷', '血红蛋白', '载脂蛋白B', '铁蛋白', '年龄', '医院、卫生院数(个)', '医生数(人)']
    assert features == expected
    matrix = np.zeros((8, 4), dtype=int)
    for i, f in enumerate(features):
        for j, s in enumerate(seeds):
            r = one(retention, feature=f, seed=s)
            if r['opposite_both_q_and_five_fold_agreement'] == 'True':
                matrix[i, j] = 1 if r['same_pooled_signs_as_seed42'] == 'True' else 2
    ax = fig.add_subplot(grid[1, :])
    cmap = ListedColormap(['#F0F0F0', '#218C8C', '#C68438'])
    ax.imshow(matrix, cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), aspect='auto')
    ax.set_yticks(range(8), names, fontsize=10)
    ax.set_xticks(range(4), seeds)
    ax.set_xlabel('Training seed')
    ax.set_title('C  Retention of the eight seed-42 features', loc='left', fontweight='bold', pad=12)
    for i in range(8):
        for j in range(4):
            value = matrix[i, j]
            ax.text(j, i, ['No', 'Yes', 'Flipped'][value], ha='center', va='center', color='white' if value == 1 else 'black', fontsize=10)
    fig.subplots_adjust(left=0.3, right=0.98, top=0.96, bottom=0.065)
    save(fig, 'FigureC3_shap_seeds', [source, retain, summary])
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('figure', choices=('c1', 'c2', 'c3', 'c4'))
    args = p.parse_args()
    globals()[args.figure]()
