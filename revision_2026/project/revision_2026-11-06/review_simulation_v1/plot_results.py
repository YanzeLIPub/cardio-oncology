from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import sys, json, csv
R = Path(__file__).resolve().parent
ROOT = R.parents[1]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = R / 'figures'
OUT.mkdir(exist_ok=True)

def rows(name):
    with (R / 'aggregate' / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))
metrics = rows('policy_metrics.csv')
examples = rows('illustrative_10percent_capacity.csv')
action = rows('actionability_sensitivity.csv')
runs = ['SCH_original', 'WHU_original', 'MIMIC_original', 'MIMIC_exclude_AST_HDL']
labels = ['SCH\nPatients', 'WHU\nPatients', 'MIMIC\nAdmissions', 'MIMIC without AST/HDL\nAdmissions']
short = ['SCH', 'WHU', 'MIMIC', 'MIMIC\nwithout AST/HDL']
colors = {'primary_raw': '#0072B2', 'age': '#D55E00', 'random': '#777777'}
names = {'primary_raw': 'Main ensemble', 'age': 'Age only', 'random': 'Random review (expected)'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def save(fig, stem):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / (stem + '.' + ext), dpi=200, bbox_inches='tight')
    plt.close(fig)
fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0), layout='constrained')
for i, (ax, run, label) in enumerate(zip(axes.flat, runs, labels)):
    for model in colors:
        data = sorted([v for v in metrics if v['run'] == run and v['model'] == model and (v['mode'] == 'capacity')], key=lambda v: float(v['setting']))
        x = np.array([float(v['setting']) * 100 for v in data])
        y = np.array([float(v['selected_positive_per_1000']) for v in data])
        low = np.array([float(v['selected_positive_per_1000_low']) for v in data])
        high = np.array([float(v['selected_positive_per_1000_high']) for v in data])
        ax.plot(x, y, marker='o', ms=4, color=colors[model], label=names[model], lw=1.6, ls='--' if model == 'random' else '-')
        ax.fill_between(x, low, high, color=colors[model], alpha=0.14, lw=0)
    ax.set_title(f'{chr(97 + i)}  ' + label.replace('\n', ' — '), loc='left', fontweight='bold', fontsize=10)
    ax.set_xticks([5, 10, 20])
    ax.set_xlabel('Records reviewed (%)')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Positive records captured\nper 1,000 ' + ('admissions' if 'MIMIC' in run else 'patients'))
    ax.grid(axis='y', alpha=0.18)
handles, legend = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, legend, loc='outside lower center', ncol=3, frameon=False, fontsize=9)
fig.suptitle('Retrospective record review at fixed capacity', fontsize=13)
save(fig, 'review_capacity_yield')
fig, axs = plt.subplots(1, 3, figsize=(13.3, 4.5), layout='constrained', gridspec_kw={'width_ratios': [1.25, 1, 1.1]})
ax = axs[0]
x = np.arange(4)
width = 0.23
for offset, model in enumerate(['primary_raw', 'age', 'random']):
    data = [next((v for v in examples if v['run'] == run and v['model'] == model)) for run in runs]
    y = np.array([float(v['process_improvements_per_1000']) for v in data])
    lo = np.array([float(v['process_improvements_low']) for v in data])
    hi = np.array([float(v['process_improvements_high']) for v in data])
    ax.bar(x + (offset - 1) * width, y, width, color=colors[model], label=names[model], yerr=[y - lo, hi - y], error_kw={'elinewidth': 0.7, 'capsize': 2}, alpha=0.9)
ax.set_xticks(x, short, fontsize=8)
ax.set_ylabel('Assumed process improvements\nper 1,000 analysis units')
ax.set_title('a  Illustrative process benefit\n10% reviewed; a = 0.50, u = 0.80, b = 0.25', loc='left', fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.15)
ax.legend(fontsize=7, frameon=False, loc='upper left')
ax = axs[1]
h = np.linspace(0, 0.01, 101)
for q, col in zip([0.05, 0.1, 0.2], ['#56B4E9', '#0072B2', '#CC79A7']):
    ax.plot(h * 100, 1000 * q * h, color=col, label=f'{q:.0%} reviewed', lw=1.7)
ax.scatter([0.5], [0.5], color='#0072B2', s=24, zorder=3)
ax.set_xlabel('Assumed adverse-process probability\nper reviewed record (%)')
ax.set_ylabel('Assumed adverse-process consequences\nper 1,000 analysis units')
ax.set_title('b  Adverse-process consequences\nAssumed probability and review capacity', loc='left', fontsize=10, fontweight='bold')
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.15)
ax = axs[2]
for offset, ref, col in [(-0.15, 'age', '#D55E00'), (0.15, 'random', '#777777')]:
    data = [next((v for v in action if v['run'] == run and v['comparator'] == ref and (float(v['capacity']) == 0.1))) for run in runs]
    vals = [float(v['equal_benefit_actionability_ratio_model_to_comparator']) for v in data]
    ax.barh(x + offset, vals, height=0.28, color=col, label='Compared with ' + ('age only' if ref == 'age' else 'random review'))
    for y0, val in zip(x + offset, vals):
        ax.text(val + 0.02, y0, f'{val:.2f}', va='center', fontsize=8)
ax.set_yticks(x, short, fontsize=8)
ax.set_ylim(4.4, -0.5)
ax.axvline(1, color='black', lw=0.8, ls=':')
ax.set_xlim(0, 1.12)
ax.set_xlabel('Actionability ratio for equal assumed benefit\n(model-selected / comparator-selected)')
ax.set_title('c  Dependence on actionability\n10% reviewed; equal uptake and success', loc='left', fontsize=10, fontweight='bold')
ax.legend(fontsize=7, frameon=False, loc='lower right')
fig.suptitle('Hypothetical process effects of additional record review', fontsize=12)
save(fig, 'process_effect_scenarios')
(R / 'plot_environment.json').write_text(json.dumps({'python': sys.executable, 'matplotlib': matplotlib.__version__, 'numpy': np.__version__, 'dependency_directory': (ROOT / 'revision_2026-11-02/graphical_panels_v1/editorial/plot_dependencies').as_posix()}, indent=2), encoding='utf-8')
print('Saved two figure compositions as vector PDF/SVG and PNG.')
