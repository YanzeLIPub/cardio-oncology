"""Create a supplementary figure from verified aggregate probabilities only."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent

def ident(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def main():
    source = ROOT / 'results/era_metrics.csv'
    frame = pd.read_csv(source)
    models = ['full_104', 'no_regional_99', 'regional_only_5']
    colors = ['#275D8C', '#167D6A', '#BE5834']
    labels = ['Full\n104 inputs', 'No regional\n99 inputs', 'Regional only\n5 inputs']
    metrics = [('auc', 'ROC AUC', (0.2, 1.0)), ('average_precision', 'Average precision', (0, 0.42)), ('brier', 'Brier score', (0, 0.25))]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 9, 'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(3, 2, figsize=(174 / 25.4, 190 / 25.4))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.9, bottom=0.08, hspace=0.47, wspace=0.26)
    fig.suptitle('Acquisition-era diagnostic extrapolation', fontsize=11, y=0.985)
    for row, (metric, ylabel, limits) in enumerate(metrics):
        for col, cohort in enumerate(('SCH', 'WHU')):
            ax = axes[row, col]
            rows = frame.loc[frame.cohort.eq(cohort) & frame.metric.eq(metric)].set_index('model').loc[models]
            for x, model in enumerate(models):
                r = rows.loc[model]
                ax.errorbar(x, r.estimate, yerr=[[r.estimate - r.ci_low], [r.ci_high - r.estimate]], color=colors[x], marker='o', markersize=4.5, capsize=3, elinewidth=1.1)
            ax.set_xticks(np.arange(3), labels)
            ax.set_xlim(-0.55, 2.55)
            ax.set_ylim(*limits)
            ax.set_ylabel(ylabel)
            ax.grid(axis='y', color='#DADADA', linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)
            ax.text(-0.16, 1.05, chr(65 + 2 * row + col), transform=ax.transAxes, fontsize=10, fontweight='bold')
            if row == 0:
                n, e = (int(rows.iloc[0].test_patients), int(rows.iloc[0].test_positive_labels))
                ax.set_title(f'{cohort}: {n:,} patients, {e} positive labels', pad=16)
                ax.axhline(0.5, color='#777777', linestyle='--', linewidth=0.7)
    dest = ROOT / 'figures'
    dest.mkdir(exist_ok=True)
    files = []
    for extension in ('png', 'pdf', 'svg'):
        file = dest / f'FigureD2_acquisition_era.{extension}'
        assert not file.exists()
        fig.savefig(file, dpi=300, facecolor='white')
        files.append(ident(file))
    plt.close(fig)
    caption = 'Figure D2. Acquisition-era extrapolation with regional-feature ablations. Models were trained on patients whose entire processed laboratory record preceded an outcome-blind date boundary (SCH, 1 May 2024; WHU, 1 January 2019) and tested on patients whose entire laboratory record was at or after it. Patients crossing the boundary were excluded. Points and whiskers show ROC AUC (A-B), average precision (C-D), and Brier score (E-F), with 95% percentile intervals from 2,000 patient-bootstrap replicates of fixed held-out probabilities. All three uncalibrated 50:50 XGBoost/RandomForest models used the same patient roles and training-only stopping selection. The horizontal dashed line in A-B denotes AUC 0.5. Lower Brier scores are better. Full-record diagnosis labels and nonlaboratory information were not verified as available before the boundary. These are retrospective diagnostic stress tests, not prospective temporal validation or estimates of warning lead time. The WHU test set contains only 18 positive labels.'
    report = {'source': ident(source), 'protocol': ident(ROOT / 'protocol.json'), 'script': ident(__file__), 'width_mm': 174, 'height_mm': 190, 'minimum_font_points': 8, 'caption': caption, 'outputs': files}
    with (dest / 'FigureD2_provenance.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
if __name__ == '__main__':
    main()
