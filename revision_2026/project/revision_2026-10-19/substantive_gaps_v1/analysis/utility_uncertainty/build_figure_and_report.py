"""Publication asset from aggregate outputs only; does not evaluate models."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent

def read(name):
    with (ROOT / name).open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ('threshold', 'estimate', 'ci_low', 'ci_high'):
            if key in row:
                row[key] = float(row[key])
    return rows
curves, contrasts = (read('dca_curves.csv'), read('dca_contrasts.csv'))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(4, 2, figsize=(7.4, 10.3), layout='constrained')
for row, (run, title) in enumerate((('SCH_original', 'SCH'), ('WHU_original', 'WHU'), ('MIMIC_original', 'MIMIC'), ('MIMIC_exclude_AST_HDL', 'MIMIC: AST/HDL excluded'))):
    ax, bx = axes[row]
    for model, label, color, style in (('base_raw', 'Base raw', '#0072B2', '-'), ('base_platt', 'Independent Platt', '#D55E00', '-'), ('treat_all', 'Treat all', '#777777', '--'), ('treat_none', 'Treat none', '#222222', ':')):
        series = [r for r in curves if r['run'] == run and r['model'] == model and (r['threshold'] <= 0.5)]
        x = [r['threshold'] for r in series]
        ax.plot(x, [r['estimate'] for r in series], color=color, ls=style, label=label, lw=1.2)
        if model in ('base_raw', 'base_platt'):
            ax.fill_between(x, [r['ci_low'] for r in series], [r['ci_high'] for r in series], color=color, alpha=0.18, lw=0)
    series = [r for r in contrasts if r['run'] == run and r['contrast'] == 'base_platt_minus_base_raw' and (r['threshold'] <= 0.5)]
    x = [r['threshold'] for r in series]
    bx.plot(x, [r['estimate'] for r in series], color='#009E73', lw=1.3)
    bx.fill_between(x, [r['ci_low'] for r in series], [r['ci_high'] for r in series], color='#009E73', alpha=0.2, lw=0)
    bx.axhline(0, color='#555555', lw=0.7, ls=':')
    ax.set_title(f'{chr(65 + 2 * row)}  {title}', loc='left', fontsize=10)
    bx.set_title(f'{chr(66 + 2 * row)}  Platt minus raw', loc='left', fontsize=10)
    ax.set_ylabel('Diagnostic-label net benefit')
    bx.set_ylabel('Paired net-benefit difference')
    for a in (ax, bx):
        a.set_xlim(0.01, 0.5)
        a.set_xlabel('Numerical threshold probability')
        a.grid(axis='y', alpha=0.18, lw=0.5)
    if row == 0:
        ax.legend(fontsize=7, frameon=False, loc='lower left')
for ext in ('pdf', 'png', 'svg'):
    fig.savefig(ROOT / f'FigureE2_diagnostic_decision_curves.{ext}', dpi=240)
plt.close(fig)
report = {'results': 'Independent calibration changed diagnostic-label net benefit, with substantially larger raw-versus-Platt differences in SCH than MIMIC. At the prespecified numerical threshold of 0.10, Platt-minus-raw net-benefit differences were 0.07283 (95% CI 0.07092 to 0.07483) in SCH, 0.02333 (0.01613 to 0.02975) in WHU, 0.00032 (-0.00021 to 0.00077) in MIMIC and 0.00013 (-0.00039 to 0.00058) after excluding AST/HDL. These contrasts are not recommendations for a 10% clinical threshold. Continuous NRI of the full-training ensemble relative to age-only classification was 0.000 (0.000 to 0.000), 0.410 (0.347 to 0.471), 0.486 (0.446 to 0.526) and 0.479 (0.438 to 0.519), respectively.', 'nri_caution': 'In SCH, every full-training ensemble probability exceeded its age-only comparator probability. Consequently, the positive-label NRI component was +1 and the negative-label component was -1, summing to zero in every conditional bootstrap sample. This degenerate interval reflects a fixed probability-ordering pattern, not certainty about model utility or equal predictive quality. Together with the negative non-event component in WHU, it demonstrates why continuous NRI and class-weighted raw probabilities should not be interpreted as clinical improvement. Calibration-dependent NRI comparisons do not validate an intervention or the original clinical proxies.', 'interpretation': 'The newly computed intervals close the statistical-uncertainty component of the review request for these independent diagnostic-label analyses only. No clinical action, clinically acceptable threshold, event-free prediction time, full negative ascertainment or treatment effect was established, and historical same-OOF isotonic curves remain unvalidated.', 'visual_review': {'main_01_50_grid': 'All eight panels inspected at readable resolution; legends, axes, titles and ribbons are non-overlapping', 'full_grid': 'Separate full-grid figure retains all 99 thresholds and uses a labelled symmetric-log net-benefit axis'}, 'cohorts': read('cohorts.csv'), 'nri_continuous': read('nri_continuous.csv'), 'paired_calibration_selected': [r for r in read('dca_selected_thresholds.csv') if r['contrast'] == 'base_platt_minus_base_raw'], 'platt_selected': [r for r in read('dca_selected_curves.csv') if r['model'] == 'base_platt'], 'figure': 'FigureE2_diagnostic_decision_curves.png'}
with (ROOT / 'publication_results.json').open('x', encoding='utf-8') as f:
    json.dump(report, f, indent=2)
    f.write('\n')
print('Publication figure triplet and aggregate results text written.')
