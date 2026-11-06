"""Plot frozen aggregate results; preserve four main figures byte for byte."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import hashlib
import csv
import io
import json
import shutil
import sys
if '--main' in sys.argv:
    from pypdf import PdfReader, PdfWriter, Transformation
    from pypdf._page import PageObject
    from reportlab.pdfgen import canvas
    import pypdfium2 as pdfium
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    import pandas as pd
ROOT = Path(__file__).resolve().parent
AN = ROOT.parent / 'reviewer_priority_analyses_v1'
OLD = ROOT.parent / 'merged_revision_v1'
INC = ROOT.parent.parent / 'revision_2026-10-15/incremental_revision_v1/figures'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
INPUTS = {}
COLORS = {'SCH': '#22679a', 'WHU': '#ba553a', 'MIMIC': '#25855c'}
if '--main' not in sys.argv:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})

def sha(p):
    return None

def read(relative):
    p = AN / relative
    INPUTS[str(p)] = sha(p)
    return pd.read_csv(p)

def save(fig, name):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=240, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def forest(ax, rows, xlabel, title, reference=None, limits=None):
    for i, (label, val, lo, hi, color) in enumerate(rows):
        ax.errorbar(val, i, xerr=[[val - lo], [hi - val]], fmt='o', color=color, capsize=3, markersize=5, linewidth=1.3)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc='left', fontweight='bold', pad=12)
    if reference is not None:
        ax.axvline(reference, color='#777777', linestyle='--', linewidth=0.9)
    if limits:
        ax.set_xlim(*limits)
    ax.grid(axis='x', color='#dedede', linewidth=0.6)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_axisbelow(True)

def metric_row(row, label, color, key='roc_auc'):
    return (label, row[key], row[key + '_ci_low'], row[key + '_ci_high'], color)

def retain_main():
    history = AN / 'historical_transfer/transfer_auc_intervals.csv'
    INPUTS[str(history)] = sha(history)
    historical = list(csv.DictReader(history.open(encoding='utf-8-sig')))
    ci1 = next((r for r in historical if r['source'] == 'SCH' and r['target'] == 'WHU'))
    ci2 = next((r for r in historical if r['source'] == 'WHU' and r['target'] == 'SCH'))
    for n in (1, 3, 5):
        for ext in ('png', 'pdf'):
            src = INC / f'Figure{n}_incremental.{ext}'
            INPUTS[str(src)] = sha(src)
            shutil.copyfile(src, OUT / f'Figure{n}_priority.{ext}')
    for ext in ('png', 'pdf', 'svg'):
        src = OLD / f'figures/Figure4_merged.{ext}'
        INPUTS[str(src)] = sha(src)
        shutil.copyfile(src, OUT / f'Figure4_priority.{ext}')
    src = INC / 'Figure2_incremental.pdf'
    INPUTS[str(src)] = sha(src)
    p = PdfReader(src).pages[0]
    width, height = (float(p.mediabox.width), float(p.mediabox.height))
    footer = 48
    new = PageObject.create_blank_page(width=width, height=height + footer)
    new.merge_transformed_page(p, Transformation().translate(0, footer))
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(width, height + footer))
    c.setFont('Helvetica', 8.5)
    lines = [f"Recovered historical 95% CIs: SCH to WHU {float(ci1['ci_low']):.3f}-{float(ci1['ci_high']):.3f}; WHU to SCH {float(ci2['ci_low']):.3f}-{float(ci2['ci_high']):.3f}.", 'Target-patient bootstrap of fixed predictions; source training uncertainty is not included.', 'Matching historical predictions for four MIMIC-III directions were not recovered; no CIs are imputed.']
    for y, text in zip((34, 21, 8), lines):
        c.drawString(12, y, text)
    c.save()
    new.merge_page(PdfReader(buf).pages[0])
    w = PdfWriter()
    w.add_page(new)
    w.write(OUT / 'Figure2_priority.pdf')
    pdf = pdfium.PdfDocument(str(OUT / 'Figure2_priority.pdf'))
    pdf[0].render(scale=3).to_pil().save(OUT / 'Figure2_priority.png')
    pdf.close()

def common():
    transfer = read('common_transfer/aggregate/transfer_metrics.csv')
    local = read('common_transfer/aggregate/local_metrics.csv')
    subset = read('common_transfer/coverage_sensitivity_v1/metrics.csv')
    available = read('common_transfer/aggregate/feature_availability.csv')
    fig = plt.figure(figsize=(8.4, 7.0), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    ax = fig.add_subplot(grid[0, :])
    rows = []
    for source, target in (('WHU', 'MIMIC'), ('MIMIC', 'WHU')):
        a = transfer.query("source == @source and target == @target and model == 'ensemble_median'").iloc[0]
        b = subset.query("source == @source and target == @target and model == 'ensemble_median'").iloc[0]
        rows.extend([metric_row(a, f'{source} to {target}: full target', COLORS[target]), metric_row(b, f'{source} to {target}: covered subset', '#626262')])
    forest(ax, rows, 'AUROC (95% CI)', 'A   Fixed transferred model', 0.5, (0.3, 0.65))
    ax = fig.add_subplot(grid[1, 0])
    feats = list(available.feature.unique())
    labels = ['Albumin', 'Total protein', 'Hemoglobin', 'Hematocrit']
    assert len(feats) == 4
    for offset, site in ((-0.17, 'WHU'), (0.17, 'MIMIC')):
        vals = available.query('centre == @site').set_index('feature').loc[feats, 'missing_fraction'] * 100
        ax.barh(np.arange(4) + offset, vals, height=0.32, color=COLORS[site], label=site)
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel('Unavailable accepted values (%)')
    ax.set_title('B   Four accepted laboratory inputs', loc='left', fontweight='bold', pad=12)
    ax.legend(frameon=False, loc='lower right', fontsize=10)
    rows = []
    for target in ('WHU', 'MIMIC'):
        a = local.query("target == @target and model == 'ensemble_median'").iloc[0]
        b = subset.query("source == @target and target == @target and model == 'ensemble_median'").iloc[0]
        rows.extend([metric_row(a, f'{target}: full', COLORS[target]), metric_row(b, f'{target}: covered', '#626262')])
    forest(fig.add_subplot(grid[1, 1]), rows, 'AUROC (95% CI)', 'C   Local patient-isolated model', 0.5, (0.4, 0.75))
    save(fig, 'FigureB1_common_transfer')

def missingness():
    d = read('missingness/results/missingness_paired_comparisons.csv')
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6), layout='constrained')
    for ax, metric, model, title in zip(axes.flat, ('auc', 'auc', 'brier', 'brier'), ('value_indicator_ensemble', 'native_xgb') * 2, ('A   Added indicators', 'B   Native-NaN XGBoost', 'C   Added indicators', 'D   Native-NaN XGBoost')):
        rows = []
        key = 'delta_' + metric
        for site in ('SCH', 'WHU', 'MIMIC'):
            r = d.query('cohort == @site and first == @model and metric == @key').iloc[0]
            rows.append((site, r.estimate, r.ci_low, r.ci_high, COLORS[site]))
        forest(ax, rows, ('Change in AUROC' if metric == 'auc' else 'Change in Brier score') + ' (95% CI)', title, 0)
    save(fig, 'FigureB2_missingness')

def shap():
    cross = read('shap/results/cross_cohort_associations.csv')
    assoc = read('shap/results/feature_associations.csv')
    stability = read('shap/results/missingness_importance_stability.csv')
    names = {'尿酸': 'Uric acid', '磷': 'Phosphorus', '血红蛋白': 'Hemoglobin', '载脂蛋白B': 'Apolipoprotein B', '铁蛋白': 'Ferritin', '年龄': 'Age', '医院、卫生院数(个)': 'Regional hospitals', '医生数(人)': 'Regional physicians'}
    selected = cross[cross.opposite_both_q_and_five_fold_agreement]
    assert set(selected.feature) == set(names)
    fig = plt.figure(figsize=(8.4, 7.6), layout='constrained')
    grid = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(grid[:, 0])
    for i, r in enumerate(selected.itertuples()):
        for delta, cohort in ((-0.12, 'SCH'), (0.12, 'WHU')):
            ax.plot(getattr(r, cohort + '_rho'), i + delta, 'o', color=COLORS[cohort], label=cohort if i == 0 else None)
    ax.set_yticks(range(len(selected)), [names[x] for x in selected.feature])
    ax.invert_yaxis()
    ax.set_ylim(len(selected) + 0.15, -0.5)
    ax.set_xlim(-1, 1)
    ax.axvline(0, color='#999999', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Observed-value / SHAP rho')
    ax.set_title('A   Eight exploratory features', loc='left', fontweight='bold', pad=12)
    ax.legend(frameon=False, loc='lower center', ncol=2)
    ax = fig.add_subplot(grid[0, 1])
    labels, pos, neg, miss = ([], [], [], [])
    for name, label in (('神经元特异性烯醇化酶', 'NSE'), ('白蛋白', 'Albumin')):
        for site in ('SCH', 'WHU'):
            subset = assoc.query('feature == @name and cohort == @site')
            if subset.empty and label == 'NSE':
                subset = assoc[(assoc.cohort == site) & assoc.feature.str.contains('烯醇')]
            assert len(subset) == 1
            r = subset.iloc[0]
            labels.append(f'{label} {site}')
            pos.append(r.folds_positive)
            neg.append(r.folds_negative)
            miss.append(r.folds_untestable + r.folds_zero)
    y = np.arange(4)
    ax.barh(y, pos, color='#307d7e', label='Positive')
    ax.barh(y, neg, left=pos, color='#b85a4a', label='Negative')
    ax.barh(y, miss, left=np.array(pos) + neg, color='#cccccc', label='Not testable')
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 5)
    ax.set_xticks(range(6))
    ax.set_xlabel('Outer folds')
    ax.set_title('B   Direction stability', loc='left', fontweight='bold', pad=12)
    ax.legend(frameon=False, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, -0.43), ncol=2)
    ax = fig.add_subplot(grid[1, 1])
    for i, r in enumerate(stability.itertuples()):
        ax.scatter(r.importance_rank_rho_all, i, color=COLORS[r.cohort], s=30)
        ax.text(0.01, i, f'Top 10 overlap {r.top_10_overlap_n}/10', transform=ax.get_yaxis_transform(), fontsize=9, va='center')
    labels = [r.cohort + (' indicators' if 'indicator' in r.variant else ' native NaN') for r in stability.itertuples()]
    ax.set_yticks(range(len(stability)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0.7, 1.01)
    ax.set_xlabel('Value-channel importance rank rho')
    ax.set_title('C   Missingness sensitivity', loc='left', fontweight='bold', pad=12)
    save(fig, 'FigureB3_heldout_shap')

def regional():
    metrics = read('regional/results/regional_metrics.csv')
    delta = read('regional/results/regional_paired_comparisons.csv')
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7), layout='constrained')
    rows = []
    for site in ('SCH', 'WHU'):
        for model, label in (('baseline_ensemble', 'full'), ('no_regional', 'no regional'), ('regional_only', 'regional only')):
            r = metrics.query("cohort == @site and analysis == 'original_patient_roles' and model == @model and metric == 'auc'").iloc[0]
            rows.append((f'{site}: {label}', r.estimate, r.ci_low, r.ci_high, COLORS[site]))
    forest(axes[0, 0], rows, 'AUROC (95% CI)', 'A   Original patient folds', 0.5, (0.49, 0.89))
    for ax, metric, title in ((axes[0, 1], 'auc', 'B   Removing regional variables'), (axes[1, 1], 'brier', 'D   Removing regional variables')):
        rows = []
        key = 'delta_' + metric
        for site in ('SCH', 'WHU'):
            r = delta.query("cohort == @site and analysis == 'original_patient_roles' and first == 'no_regional' and metric == @key").iloc[0]
            rows.append((site, r.estimate, r.ci_low, r.ci_high, COLORS[site]))
        forest(ax, rows, ('Change in AUROC' if metric == 'auc' else 'Change in Brier score') + ' (95% CI)', title, 0)
    rows = []
    for site in ('SCH', 'WHU'):
        for unit, label in (('patient', 'patient CI'), ('outer_group', 'group CI')):
            r = delta.query("cohort == @site and analysis != 'original_patient_roles' and bootstrap_unit == @unit and metric == 'delta_auc'").iloc[0]
            rows.append((f'{site}: {label}', r.estimate, r.ci_low, r.ci_high, COLORS[site]))
    forest(axes[1, 0], rows, 'Change in AUROC (95% CI)', 'C   Grouped stress tests', 0)
    save(fig, 'FigureB4_regional')
if __name__ == '__main__':
    if '--main' in sys.argv:
        retain_main()
    else:
        common()
        missingness()
        shap()
        regional()
    report = {'inputs': INPUTS, 'inputs_unchanged': True, 'scope': 'Aggregate-only figures; original main 1,3,4,5 byte-preserved; Figure2 adds an uncertainty footer only.', 'outputs': {p.name: sha(p) for p in OUT.iterdir() if p.suffix in ('.png', '.pdf', '.svg') and p.name.startswith('FigureB') != ('--main' in sys.argv)}}
    report['runtime'] = {'executable': sys.executable, 'version': sys.version}
    (OUT / ('main_figure_manifest.json' if '--main' in sys.argv else 'additional_figure_manifest.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'completed': True, 'outputs': len(report['outputs'])}))
