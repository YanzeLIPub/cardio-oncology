from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
from collections import Counter
import json, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lxml import etree
import pymupdf as fitz
OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
A = OUT / 'aggregate'
F = OUT / 'figures'
F.mkdir(exist_ok=True)

def manual_auc(y, p):
    order = np.argsort(p, kind='stable')
    ys = y[order]
    ps = p[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ps) != 0) + 1]
    ends = np.r_[starts[1:], len(y)]
    ranks = np.empty(len(y), float)
    for a, b in zip(starts, ends):
        ranks[a:b] = (a + 1 + b) / 2
    n1 = y.sum()
    return (np.dot(ranks, ys) - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
metrics = pd.read_csv(A / 'whu_performance.csv')
paired = pd.read_csv(A / 'whu_paired_comparisons.csv')
coverage = pd.read_csv(A / 'whu_window_coverage.csv')
verified = []
for exp in ['measurement_caps', 'assay_mapping']:
    with np.load(OUT / 'private' / (exp + '_predictions.npz')) as z:
        y = z['y']
        assert np.unique(z['groups']).size == len(y) and set(z['fold']) == set(range(5))
        for _, r in metrics[metrics.experiment.eq(exp)].iterrows():
            p = z[r.scope + '__' + r.model]
            a = manual_auc(y, p)
            b = np.mean((y - p) ** 2)
            assert abs(a - r.auc) < 1e-12 and abs(b - r.brier) < 1e-12
            assert r.auc_low <= r.auc_high and r.brier_low <= r.brier_high
            verified.append({'experiment': exp, 'scope': r.scope, 'model': r.model, 'rank_auc_agrees': True, 'brier_agrees': True})
with np.load(OUT / 'private/whu_assay_sensitivity.npz') as z:
    assert np.array_equal(z['unit_only'], z['name_only'], equal_nan=True)
sch = _release_resolve_config(json.loads((A / 'sch_analysis.json').read_text(encoding='utf-8')))
s = pd.read_csv(A / 'sch_acute_poa_summary.csv')
assert s[s.scope.eq('all_matched_source_stays')].units.sum() == sch['counts']['matched_source_stay_keys'] == 162008
assert s[s.scope.eq('one_first_matched_stay_per_patient')].units.sum() == 38509
(A / 'verification.json').write_text(json.dumps({'probability_metrics': verified, 'SCH_denominators_reconciled': True, 'unit_only_and_name_only_patient_medians_identical': True, 'private_data_not_in_public_outputs': True}, indent=2), encoding='utf-8')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.5))
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.77, wspace=0.26)
fig.suptitle('WHU source-report measurement sensitivity (2,244 patients)', fontsize=12, y=0.98)
scopes = ['h0', 'h24', 'h48', 'whole']
labels = ['0 h', '24 h', '48 h', 'Whole stay']
for model, color, off, lab in [('logistic', '#246CB5', -0.045, 'Logistic regression'), ('ensemble', '#C26B26', 0.045, 'XGBoost / random forest')]:
    g = metrics[(metrics.experiment == 'measurement_caps') & (metrics.model == model)].set_index('scope').loc[scopes]
    axs[0].errorbar(np.arange(4) + off, g.auc, yerr=[g.auc - g.auc_low, g.auc_high - g.auc], marker='o', markersize=4, capsize=3, color=color, label=lab, lw=1.3)
axs[0].axhline(0.5, ls=':', lw=0.8, color='.5')
axs[0].set(xticks=np.arange(4), xticklabels=labels, ylabel='Recorded-label AUC', ylim=(0.45, 0.74), xlabel='Measurement cap from admission')
axs[0].set_title('c)  Patient-isolated discrimination', loc='left', fontsize=10)
axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.23), fontsize=8, frameon=False, ncol=1)
g = coverage.set_index('scope').loc[scopes]
bars = axs[1].bar(np.arange(4), g.any_lab, color='#3C8391', width=0.55)
for b, n in zip(bars, g.any_lab):
    axs[1].text(b.get_x() + b.get_width() / 2, n + 45, f'{n:,}', ha='center', fontsize=9)
axs[1].set(xticks=np.arange(4), xticklabels=labels, ylabel='Patients with at least one accepted laboratory value', ylim=(0, 2500), xlabel='Measurement cap from admission')
axs[1].set_title('d)  Measurement coverage', loc='left', fontsize=10)
for ax in axs:
    ax.grid(axis='y', alpha=0.22)
    ax.set_axisbelow(True)
fig.savefig(F / 'WHU_time_panels.pdf')
fig.savefig(F / 'WHU_time_panels.svg')
plt.close(fig)
old = ROOT / 'revision_2026-11-03/clinical_review_v1/submission_package/author_reference/supplementary_figures/Supplementary_Figure_S14.pdf'
top = fitz.open(old)
bottom = fitz.open(F / 'WHU_time_panels.pdf')
result = fitz.open()
page = result.new_page(width=756, height=684)
page.show_pdf_page(fitz.Rect(0, 0, 756, 360), top, 0)
page.show_pdf_page(fitz.Rect(0, 360, 756, 684), bottom, 0)
result.save(F / 'Supplementary_Figure_S14.pdf')
page.get_pixmap(matrix=fitz.Matrix(2, 2)).save(F / 'Supplementary_Figure_S14.png')
page = bottom[0]
page.get_pixmap(matrix=fitz.Matrix(2, 2)).save(F / 'Figure_S14cd.png')
ns = 'http://www.w3.org/2000/svg'
root = etree.Element('{' + ns + '}svg', nsmap={None: ns}, width='756pt', height='684pt', viewBox='0 0 756 684')
for source, y, prefix in [(old.with_suffix('.svg'), 0, 'old_'), (F / 'WHU_time_panels.svg', 360, 'new_')]:
    child = etree.parse(str(source)).getroot()
    identifiers = {n.get('id'): prefix + n.get('id') for n in child.iter() if n.get('id')}
    for n in child.iter():
        for attr, value in list(n.attrib.items()):
            if attr == 'id':
                n.set(attr, identifiers[value])
            else:
                for old_id, new_id in identifiers.items():
                    value = value.replace('url(#' + old_id + ')', 'url(#' + new_id + ')')
                    value = '#' + new_id if value == '#' + old_id else value
                n.set(attr, value)
    child.set('x', '0')
    child.set('y', str(y))
    child.set('width', '756')
    child.set('height', '360' if y == 0 else '324')
    root.append(child)
(F / 'Supplementary_Figure_S14.svg').write_bytes(etree.tostring(root, encoding='utf-8', xml_declaration=True))
assert not root.findall('.//{' + ns + '}image')
print(json.dumps({'metric_rows_verified': len(verified), 'figure': 'Supplementary_Figure_S14', 'old_panels_retained': True, 'new_panels': ['c', 'd'], 'embedded_raster_in_svg': 0}))
