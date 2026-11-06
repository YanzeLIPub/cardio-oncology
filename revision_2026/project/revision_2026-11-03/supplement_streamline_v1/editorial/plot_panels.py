"""Visualize retained results from the preceding published table values.

No model fitting or new estimates. Error bars retain the source intervals.
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymupdf as fitz
R = Path(__file__).resolve().parents[1]
P = R.parent / 'source_based_analyses_v1'
F = R / 'figures'
F.mkdir(exist_ok=True)
SOURCE = _release_resolve_config(json.loads((R / 'review/source_content.json').read_text(encoding='utf-8')))
T = {x['old_number']: x['rows'][1:] for x in SOURCE['tables']}
BLUE = '#2875AB'
RED = '#C35D49'
TEAL = '#26847E'
PURPLE = '#7755A7'
GRAY = '#888888'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': 0.6})
PANELS = {}
USED = {}

def ci(s):
    s = re.sub('(?<=\\d)-(?=\\d)', ' to ', s.replace('−', '-'))
    vals = re.findall('[+-]?\\d*\\.?\\d+', s)
    if len(vals) < 3:
        raise ValueError(s)
    a, l, u = map(float, vals[:3])
    assert l <= a + 1e-08 and a <= u + 1e-08, (a, l, u, s)
    return (a, l, u)

def forest(ax, labels, values, color=BLUE, reference=None, offset=0):
    y = np.arange(len(labels)) + offset
    for yy, s in zip(y, values):
        a, l, u = ci(s)
        ax.errorbar(a, yy, xerr=[[a - l], [u - a]], fmt='o', color=color, ms=3, capsize=2, lw=0.8)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.4, -0.6)
    if reference is not None:
        ax.axvline(reference, c='.6', ls=':', lw=0.7)
    ax.grid(axis='x', alpha=0.18)
    ax.set_axisbelow(True)

def title(ax, letter, text):
    ax.set_title(letter + ')  ' + text, loc='left', pad=7, fontweight='bold')

def finish_part(fig, n, height):
    p = F / f'S{n}_new_panels.pdf'
    fig.savefig(p, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return p

def publish(n, part, append=True, letter_a=False):
    base = P / f'submission_package/author_reference/supplementary_figures/Supplementary_Figure_S{n}.pdf'
    out = fitz.open()
    bottom = fitz.open(part)
    w = 504
    if append:
        old = fitz.open(base)
        h = w * old[0].rect.height / old[0].rect.width
    else:
        h = 0
    bh = w * bottom[0].rect.height / bottom[0].rect.width
    page = out.new_page(width=w, height=h + bh)
    if append:
        page.show_pdf_page(fitz.Rect(0, 0, w, h), old, 0)
        if letter_a:
            page.insert_text((8, 34), 'a)', fontsize=9, fontname='hebo')
    page.show_pdf_page(fitz.Rect(0, h, w, h + bh), bottom, 0)
    path = F / f'Supplementary_Figure_S{n}'
    out.save(path.with_suffix('.pdf'), deflate=True)
    page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5)).save(path.with_suffix('.png'))
    path.with_suffix('.svg').write_text(page.get_svg_image(text_as_path=True), encoding='utf-8')
    assert not page.get_images(full=True), n
    PANELS[str(n)] = {'top_fraction': h / (h + bh), 'width_points': w, 'height_points': h + bh, 'append': append, 'source_tables': USED[n]}
    bottom.close()
    out.close()
    if append:
        old.close()
fig, axs = plt.subplots(2, 2, figsize=(7, 5.2))
fig.subplots_adjust(left=0.17, right=0.98, bottom=0.09, top=0.94, wspace=0.62, hspace=0.65)
ax = axs[0, 0]
arr = np.array([[int(v) for v in r[1:]] for r in T[6]], float)
arr = arr / arr.sum(1)[:, None] * 100
left = np.zeros(7)
for k, (lab, col) in enumerate(zip(['Present', 'Absent', 'Uncertain', 'Unknown', 'Empty'], [BLUE, TEAL, '#E3A84C', GRAY, '#DDDDDD'])):
    ax.barh(np.arange(7), arr[:, k], left=left, color=col, label=lab, height=0.7)
    left += arr[:, k]
ax.set(yticks=np.arange(7), yticklabels=[r[0] for r in T[6]], xlim=(0, 100), xlabel='Diagnosis-slot occurrences (%)')
ax.invert_yaxis()
title(ax, 'c', 'SCH admission status')
ax.legend(ncol=3, fontsize=5.7, loc='upper center', bbox_to_anchor=(0.4, -0.26), frameon=False, columnspacing=0.8)
ax = axs[0, 1]
labs = ['Cancer / pathology', 'Stage', 'Treatment plan', 'Medication / history', 'Acuity / vital signs']
v = [int(r[2]) for r in T[7]]
ax.barh(np.arange(5), v, color=TEAL)
ax.set(yticks=np.arange(5), yticklabels=labs, xlim=(0, 2600), xlabel='Patients with nonempty fields')
ax.invert_yaxis()
for i, a in enumerate(v):
    ax.text(a + 30, i, str(a), va='center', fontsize=6.5)
title(ax, 'd', 'WHU source-field coverage')
ax = axs[1, 0]
total = np.array([38509, 2562, 9946])
linked = np.array([38509, 2258, 9946])
ax.barh(range(3), 100 * linked / total, color=BLUE, label='Linked')
ax.barh(range(3), 100 * (total - linked) / total, left=100 * linked / total, color='#DDD', label='Unlinked')
ax.set(yticks=range(3), yticklabels=['SCH patients', 'WHU patients', 'MIMIC admissions'], xlim=(0, 100), xlabel='Original analysis units (%)')
ax.invert_yaxis()
title(ax, 'e', 'Source availability')
for i in range(3):
    ax.text(3, i, f'{linked[i]:,} / {total[i]:,}', va='center', color='white', fontsize=7)
ax.legend(fontsize=6, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
ax = axs[1, 1]
a = np.array([[int(r[1]), int(r[2].split('/')[0]), int(r[3].split('/')[0])] for r in T[13]])
y = np.arange(3)
ax.barh(y, a[:, 2] / a[:, 0] * 100, color=GRAY, label='No storage time')
ax.barh(y, a[:, 1] / a[:, 0] * 100, left=a[:, 2] / a[:, 0] * 100, color=RED, label='Stored after cutoff')
ax.set(yticks=y, yticklabels=['0 h', '24 h', '48 h'], xlim=(0, 100), xlabel='Chart-selected note records (%)')
ax.invert_yaxis()
title(ax, 'f', 'MIMIC note-time limitations')
ax.legend(fontsize=5.9, frameon=False, loc='upper center', bbox_to_anchor=(0.3, -0.3), ncol=1)
USED[1] = [6, 7, 11, 13]
publish(1, finish_part(fig, 1, 5.2))
fig = plt.figure(figsize=(7, 5.1))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], left=0.21, right=0.98, bottom=0.08, top=0.94, wspace=0.8, hspace=0.65)
ax = fig.add_subplot(gs[0, 0])
labs = ['Ever; all stays', 'Ever; survivors', 'Prior; all stays', 'Prior; survivors']
forest(ax, labs, [r[1] for r in T[9]])
title(ax, 'd', 'MIMIC eligibility sensitivity')
ax.set_xlabel('AUROC (95% CI)')
for i, r in enumerate(T[8]):
    ax.text(0.99, i, f'n={r[1]}', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=5.9)
ax.set_xlim(0.69, 0.79)
ax = fig.add_subplot(gs[0, 1])
labs = ['All−survivor / ever', 'All−survivor / prior', 'Prior−ever / all', 'Prior−ever / survivors']
forest(ax, labs, [r[2] for r in T[10]], RED, 0)
title(ax, 'e', 'Same-target contrasts')
ax.set_xlabel('Paired ΔAUROC (95% CI)')
ax = fig.add_subplot(gs[1, :])
labs = ['Historical fixed scores', 'Patient-isolated fixed scores', 'Full-training refit', 'Base refit', 'Base refit + Platt']
forest(ax, labs, [r[1] for r in T[12]], TEAL)
title(ax, 'f', 'WHU source-linked cohort (n=2,258)')
ax.set_xlabel('AUROC (95% CI)')
ax.set_xlim(0.7, 0.87)
USED[4] = [8, 9, 10, 12]
publish(4, finish_part(fig, 4, 5.1))
fig, ax = plt.subplots(figsize=(7, 1.95))
fig.subplots_adjust(left=0.16, right=0.95, bottom=0.24, top=0.8)
models = ['fixed', 'shallow', 'regularized', 'lower_rate']
matrix = np.array([[models.index(r[2]) for r in T[19] if r[0] == c] for c in ['SCH', 'WHU', 'MIMIC']])
from matplotlib.colors import ListedColormap
ax.pcolormesh(np.arange(6) - 0.5, np.arange(4) - 0.5, matrix, cmap=ListedColormap(['#EEE', '#B8D6E9', '#D9C7E5', '#F2D1AE']), vmin=0, vmax=3, edgecolors='white', linewidth=0.8)
ax.set_ylim(2.5, -0.5)
for i, c in enumerate(['SCH', 'WHU', 'MIMIC']):
    for j, r in enumerate([r for r in T[19] if r[0] == c]):
        ax.text(j, i, r[2].replace('_', ' ') + '\n' + r[4] + ' rounds', ha='center', va='center', fontsize=7)
ax.set(xticks=range(5), xticklabels=[f'Fold {i}' for i in range(1, 6)], yticks=range(3), yticklabels=['SCH', 'WHU', 'MIMIC'])
title(ax, 'b', 'Candidate and final fitting rounds selected using training data')
USED[7] = [19]
publish(7, finish_part(fig, 7, 1.95), letter_a=True)
fig, axs = plt.subplots(2, 2, figsize=(7, 5.5))
fig.subplots_adjust(left=0.18, right=0.97, bottom=0.1, top=0.94, wspace=0.72, hspace=0.55)
ax = axs[0, 0]
forest(ax, [r[0].replace(' to ', ' → ') for r in T[24]], [r[4] for r in T[24]], TEAL, 0.5)
title(ax, 'a', 'New historical-code replay')
ax.set_xlabel('AUROC (95% CI)')
labels = [r[0].replace(' to ', ' → ') for r in T[25]]
for col, name, letter, ax in [(2, 'AUROC', 'b', axs[0, 1]), (2, 'Average precision', 'c', axs[1, 0]), (3, 'Brier score', 'd', axs[1, 1])]:
    for k, (model, color) in enumerate([('Linear', BLUE), ('Spline', RED)]):
        vals = [r[2 + k] for r in T[25]] if letter == 'b' else [next((r[col] for r in T[26] if r[0] == direction[0] and r[1] == model)) for direction in T[25]]
        forest(ax, labels, vals, color, 0.5 if letter == 'b' else None, offset=(k - 0.5) * 0.22)
    title(ax, letter, 'Shared age: ' + name)
    ax.set_xlabel(name + ' (95% CI)')
handles = [plt.Line2D([0], [0], marker='o', color=c, lw=0.8, label=l) for l, c in [('Linear age', BLUE), ('Spline age', RED)]]
fig.legend(handles=handles, ncol=2, frameon=False, loc='lower center', bbox_to_anchor=(0.58, -0.005), fontsize=7)
USED[9] = [24, 25, 26]
publish(9, finish_part(fig, 9, 5.5), append=False)
fig, ax = plt.subplots(figsize=(7, 3.35))
fig.subplots_adjust(left=0.34, right=0.97, bottom=0.15, top=0.87)
rows = [r for r in T[28] if r[1] != 'Median ensemble']
labs = []
short = {'Median + indicators': 'Values + indicators', 'Missingness only': 'Missingness only', 'Logistic': 'Logistic'}
for r in rows:
    labs.append(r[0].replace(' to ', ' → ') + ' / ' + short[r[1]])
for k, (table, color, lab) in enumerate([(28, BLUE, 'Full target'), (31, GRAY, 'Covered target')]):
    vals = [next((q[2] for q in T[table] if q[0] == r[0] and q[1] == r[1])) for r in rows]
    forest(ax, labs, vals, color, 0.5, offset=(k - 0.5) * 0.23)
ax.set_xlabel('AUROC (95% CI)')
title(ax, 'd', 'Other preprocessing models in the four-laboratory task')
ax.legend(handles=[plt.Line2D([0], [0], marker='o', color=c, label=l, lw=0.8) for c, l in [(BLUE, 'Full target'), (GRAY, 'Covered target')]], loc='lower right', fontsize=7, frameon=False)
USED[11] = [28, 31]
publish(11, finish_part(fig, 11, 3.35))
fig, axs = plt.subplots(1, 3, figsize=(7, 3.15))
fig.subplots_adjust(left=0.17, right=0.98, bottom=0.2, top=0.87, wspace=0.7)
labs = [r[0] + ' / ' + r[1].replace('Primary ', 'Full ').replace('Base ', 'base ') for r in T[18]]
for ax, col, letter, name, ref in [(axs[0], 2, 'd', 'Joint intercept', 0), (axs[1], 3, 'e', 'Joint slope', 1)]:
    forest(ax, labs if col == 2 else [''] * 9, [r[col] for r in T[18]], TEAL, ref)
    title(ax, letter, name)
    ax.set_xlabel('Estimate (95% CI)')
ax = axs[2]
runs = ['SCH', 'WHU', 'MIMIC', 'MIMIC no AST/HDL']
comp = [('positive label component', BLUE, 'Positive'), ('negative label component', RED, 'Negative'), ('total', PURPLE, 'Total')]
for k, (component, c, l) in enumerate(comp):
    forest(ax, ['SCH', 'WHU', 'MIMIC', 'MIMIC\nno AST/HDL'], [next((r[2] for r in T[74] if r[0] == run and r[1] == component)) for run in runs], c, 0, offset=(k - 1) * 0.18)
title(ax, 'f', 'Continuous NRI')
ax.set_xlabel('NRI vs age-only (95% CI)')
ax.legend(handles=[plt.Line2D([0], [0], marker='o', color=c, label=l, lw=0.8) for _, c, l in comp], fontsize=6, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=3, columnspacing=0.8)
USED[13] = [18, 74]
publish(13, finish_part(fig, 13, 3.15))
fig, axs = plt.subplots(1, 2, figsize=(7, 2.9), sharey=True)
fig.subplots_adjust(left=0.34, right=0.97, bottom=0.18, top=0.85, wspace=0.35)
labs = ['Original union', 'MI', 'Stroke', 'Heart failure', 'Cardiac arrest', 'Cardiomyopathy', 'Unstable angina', 'Union without CM-specific']
labs = [a + ' (n=' + re.search('[\\d,]+', r[1]).group() + ')' for a, r in zip(labs, T[48])]
for ax, col, letter, name in [(axs[0], 2, 'e', 'AUROC'), (axs[1], 3, 'f', 'Average precision')]:
    forest(ax, labs, [r[col] for r in T[48]], BLUE, 0.5 if col == 2 else None)
    title(ax, letter, 'Recorded code groups')
    ax.set_xlabel(name + ' (95% CI)')
USED[15] = [48]
publish(15, finish_part(fig, 15, 2.9))
fig, axs = plt.subplots(1, 2, figsize=(7, 2.65))
fig.subplots_adjust(left=0.17, right=0.97, bottom=0.22, top=0.84, wspace=0.85)
ax = axs[0]
labs = [r[0] + ' / ' + r[1] + ' records' for r in T[47]]
for k, (idx, c, lab) in enumerate([(3, BLUE, 'Mean'), (4, RED, 'Earliest'), (5, GRAY, 'Count only')]):
    forest(ax, labs, [r[idx] for r in T[47]], c, 0.5, offset=(k - 1) * 0.15)
title(ax, 'd', 'Record-count strata')
ax.set_xlabel('AUROC (95% CI)')
ax.legend(handles=[plt.Line2D([0], [0], marker='o', color=c, label=l, lw=0.8) for _, c, l in [(3, BLUE, 'Mean'), (4, RED, 'Earliest'), (5, GRAY, 'Count only')]], fontsize=6, frameon=False, loc='upper center', bbox_to_anchor=(0.6, -0.22), ncol=3, columnspacing=0.8)
ax = axs[1]
rows = [r for r in T[46] if r[1] == 'AUC']
forest(ax, ['Mean−earliest', 'Mean−first observed'], [r[2] for r in rows], TEAL, 0)
title(ax, 'e', 'WHU−SCH gain')
ax.set_xlabel('ΔAUROC gain (95% CI)')
USED[17] = [46, 47]
publish(17, finish_part(fig, 17, 2.65))
fig, axs = plt.subplots(1, 2, figsize=(7, 2.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.27, top=0.83, wspace=0.36)
ax = axs[0]
for cohort, color, off in [('SCH', BLUE, -0.05), ('WHU', RED, 0.05)]:
    rows = [r for r in T[55] if r[0] == cohort]
    v = np.array([ci(r[2]) for r in rows])
    ax.errorbar(np.arange(4) + off, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], marker='o', color=color, label=cohort, ms=3, capsize=2, lw=0.9)
ax.set(xticks=range(4), xticklabels=['42', '43', '44', '45'], xlabel='Training seed', ylabel='XGBoost AUROC (95% CI)', ylim=(0.6, 0.9))
title(ax, 'd', 'Model discrimination')
ax.legend(frameon=False, fontsize=7)
ax = axs[1]
for feature, linestyle in [('NSE', '-'), ('Albumin', '--')]:
    row = next((r for r in T[58] if r[0] == feature))
    v = np.array([[float(v.strip()) for v in s.split('/')] for s in row[1:5]])
    for k, (cohort, c) in enumerate([('SCH', BLUE), ('WHU', RED)]):
        ax.plot(range(4), v[:, k], color=c, ls=linestyle, marker='o', ms=3, label=cohort + ' ' + feature, lw=0.9)
ax.axhline(0, c='.6', ls=':', lw=0.7)
ax.set(xticks=range(4), xticklabels=['42', '43', '44', '45'], xlabel='Training seed', ylabel='Observed value / SHAP Spearman ρ', ylim=(-0.9, 0.3))
title(ax, 'e', 'NSE and albumin directions')
ax.legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
USED[19] = [55, 58]
publish(19, finish_part(fig, 19, 2.5))
fig = plt.figure(figsize=(7, 5.1))
gs = fig.add_gridspec(2, 3, height_ratios=[0.65, 2.4], left=0.22, right=0.98, bottom=0.1, top=0.94, hspace=0.55, wspace=0.35)
ax = fig.add_subplot(gs[0, :])
groups = ['Defined primary', 'Multiple groups', 'Secondary / unspecified', 'No interpretable code', 'No source']
cols = [BLUE, TEAL, PURPLE, '#BBB', '#E2E2E2']
coverage = []
for cohort in ['SCH', 'WHU']:
    rr = [r for r in T[60] if r[0] == cohort]
    coverage.append([sum((int(r[2]) for r in rr[1:6])), int(rr[6][2]), int(rr[7][2]), int(rr[8][2]), int(rr[9][2])])
left = np.zeros(2)
for k, (lab, c) in enumerate(zip(groups, cols)):
    values = np.array(coverage)[:, k] / np.array([38509, 2562]) * 100
    ax.barh([0, 1], values, left=left, color=c, label=lab, height=0.6)
    left += values
ax.set(yticks=[0, 1], yticklabels=['SCH', 'WHU'], xlim=(0, 100), xlabel='Original patients (%)')
ax.invert_yaxis()
title(ax, 'c', 'Cancer-code coverage')
ax.legend(frameon=False, fontsize=5.5, ncol=3, bbox_to_anchor=(0.5, -0.48), loc='upper center', columnspacing=0.8)
rows = [r for r in T[61] if any((not s.startswith('Not estimated') for s in r[2:]))]
short = {'All original patients': 'All', 'Lung/bronchus': 'Lung / bronchus', 'Other defined primary site': 'Other primary', 'Multiple broad source-code groups or C97': 'Multiple / C97', 'Secondary/unspecified codes only': 'Secondary / unspec.', 'No interpretable malignant code': 'Uninterpretable code', 'No eligible diagnostic source': 'No source'}
labs = [r[0] + ' / ' + short.get(r[1], r[1]) for r in rows]
for k, (letter, name) in enumerate([('d', 'NSE–albumin'), ('e', 'NSE–WBC'), ('f', 'WBC–albumin')]):
    ax = fig.add_subplot(gs[1, k])
    ax.set(yticks=range(len(rows)), yticklabels=labs if k == 0 else [''] * len(rows), ylim=(len(rows) - 0.4, -0.6), xlim=(-0.55, 0.55), xlabel='Spearman ρ (95% CI)')
    ax.axvline(0, c='.6', ls=':', lw=0.7)
    for j, r in enumerate(rows):
        s = r[2 + k]
        if s.startswith('Not estimated'):
            ax.text(0.98, j, 'NE', transform=ax.get_yaxis_transform(), ha='right', va='center', color='.5', fontsize=5.8)
            continue
        a, l, u = ci(s)
        ax.errorbar(a, j, xerr=[[a - l], [u - a]], fmt='o', color=BLUE if r[0] == 'SCH' else RED, ms=3, capsize=2, lw=0.8)
    title(ax, letter, name)
    ax.grid(axis='x', alpha=0.2)
USED[23] = [60, 61]
publish(23, finish_part(fig, 23, 5.1))
fig, axs = plt.subplots(1, 2, figsize=(7, 2.6))
fig.subplots_adjust(left=0.19, right=0.98, bottom=0.25, top=0.83, wspace=0.6)
ax = axs[0]
tot = np.array([int(r[1]) for r in T[70]])
complete = np.array([int(r[2]) for r in T[70]])
matched = np.array([int(r[3]) for r in T[70]])
ax.barh([0, 1], matched / tot * 100, color=BLUE, label='Exact match')
ax.barh([0, 1], (complete - matched) / tot * 100, left=matched / tot * 100, color=GRAY, label='Complete unmatched')
ax.barh([0, 1], (tot - complete) / tot * 100, left=complete / tot * 100, color='#DDD', label='Incomplete')
ax.set(yticks=[0, 1], yticklabels=['SCH', 'WHU'], xlim=(0, 100), xlabel='Source records (%)')
ax.invert_yaxis()
title(ax, 'e', 'Regional-vector lookup')
for i in range(2):
    ax.text(2, i, f'{matched[i]:,} / {tot[i]:,}', va='center', color='white', fontsize=7)
ax.legend(frameon=False, fontsize=5.8, loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=1)
ax = axs[1]
rows = [r for r in T[71] if r[2] == 'Original']
labs = [r[0].replace('Registered city', 'Registered').replace('Current city', 'Current') + ' / ' + r[1].replace('Examination', 'check').replace('Admission', 'admit') for r in rows]
v = np.array([int(r[4]) / int(r[3]) * 100 for r in rows])
ax.barh(range(4), v, color=TEAL)
ax.set(yticks=range(4), yticklabels=labs, xlim=(0, 105), xlabel='All five regional values equal (%)')
ax.invert_yaxis()
title(ax, 'f', 'WHU residence comparison')
for i, r in enumerate(rows):
    ax.text(2, i, r[4] + ' / ' + r[3], va='center', color='white', fontsize=7)
USED[27] = [70, 71]
publish(27, finish_part(fig, 27, 2.6))
(R / 'review/figure_changes.json').write_text(json.dumps(PANELS, indent=2), encoding='utf-8')
print(json.dumps({'updated_supplementary_figures': sorted(map(int, PANELS)), 'vector_output': True, 'new_model_estimates': False}))
