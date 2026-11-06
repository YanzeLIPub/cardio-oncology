"""Preserve plotted geometry; consolidate main figure and export rounded matrix annotations."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import sys, re, json, csv, io
from lxml import etree
import pymupdf as fitz
R = Path(__file__).resolve().parents[1]
F = R / 'figures'
V = R / 'review'
out = F / 'consolidated'
out.mkdir(exist_ok=True)
features = ['Prealbumin', 'Hemoglobin', 'Magnesium', 'Globulin', 'WBC', 'Hematocrit', 'ALP', 'Plateletcrit', 'NSE', 'T-Chol', 'HDL-C', 'Albumin', 'LDH', 'Creatinine', 'CRP']
full = {'T-Chol': 'Total cholesterol', 'HDL-C': 'HDL cholesterol', 'WBC': 'White blood cell count (historical unverified SCH label)', 'ALP': 'Alkaline phosphatase', 'NSE': 'Neuron-specific enolase', 'LDH': 'Lactate dehydrogenase', 'CRP': 'C-reactive protein'}
csvs = {}
counts = {}
for n, centre in [(20, 'SCH'), (21, 'WHU'), (22, 'WHU_minus_SCH')]:
    doc = fitz.open(F / f'submission_numbering/Supplementary_Figure_S{n}.pdf')
    page = doc[0]
    words = page.get_text('words')
    labels = features if n == 20 else features[:-1]
    xs = {}
    ys = {}
    for label in labels:
        hits = [w for w in words if w[4] == label]
        horiz = min(hits, key=lambda w: w[0])
        vert = max(hits, key=lambda w: w[1])
        xs[label] = (vert[0] + vert[2]) / 2
        ys[label] = (horiz[1] + horiz[3]) / 2
    values = {}
    for w in words:
        if not re.fullmatch('[-−]?\\d\\.\\d', w[4]):
            continue
        x = (w[0] + w[2]) / 2
        y = (w[1] + w[3]) / 2
        row = min(ys, key=lambda k: abs(ys[k] - y))
        col = min(xs, key=lambda k: abs(xs[k] - x))
        assert abs(ys[row] - y) < 8 and abs(xs[col] - x) < 8, (n, w, row, col)
        assert (row, col) not in values
        values[row, col] = w[4].replace('−', '-')
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer)
    writer.writerow(['matrix', 'row_feature', 'column_feature', 'historical_display_value', 'value_status'])
    for row in labels:
        for col in labels:
            writer.writerow([centre, row, col, values.get((row, col), ''), 'rounded_graphical_annotation' if (row, col) in values else 'not_displayed_in_source'])
    csvs[f'figure3/correlation_{centre}.csv'] = buffer.getvalue().encode('utf-8')
    counts[centre] = {'cells': len(labels) ** 2, 'displayed_values': len(values), 'blank_values': len(labels) ** 2 - len(values)}
    if n == 20:
        assert values['NSE', 'Albumin'] == '-0.5'
    if n == 22:
        assert values['NSE', 'Albumin'] == '0.6'
    doc.close()
assert sum((x['displayed_values'] for x in counts.values())) == 597, counts
b = io.StringIO(newline='')
w = csv.writer(b)
w.writerow(['main_label', 'source_label', 'full_name'])
for short, name in zip(['PA', 'Hb', 'Mg', 'Glo', 'WBC', 'Hct', 'ALP', 'Pct', 'NSE', 'Chol', 'HDL', 'Alb', 'LDH', 'Cr', 'CRP'], features):
    w.writerow([short, name, full.get(name, name)])
csvs['figure3/feature_names.csv'] = b.getvalue().encode('utf-8')
csvs['figure3/README.txt'] = 'Figure 3 correlation annotations\nThese aggregate files preserve all 597 numeric annotations shown in the historical enlarged matrices. Values are rounded as displayed, not recovered full-precision correlations or newly estimated values. Twenty cells had no displayed number and remain blank. Feature names follow the historical plots; they do not certify assay equivalence. SCH Albumin here is the asterisk-prefixed field, with 10 complete NSE-Albumin pairs (rho -0.503); the separate joint-pattern analysis uses the unprefixed field and 8,841 pairs.\n'.encode('utf-8')
src = fitz.open(F / 'readability_revision/Figure3_readability.pdf')
for rect in src[0].search_for('UMAP Joint Embedding of Reversal Features (SCH + WHU)'):
    src[0].add_redact_annot(rect, fill=(1, 1, 1))
src[0].apply_redactions(images=0, graphics=0)
dst = fitz.open()
p = dst.new_page(width=468, height=558)
clips = [fitz.Rect(0, 0, 805.928, 320), fitz.Rect(0, 323, 805.928, 647), fitz.Rect(0, 648, 805.928, 983)]
rects = [fitz.Rect(0, 0, 468, 178), fitz.Rect(0, 180, 468, 367), fitz.Rect(0, 369, 468, 558)]
for clip, rect in zip(clips, rects):
    p.show_pdf_page(rect, src, 0, clip=clip, keep_proportion=True)
p.insert_font(fontname='Arial', fontfile=_release_font_path('arial.ttf'))
p.insert_text((151, 374), 'Joint embedding of the nine selected features', fontname='Arial', fontsize=8)
target = out / 'Figure3_consolidated.pdf'
dst.save(target, deflate=True)
p.get_pixmap(dpi=600, alpha=False).save(out / 'Figure3_consolidated.png')
p.get_pixmap(dpi=144, alpha=False).save(V / 'Figure3_consolidated_preview.png')
src.close()
dst.close()
for name, content in csvs.items():
    target = R / 'aggregate_csv_exports' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
print(json.dumps({'matrix_annotation_counts': counts, 'figure_exported': True}))
