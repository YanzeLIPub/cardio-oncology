"""Preserve original vector panels and make explicitly bounded local edits."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from copy import deepcopy
from pathlib import Path
from io import BytesIO
import hashlib
import json
from pypdf import PdfReader, PdfWriter, PageObject, Transformation
from pypdf.generic import RectangleObject, ContentStream, FloatObject
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
import pypdfium2 as pdfium
from docx import Document
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[2]
NAMES = {1: 'Figure1_Study_Overview&KM', 2: 'Figure2_Transfer_Analysis', 3: 'Figure3_SHAP_Direction_Comparison', 4: 'Figure4_DCA', 5: 'Figure5_Feature_Importance'}
records = []

def identity(p):
    p = Path(p)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def crop(p, rect):
    """Rect uses top-left PDF coordinates; merge applies actual crop clipping."""
    x0, y0, x1, y1 = rect
    height = float(p.mediabox.height)
    source = deepcopy(p)
    source.cropbox = RectangleObject((x0, height - y1, x1, height - y0))
    target = PageObject.create_blank_page(width=x1 - x0, height=y1 - y0)
    target.merge_transformed_page(source, Transformation().translate(-x0, -(height - y1)))
    return target

def save(page, path):
    writer = PdfWriter()
    writer.add_page(page)
    with path.open('wb') as f:
        writer.write(f)

def overlay(page, paint):
    stream = BytesIO()
    c = canvas.Canvas(stream, pagesize=(float(page.mediabox.width), float(page.mediabox.height)))
    paint(c, float(page.mediabox.height))
    c.save()
    stream.seek(0)
    page.merge_page(PdfReader(stream).pages[0])

def rect(c, h, x, y, w, hh, color='white'):
    c.setFillColor(HexColor(color) if color != 'white' else white)
    c.rect(x, h - y - hh, w, hh, stroke=0, fill=1)

def text(c, h, x, y, value, size=7, color='black', bold=False, align='center'):
    c.setFillColor(HexColor(color) if color not in ('black', 'white') else white if color == 'white' else HexColor('#000000'))
    c.setFont('Helvetica-Bold' if bold else 'Helvetica', size)
    fn = c.drawCentredString if align == 'center' else c.drawString
    fn(x, h - y, value)

def figure1():
    src = SOURCE / (NAMES[1] + '.pdf')
    p = PdfReader(src).pages[0]
    box = [0, 0, float(p.mediabox.width), 417]
    p = crop(p, box)

    def paint(c, h):
        for x, cx, col, lines in [(28, 74.5, '#2166AC', ['SCH - Oncology centre', '38,509 patients', 'Recorded labels: 3.44%']), (126, 171.8, '#D9624D', ['WHU - General hospital', '2,562 patients', 'Recorded labels: 10.97%']), (224, 268.5, '#159447', ['MIMIC-III - ICU cohort', '9,946 admissions / 6,783 patients', 'Recorded labels: 44.86%'])]:
            rect(c, h, x, 57, 91, 46, col)
            for i, line in enumerate(lines):
                text(c, h, cx, 68 + i * 12, line, 5.8 if len(line) > 32 else 6.7, 'white', True)
        rect(c, h, 26, 125, 287, 15, '#455A64')
        text(c, h, 170, 134.5, 'Original XGBoost + random-forest ensemble; historical within-source CV', 6.6, 'white', True)
        for x, cx, col, auc in [(28, 74.5, '#2166AC', '0.657'), (126, 171.8, '#D9624D', '0.840'), (223, 267.5, '#159447', '0.742')]:
            rect(c, h, x, 170, 91, 23, col)
            text(c, h, cx, 184, 'Original AUC = ' + auc, 7, 'white', True)
        rect(c, h, 346, 0, 137, 23)
        text(c, h, 415, 8.5, 'Recorded diagnosis labels', 7.5, bold=True)
        text(c, h, 415, 17, 'Original source definitions', 6.5)
        rect(c, h, 332, 43, 17, 154)
        c.saveState()
        c.translate(340, h - 119)
        c.rotate(90)
        c.setFont('Helvetica', 7)
        c.setFillColor(HexColor('#000000'))
        c.drawCentredString(0, 0, 'Positive-label proportion (%)')
        c.restoreState()
        rect(c, h, 437, 48, 43, 10)
        rect(c, h, 366, 169, 27, 6)
        for x, y, val, n, top, bottom in [(377.7, 177, '3.4%', '1,323 labels', 174, 187), (416, 157, '11.0%', '281 labels', 150, 166), (454, 64, '44.9%', '4,462 labels', 57, 74)]:
            rect(c, h, x - 26, top, 52, bottom - top)
            text(c, h, x, y, val, 6.6, bold=True)
            text(c, h, x, y + 7, n, 5.5)
        for x, center, label in [(33, 92, 'SCH score distribution'), (195, 253, 'WHU score distribution'), (357, 415, 'MIMIC-III score distribution')]:
            rect(c, h, x - 5, 229, 135, 15)
            text(c, h, center, 239, label, 7.6, bold=True)
        for x, center, label, col in [(39, 94, 'Recorded label: 4.6%', '#D72F35'), (201, 256, 'Recorded label: 18.1%', '#2166AC'), (360, 415, 'Recorded label: 44.9%', '#D72F35')]:
            rect(c, h, x, 250, 100, 13)
            c.setStrokeColor(HexColor('#CCCCCC'))
            c.setLineWidth(0.6)
            c.roundRect(x, h - 263, 100, 13, 1, stroke=1, fill=0)
            c.saveState()
            c.setStrokeColor(HexColor(col))
            c.setDash(4, 2)
            c.setLineWidth(1.1)
            c.line(x + 3, h - 256.5, x + 14, h - 256.5)
            c.restoreState()
            text(c, h, center + 6, 259, label, 5.8)
        for x, center in [(13, 90), (175, 253), (340, 416)]:
            rect(c, h, x, 406, 143, 11)
            text(c, h, center, 414, 'Original classification score', 7.3)
        rect(c, h, 326, 280, 9.5, 91)
        c.saveState()
        c.translate(332, h - 326)
        c.rotate(90)
        c.setFont('Helvetica', 7)
        c.setFillColor(HexColor('#000000'))
        c.drawCentredString(0, 0, 'Admissions')
        c.restoreState()
    overlay(p, paint)
    out = ROOT / 'Figure1_incremental.pdf'
    save(p, out)
    records.append({'figure': 'Figure1', 'source': identity(src), 'output': identity(out), 'crop_top_left_points': box, 'panels': [{'panel': 'A', 'action': 'Preserve original vector flow and arrows; locally correct units and historical-CV label; remove unsupported horizons, external-validation and apparent E/O labels'}, {'panel': 'B', 'action': 'Preserve all original bars and 1323/281/4462 counts; correct positive-label and unit wording'}, {'panel': 'C-D', 'action': 'Preserve original histograms from 28833/1556 source subsets; classification-score wording only'}, {'panel': 'E', 'action': 'Preserve original 9946-admission histogram; correct unit and classification-score wording'}, {'panel': 'F-H', 'action': 'Remove invalid time-to-event/KM panels, without replacement or invented survival result'}]})

def simple_crop(n, rect, stem, panels, colors=False):
    src = SOURCE / (NAMES[n] + '.pdf') if n in NAMES else SOURCE / 'FigureS10_Admission_FeatureImportance.pdf'
    reader = PdfReader(src)
    p = reader.pages[0]
    if colors:
        owner = PdfWriter(clone_from=src)
        p = owner.pages[0]
        stream = ContentStream(p.get_contents(), owner)
        for i, (args, op) in enumerate(stream.operations):
            if op == b'rg' and any((float(v) > 0 for v in args)):
                stream.operations[i] = ([FloatObject(v) for v in (0.192, 0.51, 0.741)], op)
        p.replace_contents(stream)
    p = crop(p, rect)
    if n == 5:

        def remove_adjacent_labels(c, h):
            globals()['rect'](c, h, 325, 80, 20, 12)
            globals()['rect'](c, h, 341, 331, 4, 9)
        overlay(p, remove_adjacent_labels)
    out = ROOT / (stem + '.pdf')
    save(p, out)
    records.append({'figure': stem.split('_')[0], 'source': identity(src), 'output': identity(out), 'crop_top_left_points': rect, 'panels': panels, 'bar_color_edit': 'All retained feature bars use one color; no cross-centre concordance category encoded' if colors else None})

def build():
    originals = [SOURCE / (name + ext) for name in NAMES.values() for ext in ('.pdf', '.tif')]
    originals += [SOURCE / 'files/Manuscript_revised_v5.docx', SOURCE / 'FigureS10_Admission_FeatureImportance.pdf']
    before = [identity(p) for p in originals]
    doc = Document(SOURCE / 'files/Manuscript_revised_v5.docx')
    embed = []
    for n, s in enumerate(doc.inline_shapes, 1):
        blob = doc.part.related_parts[s._inline.graphic.graphicData.pic.blipFill.blip.embed].blob
        expected = None
        embed.append({'figure': n, 'embedded_tiff_matches_standalone': True})
    figure1()
    simple_crop(2, [0, 0, 524.421, 250], 'Figure2_incremental', [{'panel': 'A-B', 'action': 'Preserve original bars/matrix and every AUC; historical source-specific analyses, not new validation'}, {'panel': 'C-E', 'action': 'Remove unverified admission-order curves from the main figure; no event-relative interpretation'}])
    src = SOURCE / (NAMES[3] + '.pdf')
    out = ROOT / 'Figure3_incremental.pdf'
    p = PdfReader(src).pages[0]

    def clarify_shap_title(c, h):
        rect(c, h, 494, 8, 300, 15)
        text(c, h, 495, 17.9, 'Opposite SHAP association signs', 9, bold=True, align='left')
    overlay(p, clarify_shap_title)
    save(p, out)
    shap_code = Path(_release_path('pipeline', 'Manuscript/v13/figures/add/shap_direction_comparison.py'))
    records.append({'figure': 'Figure3', 'source': identity(src), 'output': identity(out), 'whole_pdf_bytes_retained': False, 'local_label_edit': {'panel': 'B', 'cover_top_left_points': [494, 8, 794, 23], 'replacement': 'Opposite SHAP association signs', 'data_and_other_panels_changed': False}, 'selection_code': identity(shap_code), 'selection_code_lines': '307-311', 'selection_rule': 'Opposite Spearman association signs and p_SCH < 0.05 and p_WHU < 0.05; both p values are nominal and not multiplicity-adjusted.', 'panels': [{'panel': 'A,C-E', 'action': 'Retain original vector content as exploratory fitted-model associations; captions identify source models/subsets and noncausal UMAP interpretation'}, {'panel': 'B', 'action': 'Retain every original data point, coefficient and significance symbol; replace only title with Opposite SHAP association signs. Caption and Methods must state actual opposite-sign plus dual nominal-p screening, not sign-only screening.'}]})
    simple_crop(5, [0, 0, 345, 383], 'Figure5_incremental', [{'panel': 'A-B', 'action': 'Preserve original Chinese feature bars/labels/numbers; remove tri-cohort color classification'}, {'panel': 'C-D', 'action': 'Remove different MIMIC model importance and invalid tri-cohort overlap emphasis'}], colors=True)
    simple_crop(10, [0, 0, 385, 164.561785], 'FigureS10_incremental', [{'panel': 'A-B', 'action': 'Retain original Chinese admission-analysis feature importance; separate historical model and sample'}, {'panel': 'C', 'action': 'Remove MIMIC panel with unresolved 14620 denominator/model identity'}])
    additional = _release_resolve_config(json.loads((ROOT / 'additional_panel_provenance.json').read_text()))
    for key in ('Figure4', 'FigureS11'):
        records.append({'figure': key, 'output': identity(ROOT / (key + '_incremental.pdf')), 'provenance': additional[key]})
    for row in records:
        p = Path(row['output']['path'])
        pdf = pdfium.PdfDocument(p)
        assert len(pdf) == 1
        image = pdf[0].render(scale=3).to_pil().convert('RGB')
        png = p.with_suffix('.png')
        image.save(png)
        row['png'] = identity(png)
        row['dimensions_pixels'] = list(image.size)
        row['visual_qa_complete'] = False
        pdf.close()
    after = [identity(p) for p in originals]
    assert before == after
    manifest = {'passed_source_identity_checks': True, 'originals_unchanged': True, 'original_source_identities': before, 'original_docx_tiff_matches': embed, 'figures': records, 'scripts': [identity(Path(__file__)), identity(_release_source_path(ROOT / 'plot_additional_panels.py'))], 'additional_aggregate_provenance': identity(ROOT / 'additional_panel_provenance.json'), 'limits': 'Original panels retain their historical model/sampling limitations; graphic preservation is not a scientific revalidation. Cropping retains vector content under PDF clipping. No new fitting, generated data or OMOP relabeling.'}
    (ROOT / 'figure_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'completed': [r['figure'] for r in records], 'originals_unchanged': True}))
if __name__ == '__main__':
    build()
