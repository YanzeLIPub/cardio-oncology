from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream
from PIL import Image
from io import BytesIO
import json, re, sys
import pymupdf as fitz
sys.path.pop(0)
R = Path(__file__).resolve().parents[1]
ROOT = R.parents[1]
P = ROOT / 'revision_2026-11-03/supplement_streamline_v1'
source = P / 'submission_package/upload/Figure_3.pdf'
palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
groups = ['SCH No MACE', 'SCH MACE', 'WHU No MACE', 'WHU MACE']
counts = [4772, 228, 1274, 279]
r = PdfReader(source)
w = PdfWriter()
w.clone_document_from_reader(r)
page = w.pages[0]
full = page['/Resources']['/XObject']['/fzFrm0'].get_object()['/Resources']['/XObject']['/fullpage'].get_object()
color_rx = re.compile(b'(?P<a>[-+.\\d]+)\\s+(?P<b>[-+.\\d]+)\\s+(?P<c>[-+.\\d]+)\\s+(?P<op>rg|RG)\\b')

def rgb_bytes(hexcode):
    return ' '.join((f'{int(hexcode[i:i + 2], 16) / 255:.9f}' for i in (1, 3, 5))).encode('ascii')
layer = full['/Resources']['/XObject']['/Fm26'].get_object()
before = layer.get_data()
matches = list(color_rx.finditer(before))
rg = [m for m in matches if m['op'] == b'rg']
assert len(rg) == 4
offsets = [m.start() for m in rg]
replacements = []
for m in matches:
    group = max((i for i, start in enumerate(offsets) if start <= m.start()))
    replacements.append((m.start(), m.end(), rgb_bytes(palette[group]) + b' ' + m['op']))
after = before
for a, b, value in reversed(replacements):
    after = after[:a] + value + after[b:]
assert color_rx.sub(b'COLOR', before) == color_rx.sub(b'COLOR', after)
layer.set_data(after)
legend_changes = 0
for i, n in enumerate(range(27, 31)):
    form = full['/Resources']['/XObject'][f'/Fm{n}'].get_object()
    for leafref in form['/Resources']['/XObject'].values():
        leaf = leafref.get_object()
        old = leaf.get_data()
        new, nchanges = color_rx.subn(lambda m: rgb_bytes(palette[i]) + b' ' + m['op'], old)
        assert nchanges == 1
        assert color_rx.sub(b'COLOR', old) == color_rx.sub(b'COLOR', new)
        leaf.set_data(new)
        legend_changes += nchanges
target = R / 'figures/Figure_3.pdf'
with target.open('wb') as f:
    w.write(f)
pdf = fitz.open(target)
oldpdf = fitz.open(source)
assert pdf[0].get_text() == oldpdf[0].get_text()
assert not pdf[0].get_images(full=True)
pdf[0].get_pixmap(dpi=600, alpha=False).save(R / 'figures/Figure_3.png')
(R / 'figures/Figure_3.svg').write_text(pdf[0].get_svg_image(text_as_path=True), encoding='utf-8')
assert '<image' not in (R / 'figures/Figure_3.svg').read_text(encoding='utf-8')
im = Image.open(R / 'figures/Figure_3.png')
width, height = im.size
assert (width, height) == (3900, 4650)
im.crop((0, int(0.657 * height), width, height)).save(R / 'figures/Figure_3e.png')
clip = fitz.Rect(0, 558 * 0.657, 468, 558)
detail = fitz.open()
p = detail.new_page(width=clip.width, height=clip.height)
p.show_pdf_page(p.rect, pdf, 0, clip=clip)
detail.save(R / 'figures/Figure_3e.pdf', garbage=4, deflate=True)
(R / 'figures/Figure_3e.svg').write_text(detail[0].get_svg_image(text_as_path=True), encoding='utf-8')
detail[0].get_pixmap(dpi=180, alpha=False).save(R / 'figures/Figure_3e_preview.png')
newreader = PdfReader(target)
newfull = newreader.pages[0]['/Resources']['/XObject']['/fzFrm0'].get_object()['/Resources']['/XObject']['/fullpage'].get_object()
newlayer = newfull['/Resources']['/XObject']['/Fm26'].get_object()
observed = []
for operands, op in ContentStream(newlayer, newreader).operations:
    if op == b'rg':
        observed.append({'rgb': [float(x) for x in operands], 'points': 0})
    if op == b'Do':
        observed[-1]['points'] += 1
assert [g['points'] for g in observed] == counts
report = {'source_pdf': source.as_posix(), 'output_pdf': target.as_posix(), 'groups': [{'label': label, 'color': color, 'points': n} for label, color, n in zip(groups, palette, counts)], 'point_geometry_and_order_unchanged': True, 'point_color_operators_changed': len(replacements), 'legend_color_operators_changed': legend_changes, 'all_text_unchanged': True, 'raster_image_objects': 0, 'canvas_size_points': [468, 558], 'png_size': [width, height]}
(R / 'review/figure_change.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
