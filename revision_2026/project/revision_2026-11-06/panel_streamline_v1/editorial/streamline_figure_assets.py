"""Rearrange published vector objects; retain numerical marks without refitting."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import json, re, sys
import pymupdf as f
R = Path(__file__).resolve().parents[1]
P = R.parent / 'consistency_affiliations_v1/submission_package/author_reference'
OUT = R / 'figures'
review = []

def source(n):
    stem = 'Figure_2' if n == '2' else 'Supplementary_Figure_S' + n
    return f.open(P / ('main_figures' if n == '2' else 'supplementary_figures') / (stem + '.pdf'))

def text(pg, point, t, size=8, bold=False, align=None):
    fontfile = _release_font_path('arialbd.ttf') if bold else _release_font_path('arial.ttf')
    font = f.Font(fontfile=fontfile)
    x, y = point
    if align == 'center':
        x -= font.text_length(t, fontsize=size) / 2
    pg.insert_text((x, y), t, fontname='PanelArialBold' if bold else 'PanelArial', fontfile=fontfile, fontsize=size)

def rename_titles(doc, replacements):
    pg = doc[0]
    jobs = []
    for b in pg.get_text('dict')['blocks']:
        for l in b.get('lines', []):
            for s in l['spans']:
                key = s['text'][:2]
                if key in replacements:
                    jobs.append((s, replacements[key]))
                    pg.add_redact_annot(s['bbox'], fill=(1, 1, 1), cross_out=False)
    if jobs:
        pg.apply_redactions(images=0, graphics=0, text=0)
    for s, t in jobs:
        text(pg, s['origin'], t, s['size'], True)
    return jobs

def export(doc, stem, detail):
    pg = doc[0]
    assert len(pg.get_images()) == 0
    if '--only-s22' in sys.argv and stem != 'Supplementary_Figure_S22':
        detail.update(stem=stem, dimensions_points=list(pg.rect), pdf_raster_objects=0, svg_raster_images=0)
        review.append(detail)
        return
    pdf = OUT / (stem + '.pdf')
    doc.save(pdf, garbage=4, deflate=True)
    svg = pg.get_svg_image(text_as_path=True)
    assert '<image' not in svg
    (OUT / (stem + '.svg')).write_text(svg, encoding='utf-8')
    pg.get_pixmap(matrix=f.Matrix(3, 3)).save(OUT / (stem + '.png'))
    pg.get_pixmap(matrix=f.Matrix(1.7, 1.7)).save(R / 'review' / (stem + '_final.png'))
    detail.update(stem=stem, dimensions_points=list(pg.rect), pdf_raster_objects=0, svg_raster_images=0)
    review.append(detail)
s = source('2')
d = f.open()
pg = d.new_page(width=432, height=411)
text(pg, (216, 16), 'Transfer AUC matrix', 11, True, 'center')
text(pg, (216, 29), '(row=train, col=test)', 9.5, True, 'center')
pg.show_pdf_page(f.Rect(36, 35, 396, 344), s, 0, clip=f.Rect(260, 28, 514, 246))
pg.show_pdf_page(f.Rect(12, 355, 420, 401), s, 0, clip=f.Rect(10, 252, 393, 295))
export(d, 'Figure_2', {'source': 'v24 Figure 2b and original three CI notes', 'removed': ['2a'], 'panel_crops': {'2': [0, 0, 1, 1]}, 'numerical_marks': 'Unchanged clipped vector objects; no single-panel letter.'})
d.close()
s.close()
s = source('21')
t = source('22')
d = f.open()
pg = d.new_page(width=s[0].rect.width, height=s[0].rect.height)
pg.show_pdf_page(f.Rect(0, 35, pg.rect.width, pg.rect.height), s, 0, clip=f.Rect(0, 35, s[0].rect.width, s[0].rect.height))
text(pg, (pg.rect.width / 2, 18), 'Recorded-label proportions by feature tertile', 10, True, 'center')
pg.show_pdf_page(f.Rect(214, 405, 425, 547), t, 0, clip=f.Rect(0, 32, 235, 190), keep_proportion=False)
pg.show_pdf_page(f.Rect(428, 405, 639, 551), t, 0, clip=f.Rect(249, 205, 478, 363), keep_proportion=False)
text(pg, (241.9, 400), 'h)', 8, True)
text(pg, (334.5, 401), 'NSE', 8, True, 'center')
text(pg, (456.1, 400), 'i)', 8, True)
text(pg, (547.7, 401), 'WBC', 8, True, 'center')
export(d, 'Supplementary_Figure_S21', {'source': 'v24 S21a-g plus S22a/d', 'panel_mapping': {'old S21a-g': 'new S21a-g', 'old S22a': 'new S21h', 'old S22d': 'new S21i'}, 'panel_crops': {'S21h': [0.331, 0.704, 0.663, 1], 'S21i': [0.663, 0.704, 1, 1], 'S21hi': [0.331, 0.704, 1, 1]}, 'numerical_marks': 'Original seven panels retained; h is SCH NSE and i WHU WBC. No new bars or uncertainty intervals.'})
s.close()
t.close()
d.close()
s = source('23')
rename_titles(s, {'c)': 'b)  Cancer-code coverage', 'd)': 'c)  NSE-albumin', 'e)': 'd)  NSE-WBC', 'f)': 'e)  WBC-albumin'})
d = f.open()
pg = d.new_page(width=504, height=680)
pg.show_pdf_page(f.Rect(38, 0, 466, 259), s, 0, clip=f.Rect(0, 0, 306, 185))
for x, c, label in [(48, (0.17255, 0.48235, 0.71373), 'SCH'), (160, (0.90196, 0.38039, 0.00392), 'WHU: CVD admission'), (325, (0.36863, 0.23529, 0.6), 'WHU: Oncology/other')]:
    pg.draw_rect(f.Rect(x, 270, x + 11, 275), fill=c, color=c, overlay=True)
    text(pg, (x + 15, 276), label, 8)
pg.show_pdf_page(f.Rect(35, 292, 499, 421), s, 0, clip=f.Rect(80, 232, 503, 349))
pg.show_pdf_page(f.Rect(0, 435, 504, 676), s, 0, clip=f.Rect(0, 348, 503, 588.1364))
export(d, 'Supplementary_Figure_S22', {'source': 'v24 S23a,c,d,e,f', 'removed': ['old S23b'], 'panel_mapping': {'old S23a': 'new S22a', 'old S23c': 'new S22b', 'old S23d': 'new S22c', 'old S23e': 'new S22d', 'old S23f': 'new S22e'}, 'panel_crops': {'S22a': [0, 0, 1, 0.425], 'S22b': [0, 0.425, 1, 0.63], 'S22cde': [0, 0.633, 1, 1], 'S22bcde': [0, 0.425, 1, 1], 'S22c': [0, 0.633, 0.47, 1], 'S22d': [0.47, 0.633, 0.74, 1], 'S22e': [0.74, 0.633, 1, 1]}, 'numerical_marks': 'Original bar and interval vector objects unchanged; original three-entry legend reflowed. Coverage-panel clip includes both legend rows.'})
d.close()
s.close()
(R / 'review/figure_edits.json').write_text(json.dumps({'edited_figures': review, 'original_numerical_marks_retained': True, 'model_analysis_rerun': False, 'visually_verified': False}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps([{'figure': x['stem'], 'dimensions': x['dimensions_points']} for x in review]))
