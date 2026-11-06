"""Export current-numbered figures with native paths and no raster XObjects."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import sys, json, re, shutil, csv, colorsys, math
R = Path(__file__).resolve().parents[1]
ROOT = R.parents[1]
import numpy as np, matplotlib
from matplotlib.colors import TwoSlopeNorm
import pymupdf as f
sys.path.pop(0)
P = ROOT / 'revision_2026-11-02/language_refinement_v1/figures'
Q = R / 'review'
OUT = R / 'figures'
OUT.mkdir(exist_ok=True)
for sub in ['PDF', 'SVG', 'previews']:
    (OUT / sub).mkdir(exist_ok=True)
sources = {x['figure']: Path(x['source']) for x in _release_resolve_config(json.loads((Q / 'source_inventory.json').read_text()))}
records = []
RGB = re.compile(b'(?<![\\w.])([-+]?\\d*\\.?\\d+)\\s+([-+]?\\d*\\.?\\d+)\\s+([-+]?\\d*\\.?\\d+)\\s+(rg|RG)\\b')

def recolor(doc):

    def color(m):
        r, g, b = [float(m[i]) for i in [1, 2, 3]]
        if g > r + 0.06 and g > b + 0.06:
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            r, g, b = colorsys.hls_to_rgb(0.735, l, s)
        return f'{r:.7f} {g:.7f} {b:.7f} '.encode() + m[4]
    for x in range(1, doc.xref_length()):
        if not doc.xref_is_stream(x) or doc.xref_get_key(x, 'Subtype')[1] == '/Image' or doc.xref_get_key(x, 'Length1')[0] != 'null':
            continue
        raw = doc.xref_stream(x)
        if b' rg' in raw or b' RG' in raw:
            doc.update_stream(x, RGB.sub(color, raw))

def font_text(pg, origin, text, size=9, bold=False, color=(0, 0, 0)):
    file = _release_font_path('arialbd.ttf') if bold else _release_font_path('arial.ttf')
    pg.insert_text(origin, text, fontname='VectorArialBold' if bold else 'VectorArial', fontfile=file, fontsize=size, color=color)

def panel_letters(pg, n):
    edits = []
    for b in pg.get_text('dict')['blocks']:
        for l in b.get('lines', []):
            for s in l['spans']:
                if re.match('^[A-I](?:$|\\s{2,})', s['text']):
                    edits.append((s, re.sub('^([A-I])', lambda m: m[1].lower() + ')', s['text'])))
    for s, t in edits:
        pg.add_redact_annot(s['bbox'], fill=None, cross_out=False)
    if edits:
        pg.apply_redactions(images=0, graphics=0, text=0)
    for s, t in edits:
        font_text(pg, s['origin'], t, s['size'], True)
    if n in [4, 13, 14]:
        jobs = {4: (680, [(122, 50), (326, 50), (531, 50)]), 13: (680, [(44, 48), (274, 48), (504, 48)]), 14: (680, [(47, 47), (387, 47)])}
        w, xy = jobs[n]
        scale = pg.rect.width / w
        for j, (x, y) in enumerate(xy):
            font_text(pg, (x * scale, (y + 8) * scale), chr(97 + j) + ')', 10 * scale, True)
    if n == 21:
        names = ['Prealbumin', 'Hemoglobin', 'Magnesium', 'Globulin', 'Hematocrit', 'ALP', 'Plateletcrit']
        titleboxes = [pg.search_for(t)[0] for t in names]
        for j, box in enumerate(titleboxes):
            col = j % 3
            x = pg.rect.width * (0.043 + col * 0.332)
            font_text(pg, (x, box.y0 + 6), chr(97 + j) + ')', 8, True)

def image_rgb(doc, xref):
    p = f.Pixmap(doc, xref)
    if p.n != 3:
        p = f.Pixmap(f.csRGB, p)
    return np.frombuffer(p.samples, np.uint8).reshape(p.height, p.width, 3)

def replace_grid(doc, xref, values, expected_colors=None):
    """Replace a plotted matrix image with one rectangle for each original cell."""
    a = image_rgb(doc, xref)
    ny, nx = np.shape(values)
    colors = np.array([[a[min(int((y + 0.5) * a.shape[0] / ny), a.shape[0] - 1), min(int((x + 0.5) * a.shape[1] / nx), a.shape[1] - 1)] for x in range(nx)] for y in range(ny)])
    delta = None
    if expected_colors is not None:
        delta = int(np.abs(colors.astype(int) - np.asarray(expected_colors).astype(int)).max())
        assert delta <= 2, (xref, 'matrix palette mismatch', delta)
    cmds = []
    for y in range(ny):
        for x in range(nx):
            col = ' '.join((f'{v / 255:.7f}' for v in colors[y, x]))
            cmds.append(f'{col} rg {x / nx:.10f} {1 - (y + 1) / ny:.10f} {1 / nx + 0.00015:.10f} {1 / ny + 0.00015:.10f} re f')
    doc.update_object(xref, '<< /Type /XObject /Subtype /Form /FormType 1 /BBox [0 0 1 1] /Resources << >> >>')
    doc.update_stream(xref, ('\n'.join(cmds) + '\n').encode())
    return {'rows': ny, 'columns': nx, 'cell_palette_max_rgb_difference': delta, 'values': np.asarray(values).tolist()}

def replace_colorbar(doc, xref):
    a = image_rgb(doc, xref)
    column = a[:, a.shape[1] // 2, :]
    starts = [0] + [i for i in range(1, len(column)) if not np.array_equal(column[i], column[i - 1])] + [len(column)]
    cmds = []
    for lo, hi in zip(starts[:-1], starts[1:]):
        col = ' '.join((f'{v / 255:.7f}' for v in column[lo]))
        cmds.append(f'{col} rg 0 {1 - hi / len(column):.10f} 1 {(hi - lo + 0.6) / len(column):.10f} re f')
    doc.update_object(xref, '<< /Type /XObject /Subtype /Form /FormType 1 /BBox [0 0 1 1] /Resources << >> >>')
    doc.update_stream(xref, ('\n'.join(cmds) + '\n').encode())
    return len(cmds)

def cmap_rgb(name, values, vmin, vmax):
    return (matplotlib.colormaps[name]((np.asarray(values) - vmin) / (vmax - vmin))[:, :, :3] * 255).astype(np.uint8)

def matrix_sources(key):
    if key == 'S1':
        src = ROOT / 'revision_2026-10-15/incremental_revision_v1/figures/S11_feature_index.json'
        d = _release_resolve_config(json.loads(src.read_text(encoding='utf-8')))
        a = np.array([[r[c + '_missing_percent'] for r in d['China']] for c in ['SCH', 'WHU']])
        b = np.array([[r['missing_percent'] for r in d['MIMIC']]])
        return ([(a, cmap_rgb('YlGnBu', a, 0, 100)), (b, cmap_rgb('YlGnBu', b, 0, 100))], [src.as_posix()])
    if key == 'S7':
        a = np.array([[5, 0.05, 1, 1], [3, 0.05, 1, 1], [5, 0.05, 5, 5], [5, 0.025, 1, 1]])
        relative = np.log2(a / a[0])
        norm = TwoSlopeNorm(vmin=-np.log2(5), vcenter=0, vmax=np.log2(5))
        color = (matplotlib.colormaps['PuOr_r'](norm(relative))[:, :, :3] * 255).astype(np.uint8)
        return ([(a, color)], [_release_source_path(ROOT / 'revision_2026-11-02/graphical_panels_v1/editorial/plot_panels.py').as_posix()])
    if key == 'S19':
        folder = ROOT / 'revision_2026-10-18/four_analysis_v1/shap_seeds/results'
        files = [folder / s for s in ['seed_rank_stability.csv', 'retention.csv', 'publication_summary.json']]
        rows = list(csv.DictReader(files[0].open(encoding='utf-8-sig')))
        ret = list(csv.DictReader(files[1].open(encoding='utf-8-sig')))
        summary = _release_resolve_config(json.loads(files[2].read_text()))
        seeds = summary['seeds']
        result = []
        for c in ['SCH', 'WHU']:
            a = np.eye(4)
            for r in rows:
                if r['cohort'] == c:
                    i, j = (seeds.index(int(r['first_seed'])), seeds.index(int(r['second_seed'])))
                    a[i, j] = a[j, i] = float(r['rho_nonzero'])
            result.append((a, cmap_rgb('YlGnBu', a, 0.8, 1)))
        a = np.zeros((8, 4), int)
        for i, feature in enumerate(summary['original_eight']):
            for j, seed in enumerate(seeds):
                row = next((r for r in ret if r['feature'] == feature and int(r['seed']) == seed))
                if row['opposite_both_q_and_five_fold_agreement'] == 'True':
                    a[i, j] = 1 if row['same_pooled_signs_as_seed42'] == 'True' else 2
        color = np.array([[240, 240, 240], [33, 140, 140], [198, 132, 56]], np.uint8)[a]
        result.append((a, color))
        return (result, [p.as_posix() for p in files])
    return ([], [])

def scatter_forms(doc):
    old = doc[0]
    rep = f.open(Q / 'S25_raster_replay.pdf')
    vec = f.open(Q / 'S25_scatter_layers.pdf')
    recolor(vec)

    def regression(pg):
        return [p for p in pg.get_drawings() if p.get('dashes') != '[] 0' and p.get('color') and (max(p['color']) - min(p['color']) > 0.2)]
    oldlines, newlines = (regression(old), regression(rep[0]))
    assert len(oldlines) == len(newlines) == 2
    images = [im for im in old.get_images() if im[2] > 100 and im[3] > 100]
    align = []
    old_contents = old.get_contents()
    fullxref = old.show_pdf_page(old.rect, vec, 0)
    old.set_contents(old_contents[0]) if len(old_contents) == 1 else None
    if len(old_contents) != 1:
        doc.xref_set_key(old.xref, 'Contents', '[' + ' '.join((f'{v} 0 R' for v in old_contents)) + ']')
    for im, a, b in zip(images, oldlines, newlines):
        ap = np.array([list(t[1]) for t in a['items']] + [list(a['items'][-1][2])])
        bp = np.array([list(t[1]) for t in b['items']] + [list(b['items'][-1][2])])
        assert ap.shape == bp.shape
        delta = np.median(ap - bp, axis=0)
        residual = float(np.max(np.abs(ap - bp - delta)))
        assert residual < 0.001
        box = old.get_image_rects(im[0])[0]
        src = f.Rect(box.x0 - delta[0], box.y0 - delta[1], box.x1 - delta[0], box.y1 - delta[1])
        w, h = (box.width, box.height)
        stream = f'q {1 / w:.12f} 0 0 {1 / h:.12f} {-src.x0 / w:.12f} {-(vec[0].rect.height - src.y1) / h:.12f} cm /Scatter Do Q\n'
        doc.update_object(im[0], f'<< /Type /XObject /Subtype /Form /FormType 1 /BBox [0 0 1 1] /Resources << /XObject << /Scatter {fullxref} 0 R >> >> >>')
        doc.update_stream(im[0], stream.encode())
        align.append({'translation_pt': delta.tolist(), 'max_regression_coordinate_residual_pt': residual})
    return align

def open_source(path):
    if path.suffix == '.svg':
        text = path.read_text(encoding='utf-8')
        text = re.sub('font-family:[^;\\"]+', 'font-family: sans-serif', text)
        return f.open('pdf', f.open('svg', text.encode()).convert_to_pdf())
    return f.open(path)

def export(key):
    stem = 'Figure_' + key if not key.startswith('S') else 'Supplementary_Figure_' + key
    native = P / (stem + '.pdf')
    src = native if native.exists() else sources[key]
    if key in ['S4', 'S13', 'S14']:
        src = Q / f'replotted_{key}.pdf'
    doc = open_source(src)
    pg = doc[0]
    before_text = pg.get_text()
    rec = {'figure': key, 'source': src.as_posix(), 'method': 'Existing native vector content retained', 'matrix_sources': []}
    if not native.exists():
        panel_letters(pg, int(key[1:]))
        if key in ['S4', 'S5', 'S13']:
            recolor(doc)
        rec['method'] = 'Recovered native source; current panel lettering and palette restored'
    if key == 'S25':
        rec['scatter_alignment'] = scatter_forms(doc)
        rec['method'] = 'Native content retained; historical scatter coordinates redrawn as vector markers'
    matrices, paths = matrix_sources(key)
    rec['matrix_sources'] = paths
    ims = []
    for im in pg.get_images():
        if im[0] not in [v[0] for v in ims]:
            ims.append(im)
    mi = 0
    replacements = []
    for im in ims:
        if im[2] > im[3] * 0.5:
            assert mi < len(matrices), (key, im, mi)
            values, colors = matrices[mi]
            mi += 1
            replacements.append({'kind': 'matrix', **replace_grid(doc, im[0], values, colors)})
        else:
            replacements.append({'kind': 'colour scale', 'vector_strips': replace_colorbar(doc, im[0])})
    rec['replaced_raster_layers'] = replacements
    after_text = pg.get_text()
    nums = lambda s: sorted(re.findall('(?<![A-Za-z])[-−]?\\d+(?:\\.\\d+)?', s))
    assert nums(before_text) == nums(after_text), (key, 'numeric text altered')
    rec['numeric_text_preserved'] = True
    pdf = OUT / 'PDF' / (stem + '.pdf')
    if native.exists() and (not replacements) and (key != 'S25'):
        shutil.copy2(native, pdf)
        rec['identical_to_current_pdf'] = True
    else:
        doc.save(pdf, garbage=4, deflate=True)
    doc.close()
    check = f.open(pdf)
    pg = check[0]
    assert len(check) == 1 and (not pg.get_images(full=True)), (key, 'remaining raster content')
    assert len(pg.get_drawings()) > 0
    svg = pg.get_svg_image(text_as_path=True)
    assert '<image' not in svg
    (OUT / 'SVG' / (stem + '.svg')).write_text(svg, encoding='utf-8')
    pg.get_pixmap(matrix=f.Matrix(min(1.8, 1300 / pg.rect.width), min(1.8, 1300 / pg.rect.width)), alpha=False).save(OUT / 'previews' / (stem + '.png'))
    rec.update({'pdf': pdf.relative_to(R).as_posix(), 'svg': (OUT / 'SVG' / (stem + '.svg')).relative_to(R).as_posix(), 'page_size_pt': list(pg.rect), 'raster_images': 0, 'svg_image_elements': 0, 'vector_paths': len(pg.get_drawings()), 'text_characters': len(pg.get_text())})
    records.append(rec)
    print(key, 'exported', flush=True)
if __name__ == '__main__':
    for key in [str(i) for i in range(1, 6)] + ['S' + str(i) for i in range(1, 30)]:
        export(key)
    (R / 'vector_manifest.json').write_text(json.dumps({'baseline': P.parent.as_posix(), 'figures': records, 'count': len(records), 'model_fitting': False, 'scientific_estimates_changed': False}, ensure_ascii=False, indent=2), encoding='utf-8')
