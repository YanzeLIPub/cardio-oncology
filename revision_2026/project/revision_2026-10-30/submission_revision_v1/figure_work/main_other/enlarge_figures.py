"""Bounded typography edits; original geometry/resources retained outside labels."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import copy, hashlib, json, math
import numpy as np
import pdfplumber, pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, FloatObject, TextStringObject
from reportlab.pdfbase.pdfmetrics import stringWidth
OUT = Path(__file__).resolve().parent
P = OUT.parents[3] / 'revision_2026-10-29/box_time_revision_v1'

def H(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def nums(v):
    return [FloatObject(float(x)) for x in v]

def save(r, ops, out):
    writer = PdfWriter()
    writer.clone_document_from_reader(r)
    s = ContentStream(None, writer)
    s.operations = ops
    writer.pages[0].replace_contents(s)
    with out.open('wb') as f:
        writer.write(f)

def render(path):
    with pdfium.PdfDocument(path) as d:
        page = d[0]
        scale = 165.1 * 72 / 25.4 / page.get_width()
        page.render(scale=scale * 2).to_pil().save(path.with_name(path.stem + '_165mm.png'))
        page.render(scale=300 / 72).to_pil().convert('RGB').save(path.with_suffix('.png'), dpi=(300, 300))

def raster(path):
    with pdfium.PdfDocument(path) as d:
        return np.asarray(d[0].render(scale=2).to_pil().convert('RGB'))

def f1():
    src = P / 'figures/Figure1_priority.pdf'
    r = PdfReader(src)
    ops = ContentStream(r.pages[0].get_contents(), r).operations
    new = copy.deepcopy(ops)
    edits = []
    inserts = {}
    targets = {'9,946 admissions / 6,783 patients': 6.7, '1,323 labels': 6.5, '281 labels': 6.5, '4,462 labels': 6.5, 'Recorded label: 4.6%': 6.5, 'Recorded label: 18.1%': 6.5, 'Recorded label: 44.9%': 6.5}
    for i, (a, o) in enumerate(ops):
        if o != b'Tj' or str(a[0]) not in targets:
            continue
        t = str(a[0])
        tf = max((j for j in range(i) if ops[j][1] == b'Tf'))
        tm = max((j for j in range(i) if ops[j][1] == b'Tm'))
        font = 'Helvetica-Bold' if str(ops[tf][0][0]) == '/F2' else 'Helvetica'
        oldsize = float(ops[tf][0][1])
        size = targets[t]
        x = float(ops[tm][0][4])
        y = float(ops[tm][0][5])
        centre = x + stringWidth(t, font, oldsize) / 2
        new[tf][0][1] = FloatObject(size)
        if t.startswith('9,946'):
            line1 = '9,946 admissions /'
            line2 = '6,783 patients'
            new[i][0][0] = TextStringObject(line1)
            new[tm] = (nums([1, 0, 0, 1, centre - stringWidth(line1, font, size) / 2, 340.5]), b'Tm')
            inserts[i] = [(nums([1, 0, 0, 1, centre - stringWidth(line2, font, size) / 2, 332.5]), b'Tm'), ([TextStringObject(line2)], b'Tj')]
        else:
            new[tm][0][4] = FloatObject(centre - stringWidth(t, font, size) / 2)
        edits.append({'text': t, 'old_font_pt': oldsize, 'new_font_pt': size, 'new_at_165mm_pt': size * (165.1 * 72 / 25.4) / float(r.pages[0].mediabox.width), 'Tf_index': tf, 'Tm_index': tm, 'Tj_index': i})
    assert len(edits) == 7
    result = []
    for i, x in enumerate(new):
        result.append(x)
        result.extend(inserts.get(i, []))
    dst = OUT / 'Figure1_priority.pdf'
    save(r, result, dst)
    render(dst)
    allowed = {e[k] for e in edits for k in ['Tf_index', 'Tm_index']}
    allowed.add(next((e['Tj_index'] for e in edits if e['text'].startswith('9,946'))))
    assert all((x == y for i, (x, y) in enumerate(zip(ops, new)) if i not in allowed))
    return {'figure': 1, 'source': str(src), 'pdf': str(dst), 'png': str(dst.with_suffix('.png')), 'edits': edits, 'only_target_text_ops_changed': True, 'all_resources_unchanged': True, 'all_plotted_geometry_unchanged': True}

def mat(a):
    a, b, c, d, e, f = map(float, a)
    return np.array([[a, c, e], [b, d, f], [0, 0, 1]])

def f2():
    src = P / 'figures/Figure2_priority.pdf'
    r = PdfReader(src)
    page = r.pages[0]
    h = float(page.mediabox.height)
    ops = ContentStream(page.get_contents(), r).operations
    new = copy.deepcopy(ops)
    clusters = [('0.657', (34, 49, 104, 110), 0), ('0.485', (50.5, 63.4, 163.9, 168.4), 5), ('0.559', (64.8, 77.6, 138.5, 142.9), 0), ('0.840', (108.4, 123.4, 41.7, 46.8), 0), ('0.438', (124.9, 137.7, 179.9, 184.3), 0), ('0.444', (139.1, 152.0, 177.8, 182.3), 8), ('0.742', (182.8, 197.7, 75.4, 80.5), 0), ('0.500', (199.2, 212.1, 158.8, 163.2), 8), ('0.494', (213.5, 226.4, 160.8, 165.3), 0)]
    found = [[] for _ in clusters]
    ctm = np.eye(3)
    stack = []
    path = []
    coords = []
    path_ctm = None
    for i, (a, o) in enumerate(ops):
        if o == b'q':
            stack.append(ctm.copy())
        elif o == b'Q':
            ctm = stack.pop()
        elif o == b'cm':
            ctm = ctm @ mat(a)
        elif o in [b'm', b'l', b'c', b'v', b'y', b're', b'h']:
            if not path:
                path_ctm = ctm.copy()
            path.append(i)
            if o in [b'm', b'l', b'c', b'v', b'y']:
                for j in range(0, len(a), 2):
                    coords.append(ctm @ np.array([float(a[j]), float(a[j + 1]), 1]))
            elif o == b're':
                x, y, w, ht = map(float, a)
                for x1, y1 in [(x, y), (x + w, y), (x + w, y + ht), (x, y + ht)]:
                    coords.append(ctm @ np.array([x1, y1, 1]))
        elif o in [b'f', b'f*', b'B', b'B*', b'S', b's', b'b', b'b*', b'n']:
            if path and coords and (o in [b'f', b'f*']):
                xs = [q[0] for q in coords]
                ys = [h - q[1] for q in coords]
                box = (min(xs), max(xs), min(ys), max(ys))
                for k, (_, bb, dy) in enumerate(clusters):
                    if box[0] >= bb[0] and box[1] <= bb[1] and (box[2] >= bb[2]) and (box[3] <= bb[3]):
                        assert np.allclose(path_ctm, ctm)
                        found[k].append({'indices': path.copy(), 'paint': i, 'bbox': box, 'ctm': ctm.copy()})
            path = []
            coords = []
            path_ctm = None
    changed = []
    proof = []
    for (label, bb, dy), groups in zip(clusters, found):
        assert len(groups) == 5, (label, len(groups))
        xmin = min((g['bbox'][0] for g in groups))
        xmax = max((g['bbox'][1] for g in groups))
        top = min((g['bbox'][2] for g in groups))
        bottom = max((g['bbox'][3] for g in groups))
        cx = (xmin + xmax) / 2
        cy = h - bottom
        transform = np.array([[1.5, 0, -0.5 * cx], [0, 1.5, -0.5 * cy + dy], [0, 0, 1]])
        for g in groups:
            local = np.linalg.inv(g['ctm']) @ transform @ g['ctm']
            for idx in g['indices']:
                a, o = new[idx]
                if o == b'h':
                    continue
                if o == b're':
                    x, y, w, ht = map(float, a)
                    q = local @ np.array([x, y, 1])
                    q2 = local @ np.array([x + w, y + ht, 1])
                    new[idx] = (nums([q[0], q[1], q2[0] - q[0], q2[1] - q[1]]), o)
                else:
                    v = []
                    for j in range(0, len(a), 2):
                        q = local @ np.array([float(a[j]), float(a[j + 1]), 1])
                        v.extend(q[:2])
                    new[idx] = (nums(v), o)
                changed.append(idx)
        proof.append({'label': label, 'old_bbox': [xmin, xmax, top, bottom], 'new_bbox': [cx + 1.5 * (xmin - cx), cx + 1.5 * (xmax - cx), bottom - 1.5 * (bottom - top) - dy, bottom - dy], 'glyph_paths': len(groups), 'scale': 1.5, 'vertical_offset_pdf_pt': dy, 'old_physical_glyph_height_pt': (bottom - top) * (165.1 * 72 / 25.4) / float(page.mediabox.width), 'new_physical_glyph_height_pt': 1.5 * (bottom - top) * (165.1 * 72 / 25.4) / float(page.mediabox.width)})
    allowed = set(changed)
    assert all((x == y for i, (x, y) in enumerate(zip(ops, new)) if i not in allowed))
    dst = OUT / 'Figure2_priority.pdf'
    save(r, new, dst)
    render(dst)
    before = raster(src)
    after = raster(dst)
    diff = np.any(before != after, axis=2)
    mask = np.zeros(diff.shape, dtype=bool)
    for p in proof:
        a, b = (p['old_bbox'], p['new_bbox'])
        x0 = max(0, int(min(a[0], b[0]) * 2) - 3)
        x1 = int(max(a[1], b[1]) * 2) + 4
        y0 = max(0, int(min(a[2], b[2]) * 2) - 3)
        y1 = int(max(a[3], b[3]) * 2) + 4
        mask[y0:y1, x0:x1] = True
    outside = int(np.sum(diff & ~mask))
    assert outside == 0, outside
    return {'figure': 2, 'source': str(src), 'pdf': str(dst), 'png': str(dst.with_suffix('.png')), 'glyph_clusters': proof, 'changed_path_ops': len(allowed), 'all_other_ops_exact': True, 'all_resources_unchanged': True, 'outside_label_regions_changed_pixels_144dpi': outside, 'text_values_and_shapes_preserved_by_affine_scaling': True, 'precise_font_metadata': 'Original labels are vector outlines; physical glyph heights measured. No exact live-font-size claim.'}
if __name__ == '__main__':
    result = [f1(), f2()]
    for r in result:
        r['pdf_sha256'] = H(r['pdf'])
        r['png_sha256'] = H(r['png'])
        r['publish_pdf_relative'] = 'figures/' + Path(r['pdf']).name
        r['publish_png_relative'] = 'figures/' + Path(r['png']).name
    (OUT / 'main_other_typography.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps([{'figure': x['figure'], 'pdf': x['pdf'], 'edits': len(x.get('edits', x.get('glyph_clusters', [])))} for x in result]))
