"""Presentation-only Figure 3 patch; no data loading, fitting, or recomputation."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import argparse
import ast
import copy
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np
import pdfplumber
import pypdfium2 as pdfium
from pypdf import PageObject, PdfReader, PdfWriter, Transformation
from pypdf.generic import ArrayObject, ByteStringObject, ContentStream, DecodedStreamObject
from pypdf.generic import FloatObject, NameObject, NumberObject, RectangleObject, TextStringObject
FEATURES = [('Prealbumin', 'PA'), ('Hemoglobin', 'Hb'), ('Magnesium', 'Mg'), ('Globulin', 'Glo'), ('WBC', 'WBC'), ('Hematocrit', 'Hct'), ('ALP', 'ALP'), ('Plateletcrit', 'Pct'), ('NSE', 'NSE'), ('T-Chol', 'Chol'), ('HDL-C', 'HDL'), ('Albumin', 'Alb'), ('LDH', 'LDH'), ('Creatinine', 'Cr'), ('CRP', 'CRP')]
ABBREVIATIONS = dict(FEATURES)
MAIN_SIZE = 13.8
DETAIL_NUMBER_SIZE = 5.3
TEXT_OPS = {b'Tm', b'Td', b'TD', b'Tj', b'TJ', b'Tf'}
ROOT = Path(__file__).resolve().parent
COLOR_SOURCE = Path(_release_path('pipeline', 'Manuscript/v13/figures/add/run_C1_C4_analyses.py'))
COLOR_SOURCE_PDF = Path(_release_path('pipeline', 'Manuscript/Submit/FigureS18_CorrelationNetworks.pdf'))
PALETTE_SOURCE = _release_source_path(ROOT.parent / 'umap_sources/runtime_packages/matplotlib/_cm.py')

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def canonical(obj, active=None):
    active = set() if active is None else active
    if hasattr(obj, 'get_object') and obj.__class__.__name__ == 'IndirectObject':
        key = (id(obj.pdf), obj.idnum, obj.generation)
        if key in active:
            return 'cycle'
        return canonical(obj.get_object(), active | {key})
    if hasattr(obj, 'get_data'):
        return {'dictionary': {str(k): canonical(v, active) for k, v in sorted(obj.items()) if str(k) not in {'/Length', '/Filter', '/DecodeParms'}}}
    if isinstance(obj, dict):
        return {str(k): canonical(v, active) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [canonical(v, active) for v in obj]
    if isinstance(obj, bytes):
        return {'bytes': obj.hex()}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)

def op_payload(ops):
    return [[canonical(args), op.decode('ascii')] for args, op in ops]

def floats(values):
    return [FloatObject(v) for v in values]

def char_maps(page):
    maps = {}
    for key, ref in page['/Resources']['/Font'].items():
        font = ref.get_object()
        if font.get('/ToUnicode'):
            text = font['/ToUnicode'].get_data().decode('latin1')
            maps[str(key)] = {bytes.fromhex(a): bytes.fromhex(b).decode('utf-16-be') for a, b in re.findall('<([0-9A-Fa-f]{4})>\\s+<([0-9A-Fa-f]{4})>', text)}
        else:
            maps[str(key)] = None
    return maps

def raw_bytes(value):
    if isinstance(value, bytes):
        return bytes(value)
    if getattr(value, '_original_bytes', None) is not None:
        return value._original_bytes
    if hasattr(value, 'get_original_bytes'):
        return value.get_original_bytes()
    return str(value).encode('cp1252')

def decode(value, font, maps):
    raw = raw_bytes(value)
    if maps[font] is None:
        return raw.decode('cp1252')
    assert len(raw) % 2 == 0
    return ''.join((maps[font][raw[i:i + 2]] for i in range(0, len(raw), 2)))

def text_runs(ops, initial_font, maps):
    font = initial_font
    result = []
    for args, op in ops:
        if op == b'Tf':
            font = str(args[0])
        elif op == b'Tj':
            result.append((decode(args[0], font, maps), font, args[0]))
        elif op == b'TJ':
            raise AssertionError('Unexpected TJ in bounded target text')
    return (result, font)

def font_width(page, fontname, text, maps):
    f = page['/Resources']['/Font'][fontname]
    if f.get('/Subtype') == '/Type0':
        inverse = {v: int.from_bytes(k, 'big') for k, v in maps[fontname].items()}
        descendant = f['/DescendantFonts'][0].get_object()
        widths = {}
        entries = list(descendant.get('/W', []))
        i = 0
        while i < len(entries):
            start = int(entries[i])
            following = entries[i + 1]
            if isinstance(following, (list, ArrayObject)):
                widths.update({start + j: float(w) for j, w in enumerate(following)})
                i += 2
            else:
                widths.update({j: float(entries[i + 2]) for j in range(start, int(following) + 1)})
                i += 3
        return sum((widths.get(inverse[c], float(descendant.get('/DW', 1000))) for c in text)) / 1000
    first = int(f['/FirstChar'])
    widths = f['/Widths']
    values = [float(widths[ord(c) - first]) for c in text]
    assert all((v > 0 for v in values)), (fontname, text, values)
    return sum(values) / 1000

def descent(page, font):
    f = page['/Resources']['/Font'][font]
    if f.get('/Subtype') == '/Type0':
        f = f['/DescendantFonts'][0].get_object()
    return float(f['/FontDescriptor']['/Descent']) / 1000

def encoded(text, font, maps):
    if maps[font] is None:
        return ByteStringObject(text.encode('cp1252'))
    inverse = {v: k for k, v in maps[font].items()}
    return ByteStringObject(b''.join((inverse[c] for c in text)))

def matching_word(words, text, matrix, height):
    e, f = matrix[4:6]
    rotated = abs(matrix[1]) > 0
    candidates = [w for w in words if w['text'] in {text, text[::-1]} and 330 < w['top'] < 625]
    assert candidates, text
    if rotated:
        return min(candidates, key=lambda w: abs((w['x0'] + w['x1']) / 2 - e) + abs(height - w['bottom'] - f))
    return min(candidates, key=lambda w: abs(w['x0'] - e) + abs(height - w['bottom'] - f))

def new_text_ops(page, maps, text, font, matrix, restore_font):
    return [([NameObject(font), NumberObject(1)], b'Tf'), (floats(matrix), b'Tm'), ([encoded(text, font, maps)], b'Tj'), ([NameObject(restore_font), NumberObject(1)], b'Tf')]

def replace_text_preserving_paint_order(segment, new_ops):
    first_show = next((i for i, (_, op) in enumerate(segment) if op in {b'Tj', b'TJ'}))
    return [(a, o) for a, o in segment[:first_show] if o not in TEXT_OPS] + new_ops + [(a, o) for a, o in segment[first_show:] if o not in TEXT_OPS]

def patch_source(reader, words, mode):
    page = reader.pages[0]
    height = float(page.mediabox.height)
    stream = ContentStream(page.get_contents(), reader)
    old_ops = stream.operations
    maps = char_maps(page)
    new_ops, changes, inventories = ([], [], [])
    current_font = None
    i = 0
    while i < len(old_ops):
        args, op = old_ops[i]
        if op == b'Tf':
            current_font = str(args[0])
        if op != b'Tm':
            new_ops.append((args, op))
            i += 1
            continue
        j = i + 1
        while j < len(old_ops) and old_ops[j][1] not in {b'Tm', b'ET'}:
            j += 1
        segment = old_ops[i:j]
        runs, final_font = text_runs(segment, current_font, maps)
        text = ''.join((t for t, _, _ in runs))
        matrix = [float(x) for x in args]
        a, b, c, d, e, f = matrix
        size = math.hypot(a, b)
        in_middle = 350 < f < 660
        numeric = in_middle and any((abs(size - x) < 0.0001 for x in [4.2981, 4.6178, 5.7905]))
        feature = in_middle and text in ABBREVIATIONS
        title = in_middle and (text in {'SCH', 'WHU', 'Δ (WHU − SCH)'} or text.startswith('Feature Correlation Structure:'))
        corner = size == 10 and text in {'C', 'D'} and (635 < f < 655)
        replacement = segment
        if numeric:
            inventory = {'panel': 'C-SCH' if e < 260 else 'C-WHU' if e < 525 else 'D', 'source_font_pt': size, 'display_values': [t for t, _, _ in runs], 'display_value_count': len(runs)}
            inventories.append(inventory)
            if mode == 'main':
                replacement = [(aa, oo) for aa, oo in segment if oo not in {b'Tj', b'TJ'}]
                changes.append({'kind': 'omit_numeric_annotation_only', **inventory})
            elif size < DETAIL_NUMBER_SIZE:
                ratio = DETAIL_NUMBER_SIZE / size
                replacement = []
                font = current_font
                for aa, oo in segment:
                    if oo == b'Tf':
                        font = str(aa[0])
                    if oo == b'Tm':
                        updated = [float(x) for x in aa]
                        updated[:4] = [x * ratio for x in updated[:4]]
                        replacement.append((floats(updated), oo))
                    elif oo == b'Td':
                        replacement.append((floats([float(x) / ratio for x in aa]), oo))
                    elif oo == b'Tj':
                        t = decode(aa[0], font, maps)
                        dx = -(DETAIL_NUMBER_SIZE - size) * font_width(page, font, t, maps) / 2 / DETAIL_NUMBER_SIZE
                        dy = -(DETAIL_NUMBER_SIZE - size) * (descent(page, font) + 0.5) / DETAIL_NUMBER_SIZE
                        replacement.extend([(floats([dx, dy]), b'Td'), (aa, oo), (floats([-dx, -dy]), b'Td')])
                    else:
                        replacement.append((aa, oo))
                changes.append({'kind': 'numeric_font_enlargement', 'panel': inventory['panel'], 'old_pt': size, 'new_pt': DETAIL_NUMBER_SIZE})
        elif mode == 'main' and feature:
            word = matching_word(words, text, matrix, height)
            abbreviation = ABBREVIATIONS[text]
            font = '/TT1' if list(ABBREVIATIONS).index(text) < 9 else '/C2_0'
            width = font_width(page, font, abbreviation, maps) * MAIN_SIZE
            desc = descent(page, font)
            if abs(b) > 0:
                center_x = (word['x0'] + word['x1']) / 2
                new_matrix = [0, MAIN_SIZE, -MAIN_SIZE, 0, center_x + (desc + 0.5) * MAIN_SIZE, height - word['top'] - width]
            else:
                center_y = height - (word['top'] + word['bottom']) / 2
                new_matrix = [MAIN_SIZE, 0, 0, MAIN_SIZE, word['x1'] - width, center_y - (desc + 0.5) * MAIN_SIZE]
            replacement = replace_text_preserving_paint_order(segment, new_text_ops(page, maps, abbreviation, font, new_matrix, final_font))
            changes.append({'kind': 'abbreviate_axis_label', 'old': text, 'new': abbreviation, 'old_matrix': matrix, 'new_matrix': new_matrix, 'orientation': 'vertical' if abs(b) else 'horizontal'})
        elif mode == 'main' and title:
            word_chars = []
            font = '/TT1' if text.startswith('Feature Correlation') else current_font
            width = font_width(page, font, text, maps) * MAIN_SIZE
            if text.startswith('Feature Correlation'):
                center_x, top = (261.0, 326.5)
            elif text == 'SCH':
                center_x, top = (148.67, 344.4)
            elif text == 'WHU':
                center_x, top = (411.0, 344.4)
            else:
                center_x, top = (678.62, 337.0)
            baseline = height - top - (1 + descent(page, font)) * MAIN_SIZE
            new_matrix = [MAIN_SIZE, 0, 0, MAIN_SIZE, center_x - width / 2, baseline]
            replacement = replace_text_preserving_paint_order(segment, new_text_ops(page, maps, text, font, new_matrix, final_font))
            changes.append({'kind': 'enlarge_panel_title', 'text': text, 'old_matrix': matrix, 'new_matrix': new_matrix})
        elif mode == 'detail' and (corner or text.startswith('Feature Correlation Structure:')):
            replacement = [(aa, oo) for aa, oo in segment if oo not in {b'Tj', b'TJ'}]
            changes.append({'kind': 'omit_composite_only_heading_in_detail', 'text': text})
        elif mode == 'detail' and text == 'ABCDE':
            replacement = [(aa, oo) for aa, oo in segment if not (oo == b'Tj' and decode(aa[0], current_font, maps) in {'C', 'D'})]
            changes.append({'kind': 'omit_composite_only_panel_letters_in_detail', 'text': 'C,D'})
        new_ops.extend(replacement)
        current_font = final_font
        i = j
    stream.operations = new_ops
    output_page = copy.deepcopy(page)
    output_page[NameObject('/Contents')] = stream
    old_non_text = op_payload([(a, o) for a, o in old_ops if o not in TEXT_OPS])
    new_non_text = op_payload([(a, o) for a, o in new_ops if o not in TEXT_OPS])
    assert old_non_text == new_non_text, 'Non-text geometry, paint, or image invocation changed'
    assert len(inventories) == 3
    if mode == 'main':
        assert len([x for x in changes if x['kind'] == 'abbreviate_axis_label']) == 86
        assert len([x for x in changes if x['kind'] == 'enlarge_panel_title']) == 4
    return (output_page, changes, inventories)

def save_page(page, path):
    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata({'/Title': path.stem, '/Subject': 'Presentation-only original-vector figure patch'})
    with path.open('wb') as handle:
        writer.write(handle)

def add_colorbars(page, reader):
    source_text = COLOR_SOURCE.read_text(encoding='utf-8')
    assert 'cmap="RdBu_r", vmin=-1, vmax=1' in source_text
    assert 'cmap="PuOr",  vmin=-1, vmax=1' in source_text
    with pdfplumber.open(COLOR_SOURCE_PDF) as pdf:
        historical_text = pdf.pages[0].extract_text()
    assert '−1.00' in historical_text and '1.00' in historical_text
    maps = char_maps(page)
    h = float(page.mediabox.height)
    original = ContentStream(page.get_contents(), reader)
    added = [([], b'q')]
    bars = []
    palette_ast = ast.parse(PALETTE_SOURCE.read_text(encoding='utf-8'))
    palette_anchors = {node.targets[0].id: ast.literal_eval(node.value) for node in palette_ast.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and (node.targets[0].id in {'_RdBu_data', '_PuOr_data'})}
    for cmap_name, x0, x1, label, label_x in [('RdBu_r', 83.0, 220.0, 'ρ', 48.0), ('PuOr', 632.0, 782.0, 'Δρ', 575.0)]:
        anchors = np.asarray(palette_anchors['_RdBu_data' if cmap_name == 'RdBu_r' else '_PuOr_data'])
        if cmap_name == 'RdBu_r':
            anchors = anchors[::-1]
        positions = np.linspace(0, 1, len(anchors))
        palette = np.column_stack([np.interp(np.linspace(0, 1, 256), positions, anchors[:, channel]) for channel in range(3)])
        palette = np.round(palette * 255) / 255
        for index in range(256):
            rgb = palette[index]
            added.extend([(floats(rgb), b'rg'), (floats([x0 + (x1 - x0) * index / 256, h - 627.5, (x1 - x0) / 256, 6]), b're'), ([], b'f')])
        added.extend([(floats([0]), b'g'), (floats([0.6]), b'w')])
        for value, position in [(-1, x0), (0, (x0 + x1) / 2), (1, x1)]:
            added.extend([(floats([position, h - 627.5]), b'm'), (floats([position, h - 630.0]), b'l'), ([], b'S')])
            text = '−1' if value == -1 else str(value)
            width = font_width(page, '/C2_0', text, maps) * MAIN_SIZE
            baseline = h - 631.2 - (1 + descent(page, '/C2_0')) * MAIN_SIZE
            added.append(([], b'BT'))
            added.extend(new_text_ops(page, maps, text, '/C2_0', [MAIN_SIZE, 0, 0, MAIN_SIZE, position - width / 2, baseline], '/C2_0'))
            added.append(([], b'ET'))
        for char in label:
            font = '/C2_1' if char == 'Δ' else '/C2_0'
            baseline = h - 624.5 - (descent(page, font) + 0.5) * MAIN_SIZE
            added.append(([], b'BT'))
            added.extend(new_text_ops(page, maps, char, font, [MAIN_SIZE, 0, 0, MAIN_SIZE, label_x, baseline], font))
            added.append(([], b'ET'))
            label_x += font_width(page, font, char, maps) * MAIN_SIZE + 1
        bars.append({'colormap': cmap_name, 'vmin': -1, 'vmax': 1, 'label': label, 'rect_top_origin_pt': [x0, 621.5, x1, 627.5], 'source_font_pt': MAIN_SIZE, 'vector_color_rectangles': 256})
    added.append(([], b'Q'))
    original.operations = original.operations + added
    page[NameObject('/Contents')] = original
    return {'source_code': identity(COLOR_SOURCE), 'source_pdf': identity(COLOR_SOURCE_PDF), 'palette_definition_source': identity(PALETTE_SOURCE), 'palette_method': '256 linear RGB steps from the named Matplotlib 3.10.8 ColorBrewer anchor definitions; RdBu reversed, channels rounded to 8-bit as verified against original cell colors.', 'source_lines_1_based': [285, 286, 287, 288], 'bars': bars, 'original_heatmap_operators_changed': False}

def cropped_page(source_page, rect, width_mm):
    x0, top, x1, bottom = rect
    h = float(source_page.mediabox.height)
    source = copy.deepcopy(source_page)
    source.cropbox = RectangleObject([x0, h - bottom, x1, h - top])
    source.trimbox = source.cropbox
    width = width_mm / 25.4 * 72
    scale = width / (x1 - x0)
    page = PageObject.create_blank_page(width=width, height=(bottom - top) * scale)
    page.merge_transformed_page(source, Transformation().translate(-x0, -(h - bottom)).scale(scale))
    return (page, scale)

def render(path, dpi, suffix, *, width_mm=None):
    doc = pdfium.PdfDocument(str(path))
    page = doc[0]
    scale = dpi / 72 if width_mm is None else width_mm / 25.4 * dpi / page.get_width()
    image = page.render(scale=scale).to_pil().convert('RGB')
    out = path.with_name(path.stem + suffix)
    if out.suffix.lower() in {'.tif', '.tiff'}:
        image.save(out, compression='tiff_lzw', dpi=(dpi, dpi))
    else:
        image.save(out, dpi=(dpi, dpi))
    page.close()
    doc.close()
    return identity(out)

def raster(path, dpi=144):
    doc = pdfium.PdfDocument(str(path))
    page = doc[0]
    result = np.asarray(page.render(scale=dpi / 72).to_pil().convert('RGB'))
    page.close()
    doc.close()
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    assert ROOT.name == 'figure3' and ROOT.parent.name == 'figure_work'
    with pdfplumber.open(source) as pdf:
        words = pdf.pages[0].extract_words()
    reader = PdfReader(source)
    main_page, main_changes, original_values = patch_source(reader, words, 'main')
    detail_page, detail_changes, detail_values = patch_source(reader, words, 'detail')
    assert original_values == detail_values
    assert canonical(main_page['/Resources']) == canonical(reader.pages[0]['/Resources'])
    assert canonical(detail_page['/Resources']) == canonical(reader.pages[0]['/Resources'])
    colorbar_evidence = add_colorbars(main_page, reader)
    main_pdf = ROOT / 'Figure3_readability.pdf'
    save_page(main_page, main_pdf)
    outputs = [{'figure_id': 'Figure3', 'pdf': identity(main_pdf), 'png': render(main_pdf, 300, '.png'), 'tiff': render(main_pdf, 600, '.tif'), 'actual_165_1mm_preview': render(main_pdf, 144, '_165_1mm_144dpi.png', width_mm=165.1), 'actual_149_52mm_preview': render(main_pdf, 144, '_149_52mm_144dpi.png', width_mm=149.52)}]
    crops = [('S20', 'C-SCH', [0, 347, 263, 620]), ('S21', 'C-WHU', [262, 347, 522, 620]), ('S22', 'D', [523, 342, 805.928, 620])]
    for figure_id, panel, rect in crops:
        out = ROOT / f'Supplementary_Figure_{figure_id}_readability.pdf'
        page, scale = cropped_page(detail_page, rect, 169.16)
        save_page(page, out)
        number_source = DETAIL_NUMBER_SIZE if panel != 'D' else 5.7905
        outputs.append({'figure_id': figure_id, 'panel': panel, 'source_crop_top_origin_pt': rect, 'scale': scale, 'actual_width_mm': 169.16, 'numeric_font_pt': number_source * scale, 'feature_label_font_pt': 6 * scale, 'pdf': identity(out), 'png': render(out, 300, '.png'), 'tiff': render(out, 600, '.tif'), 'actual_169_16mm_preview': render(out, 144, '_169_16mm_144dpi.png')})
    old_image, new_image = (raster(source), raster(main_pdf))
    assert old_image.shape == new_image.shape
    changed = np.any(old_image != new_image, axis=2)
    outside = changed.copy()
    outside[640:1242, :] = False
    outside[1236:1294, 90:446] = False
    outside[1236:1294, 1130:1590] = False
    assert not outside.any(), 'A/B/E or material outside bounded C/D region changed'
    letters = [{'panel': name, 'rect_pt': rect, 'exact_pixels_unchanged': bool(np.array_equal(old_image[int(rect[1] * 2):int(rect[3] * 2), int(rect[0] * 2):int(rect[2] * 2)], new_image[int(rect[1] * 2):int(rect[3] * 2), int(rect[0] * 2):int(rect[2] * 2)]))} for name, rect in [('A', [10, 10, 30, 30]), ('B', [430, 10, 455, 30]), ('C', [15, 338, 27, 352]), ('D', [550, 338, 562, 352]), ('E', [15, 654, 27, 668]), ('E_original_title', [235, 640, 550, 654])]]
    assert all((x['exact_pixels_unchanged'] for x in letters))
    width_pt = float(reader.pages[0].mediabox.width)
    report = {'source': identity(source), 'script': identity(__file__), 'scope': 'Presentation only; matrix cells/colors, non-text operators, A/B/E, resources, and panel letters retained. No model or UMAP fitting.', 'abbreviation_key': [{'full_name': k, 'main_label': v, 'C_SCH': True, 'C_WHU': k != 'CRP', 'D': k != 'CRP'} for k, v in FEATURES], 'matrix_dimensions': {'C-SCH': [15, 15], 'C-WHU': [14, 14], 'D': [14, 14]}, 'original_numeric_annotation_inventory': original_values, 'main_text_changes': main_changes, 'detail_text_changes': detail_changes, 'original_non_text_operations_exactly_preserved': True, 'all_source_resources_exactly_preserved': True, 'added_colorbar_evidence': colorbar_evidence, 'outside_C_D_pixels_exactly_preserved_at_144dpi': True, 'panel_letter_checks': letters, 'main_C_D_feature_label_font_pt_at_width': {'149.52mm': MAIN_SIZE * (149.52 / 25.4 * 72) / width_pt, '165.1mm': MAIN_SIZE * (165.1 / 25.4 * 72) / width_pt}, 'outputs': outputs, 'visual_review': 'not_yet_performed', 'remaining_limits': ['A/B/E font sizes and unchanged panel-letter sizes are not certified as 8 pt by this C/D-only patch.', 'Figure 3 E raster/source authentication is managed separately; this patch does not increase image information.', 'Original empty or omitted matrix annotations are preserved as such; no missing correlations are imputed.']}
    (ROOT / 'figure3_readability_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': identity(ROOT / 'figure3_readability_report.json'), 'figure_count': len(outputs), 'main_label_changes': 86, 'numeric_counts': {x['panel']: x['display_value_count'] for x in original_values}}, indent=2))
if __name__ == '__main__':
    main()
