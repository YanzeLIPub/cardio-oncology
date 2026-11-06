"""Bounded presentation edit to the frozen vector Figure 3; no patient data or model loading."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import copy, importlib.util, json, math, hashlib
from pathlib import Path
from pypdf import PdfReader
from pypdf.generic import ContentStream, NameObject
import pdfplumber
ROOT = Path(__file__).resolve().parent
P = Path(_release_path('project', 'revision_2026-10-29/box_time_revision_v1'))
spec = importlib.util.spec_from_file_location('base', Path(_release_path('project', 'revision_2026-10-24/figure_readability_v1/figure_work/figure3/improve_figure3_readability.py')))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
SOURCE = P / 'figures/readability_revision/Figure3_readability.pdf'
TARGET = ROOT / 'Figure3_submission.pdf'
_width = b.font_width

def width(page, font, text, maps):
    if '/FirstChar' not in page['/Resources']['/Font'][font] and page['/Resources']['/Font'][font].get('/Subtype') != '/Type0':
        from reportlab.pdfbase.pdfmetrics import stringWidth
        return stringWidth(text, str(page['/Resources']['/Font'][font]['/BaseFont']).lstrip('/'), 1)
    return _width(page, font, text, maps)
b.font_width = width

def frame(rect):
    return [([], b'q'), (b.floats([1]), b'g'), (b.floats([0.8]), b'G'), (b.floats([0.8]), b'w'), (b.floats(rect), b're'), ([], b'B'), ([], b'Q')]

def main():
    r = PdfReader(SOURCE)
    page = copy.deepcopy(r.pages[0])
    maps = b.char_maps(page)
    old = ContentStream(page.get_contents(), r).operations
    out = []
    changes = []
    text_records = []
    font = None
    fs = 1
    line = None
    tm = None
    in_bt = False
    leading = 0
    frame_ops = {586: [245, 720, 145, 66], 1879: [731, 715, 53, 34], 6658: [273, 277, 113, 35], 6985: [675, 59, 112, 61]}
    rect_ops = {592: [250, 766, 12, 6], 602: [250, 748, 12, 6], 612: [250, 735, 12, 6], 621: [250, 723, 12, 6], 6664: [279, 297.213, 17.367, 6.079], 6680: [279, 284.73, 17.367, 6.079]}
    marker_ops = {1883: -9, 1900: -9, 1925: -9, 1942: -9}
    form_marker_ops = {6990: (684 - 448.365, 110 - 299.5895), 7005: (684 - 448.365, 96 - 287.3005), 7019: (684 - 549.4565, 82 - 299.5895), 7033: (684 - 549.4565, 68 - 287.3005)}
    quarter = {420: (334, 953), 423: (336, 940), 426: (58, 740), 429: (60, 727), 434: (345, 808), 438: (345, 795), 441: (59, 945), 443: (59, 932)}
    for index, (args, op) in enumerate(old):
        if index == 1937:
            assert op == b'RG'
            out.append((b.floats([0.843, 0.098, 0.11]), op))
            changes.append({'op': index, 'kind': 'WHU_legend_edge_matches_red_fill'})
            continue
        if index in form_marker_ops:
            dx, dy = form_marker_ops[index]
            out.extend([([], b'q'), (b.floats([1, 0, 0, 1, dx, dy]), b'cm'), (args, op), ([], b'Q')])
            changes.append({'op': index, 'kind': 'outcome_legend_marker_translation'})
            continue
        if index in frame_ops:
            assert op == b'Do'
            out += frame(frame_ops[index])
            changes.append({'op': index, 'kind': 'legend_frame'})
            continue
        if index in rect_ops:
            assert op == b're'
            out.append((b.floats(rect_ops[index]), op))
            changes.append({'op': index, 'kind': 'legend_swatch'})
            continue
        if index in marker_ops:
            assert op == b'cm'
            a = list(map(float, args))
            a[4] += marker_ops[index]
            out.append((b.floats(a), op))
            changes.append({'op': index, 'kind': 'legend_marker_translation'})
            continue
        if op == b'BT':
            in_bt = True
            line = tm = None
        if op == b'ET':
            in_bt = False
        if op == b'Tf':
            font = str(args[0])
            fs = float(args[1])
        if op == b'TL':
            leading = float(args[0])
        if in_bt and op == b'Tm':
            line = list(map(float, args))
            tm = line.copy()
            continue
        if in_bt and op == b'Td':
            tx, ty = map(float, args)
            line[4] += tx * line[0] + ty * line[2]
            line[5] += tx * line[1] + ty * line[3]
            tm = line.copy()
            continue
        if in_bt and op == b'T*':
            if line is not None:
                line[4] -= leading * line[2]
                line[5] -= leading * line[3]
                tm = line.copy()
            continue
        if in_bt and op == b'Tj':
            txt = b.decode(args[0], font, maps)
            m = tm.copy()
            size = math.hypot(m[0], m[1]) * fs
            x, y = m[4:6]
            target = 35 <= index <= 626 or 650 <= index <= 1958 or 6389 <= index <= 7047
            if target:
                newsize = max(size, 12.1 if 6389 <= index <= 7047 else 11.2)
                ns = newsize / fs
                ratio = newsize / size
                oldw = b.font_width(page, font, txt, maps) * size
                m[:4] = [v * ratio for v in m[:4]]
                newtxt = txt
                if index in quarter:
                    m[4], m[5] = quarter[index]
                elif index == 447:
                    m[4] = 51.1
                    m[5] = 963
                elif index in (451, 452, 454):
                    m[4] = 44.1846 + (x - 44.1846) * ratio
                elif index in (141, 142, 145):
                    m[4] = 118.3 + (x - 134.1616) * ratio
                elif index in (267, 268, 271):
                    m[5] = 744.4 + (y - 760.6846) * ratio
                elif index == 576:
                    m[4] = 149
                    m[5] = 760
                elif index in (790, 791, 793):
                    m[4] = 589.1 + (x - 596.47995) * ratio
                elif index == 796:
                    m[4] = 557.3
                    m[5] = 677.8
                elif index in (7041, 7042, 7044):
                    m[4] = 222.8 + (x - 235.8589) * ratio
                elif index in (598, 608, 617, 626):
                    m[4] = 267
                    m[5] = {598: 774, 608: 748, 617: 735, 626: 723}[index]
                elif index == 1917:
                    m[4] = 751.0
                    m[5] = 733
                elif index == 1958:
                    m[4] = 751.0
                    m[5] = 719
                elif index in (6670, 6685):
                    m[4] = 301.5
                    m[5] -= 1.1
                elif index in (6996, 7010, 7024, 7038):
                    m[4] = 693
                    m[5] = {6996: 107, 7010: 93, 7024: 79, 7038: 65}[index]
                elif 35 <= index <= 138 or 161 <= index <= 264:
                    newtxt = txt.replace('−0.', '−.').replace('0.', '.').replace('.00', '0')
                    neww = b.font_width(page, font, newtxt, maps) * newsize
                    m[4] = x + oldw / 2 - neww / 2 if y < 720 else x + oldw - neww
                    m[5] = y - (newsize - size) * 0.25
                elif 650 <= index <= 787 or 6389 <= index <= 6498 or 6714 <= index <= 6823:
                    m[4] = x + oldw * (1 - ratio) / 2
                    m[5] = y - (newsize - size) * 0.25
                elif 6517 <= index <= 6626 or 6842 <= index <= 6951:
                    m[4] = x + oldw * (1 - ratio)
                    m[5] = y - (newsize - size) * 0.25
                elif 813 <= index <= 954:
                    m[4] = x + oldw * (1 - ratio)
                    m[5] = y - (newsize - size) * 0.25
                elif index in (140 + 1, 789 + 1, 6499 + 1, 6824 + 1, 6648 + 1, 6975 + 1):
                    m[4] = x + oldw * (1 - ratio) / 2
                elif index in (266 + 1, 6952 + 1, 7046 + 1):
                    m[5] = y + oldw * (1 - ratio) / 2
                    if index == 267:
                        m[4] = 13.0
                elif 1152 <= index <= 1867:
                    if index in (1153, 1237, 1321, 1405, 1489, 1573, 1657, 1741, 1825):
                        m[4] = x + oldw * (1 - ratio)
                if index == 598:
                    parts = ['Reversal (both sig,', 'opposite sign, n=9)']
                    for k, t in enumerate(parts):
                        mm = m.copy()
                        mm[5] -= 13 * k
                        out.extend([(b.floats(mm), b'Tm'), ([b.encoded(t, font, maps)], b'Tj')])
                else:
                    out.extend([(b.floats(m), b'Tm'), ([b.encoded(newtxt, font, maps)], b'Tj')])
                changes.append({'op': index, 'kind': 'text', 'old': txt, 'new': newtxt, 'old_size': size, 'new_size': newsize, 'old_tm': tm.copy(), 'new_tm': m})
            else:
                out.extend([(b.floats(m), b'Tm'), (args, op)])
            text_records.append({'op': index, 'text': txt, 'font': font, 'size': size, 'tm': tm.copy(), 'targeted': target})
            advance = b.font_width(page, font, txt, maps) * fs
            tm[4] += advance * tm[0]
            tm[5] += advance * tm[1]
            continue
        if in_bt and op in (b'TJ', b'TD'):
            raise AssertionError((index, op))
        out.append((args, op))
    stream = ContentStream(page.get_contents(), r)
    stream.operations = out
    page[NameObject('/Contents')] = stream
    b.save_page(page, TARGET)
    newr = PdfReader(TARGET)
    newp = newr.pages[0]
    assert b.canonical(newp['/Resources']) == b.canonical(r.pages[0]['/Resources'])
    assert len(newp.images) == 0
    old_nontext = [(i, a, o) for i, (a, o) in enumerate(old) if o not in {b'Tm', b'Td', b'Tj', b'T*'}]
    expected = []
    for i, a, o in old_nontext:
        if i == 1937:
            expected.append((b.floats([0.843, 0.098, 0.11]), o))
        elif i in form_marker_ops:
            dx, dy = form_marker_ops[i]
            expected.extend([([], b'q'), (b.floats([1, 0, 0, 1, dx, dy]), b'cm'), (a, o), ([], b'Q')])
        elif i in frame_ops:
            expected += frame(frame_ops[i])
        elif i in rect_ops:
            expected.append((b.floats(rect_ops[i]), o))
        elif i in marker_ops:
            ar = list(map(float, a))
            ar[4] += marker_ops[i]
            expected.append((b.floats(ar), o))
        else:
            expected.append((a, o))
    assert b.op_payload(expected) == b.op_payload([(a, o) for a, o in out if o not in {b'Tm', b'Td', b'Tj', b'T*'}])
    outputs = [{'figure': 'Figure3', 'pdf': b.identity(TARGET), 'png': b.render(TARGET, 300, '.png'), 'tif': b.render(TARGET, 600, '.tif'), 'proof': b.render(TARGET, 144, '_165_1mm.png', width_mm=165.1)}]
    crop_page = copy.deepcopy(newp)
    s = ContentStream(crop_page.get_contents(), newr)
    f = None
    kept = []
    for a, o in s.operations:
        if o == b'Tf':
            f = str(a[0])
        if o == b'Tj' and b.decode(a[0], f, maps) in {'A', 'B', 'C', 'D', 'E'}:
            continue
        kept.append((a, o))
    s.operations = kept
    crop_page[NameObject('/Contents')] = s
    for num, rect in [('S18', [0, 0, 418, 318]), ('S19', [418, 0, 805.928, 318]), ('S23', [0, 660, 392, 992.603]), ('S24', [392, 660, 805.928, 992.603])]:
        dest = ROOT / f'Supplementary_Figure_{num}_submission.pdf'
        detail, scale = b.cropped_page(crop_page, rect, 169.16)
        if num in {'S18', 'S19'}:
            from PIL import Image
            ow, oh = Image.open(P / f'figures/integrated_numbering/Supplementary_Figure_{num}.png').size
            height = float(detail.mediabox.width) * oh / ow
            padded = b.PageObject.create_blank_page(width=float(detail.mediabox.width), height=height)
            padded.merge_transformed_page(detail, b.Transformation().translate(0, height - float(detail.mediabox.height)))
            detail = padded
        b.save_page(detail, dest)
        outputs.append({'figure': num, 'pdf': b.identity(dest), 'png': b.render(dest, 300, '.png'), 'tif': b.render(dest, 600, '.tif'), 'proof': b.render(dest, 144, '_169_16mm.png'), 'crop': rect})
    report = {'source': b.identity(SOURCE), 'outputs': outputs, 'changes': changes, 'text_records': text_records, 'all_resources_identical': True, 'nontext_unchanged_except_explicit_legend_frames_swatches_and_markers': True, 'UMAP_forms_point_counts': [6553, 6553], 'models_or_embeddings_fitted': False, 'source_canvas_pt': [float(page.mediabox.width), float(page.mediabox.height)], 'publication_width_mm': 165.1, 'target_min_A_B_pt': 11.2 * (165.1 / 25.4 * 72) / 805.928, 'target_min_E_pt': 12.1 * (165.1 / 25.4 * 72) / 805.928, 'visual_review': 'pending'}
    (ROOT / 'figure3_submission_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'outputs': outputs, 'changes': len(changes)}))
if __name__ == '__main__':
    main()
