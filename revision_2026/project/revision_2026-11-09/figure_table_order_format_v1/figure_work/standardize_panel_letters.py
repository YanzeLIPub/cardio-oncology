"""Give existing vector figures consistent panel letters without changing plots.

Inputs are the preceding revision's vector figure PDFs and crop descriptions.
No clinical records or statistical data are required.
"""
from pathlib import Path
import argparse
import json
import math
import re

import fitz
from pdf_line_weights import adjust_stroke_widths


def spans(page):
    result = []
    for block in page.get_text('rawdict')['blocks']:
        for line in block.get('lines', []):
            for span in line['spans']:
                item = dict(span)
                item['text'] = ''.join(c['c'] for c in span['chars'])
                item['direction'] = line['dir']
                result.append(item)
    return result


def labels(page, key):
    candidates = []
    for span in spans(page):
        text = span['text']
        match = re.match(r'^([a-i])\)', text)
        if not match and 'Bold' in span['font']:
            match = re.match(r'^([a-i])(?:\s{2,}|$)', text)
        if match:
            span['letter'] = match.group(1)
            span['prefix_length'] = 2 if text[1:2] == ')' else 1
            candidates.append(span)
    # These three figures contain covered earlier labels in their vector source.
    # Select the visible, last-composited label rather than the covered text.
    if key == 'S8':
        candidates = [s for s in candidates if
                      s['text'].startswith('a)  Four accepted') or
                      (s['text'] == 'b)' and s['font'] == 'Helvetica-Bold')]
    elif key == 'S14':
        candidates = [s for s in candidates if s['font'] == 'Helvetica-Bold']
    elif key == 'S15':
        candidates = [s for s in candidates if
                      (s['letter'] in 'ab' and s['font'] == 'Arial-BoldMT') or
                      (s['text'] in ('c)', 'd)') and s['font'] == 'Helvetica-Bold')]
    return sorted(candidates, key=lambda s: s['letter'])


def save_formats(document, path, display_width=468):
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(path.with_suffix('.pdf'), garbage=4, deflate=True)
    page = document[0]
    path.with_suffix('.svg').write_text(page.get_svg_image(text_as_path=True), encoding='utf-8')
    scale = max(4.2, 300 * display_width / 72 / page.rect.width)
    page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(path.with_suffix('.png'))


def normalize_fonts(page, font_directory, title_replacements=None, restore_overlapping_titles=False):
    """Replace non-journal fonts, preserving visible text, orientation and span width."""
    originals = spans(page)
    selected = [s for s in originals if not any(x in s['font'] for x in ('Arial', 'Symbol'))
                or (restore_overlapping_titles and re.match(r'^[c-e]\)\s+', s['text']))]
    if not selected:
        return {'replaced_spans': 0, 'visible_text_preserved': True}
    trace_chars = {}
    for trace in page.get_texttrace():
        for char in trace['chars']:
            key = (char[0], round(char[2][0], 2), round(char[2][1], 2))
            trace_chars[key] = trace['seqno']
    white_boxes = []
    active_clips = {}
    for drawing in page.get_drawings(extended=True):
        level = drawing.get('level', 0)
        active_clips = {k: v for k, v in active_clips.items() if k < level}
        if drawing['type'] == 'clip':
            active_clips[level] = fitz.Rect(drawing['scissor'])
            continue
        if (drawing.get('fill') != (1.0, 1.0, 1.0) or drawing.get('fill_opacity') != 1
                or len(drawing.get('items', [])) != 1 or drawing['items'][0][0] not in ('re', 'qu')):
            continue
        rect = fitz.Rect(drawing['rect'])
        for clip in active_clips.values():
            rect &= clip
        if not rect.is_empty:
            white_boxes.append((drawing['seqno'], rect))
    jobs = []
    hidden_count = 0
    for span in selected:
        text = span['text']
        prefix = re.match(r'^[a-i]\)', text)
        if not prefix and 'Bold' in span['font']:
            prefix = re.match(r'^[a-i](?=\s{2,}|$)', text)
        skip = len(prefix.group()) if prefix else 0
        chunks = []
        chunk = []
        for index, char in enumerate(span['chars']):
            box = fitz.Rect(char['bbox'])
            seq = trace_chars.get((ord(char['c']), round(char['origin'][0], 2), round(char['origin'][1], 2)), -1)
            covered = seq >= 0 and any(n > seq and rect.contains(box + (0.1, 0.1, -0.1, -0.1))
                          for n, rect in white_boxes)
            if index < skip or covered:
                if covered:
                    hidden_count += 1
                if chunk:
                    chunks.append(chunk)
                    chunk = []
            else:
                chunk.append(char)
        if chunk:
            chunks.append(chunk)
        style = ('b' if 'Bold' in span['font'] else '') + ('i' if any(x in span['font'] for x in ('Italic', 'Oblique')) else '')
        for chars in chunks:
            jobs.append((span, chars, style))
        page.add_redact_annot(fitz.recover_quad(span['direction'], span), fill=None, cross_out=False)
    page.apply_redactions(images=0, graphics=0, text=0)
    font_files = {'': 'arial.ttf', 'b': 'arialbd.ttf', 'i': 'ariali.ttf', 'bi': 'arialbi.ttf'}
    fonts = {}
    for style in set(j[2] for j in jobs):
        file = font_directory / font_files[style]
        fonts[style] = fitz.Font(fontfile=str(file))
        page.insert_font(fontname='FigureArial' + style, fontfile=str(file))
    written = []
    for span, chars, style in jobs:
        text = ''.join(c['c'] for c in chars)
        for original, replacement in (title_replacements or {}).items():
            text = text.replace(original, replacement)
        font = fonts[style]
        missing = [c for c in text if not c.isspace() and not font.has_glyph(ord(c))]
        if missing:
            assert set(missing) == {'◆'}, (span['font'], missing)
            for char in chars:
                if char['c'] != '◆':
                    page.insert_text(char['origin'], char['c'], fontname='FigureArial' + style,
                                     fontsize=span['size'], color=fitz.sRGB_to_pdf(span['color']))
                    continue
                box = fitz.Rect(char['bbox'])
                size = min(box.width, box.height) * 0.82
                centre = box.tl + (box.br - box.tl) * 0.5
                shape = page.new_shape()
                shape.draw_polyline([centre + (0, -size/2), centre + (size/2, 0),
                                     centre + (0, size/2), centre + (-size/2, 0), centre + (0, -size/2)])
                shape.finish(color=None, fill=fitz.sRGB_to_pdf(span['color']), closePath=True)
                shape.commit()
            written.append(text)
            continue
        point = fitz.Point(chars[0]['origin'])
        last = fitz.Point(chars[-1]['origin'])
        last_quad = fitz.recover_char_quad(span['direction'], span, chars[-1])
        old_width = abs(last - point) + abs(last_quad.ur - last_quad.ul)
        new_width = font.text_length(text, fontsize=span['size'])
        ratio = old_width / new_width if new_width else 1
        angle = -math.degrees(math.atan2(span['direction'][1], span['direction'][0]))
        matrix = fitz.Matrix(ratio, 1) * fitz.Matrix(angle)
        color = fitz.sRGB_to_pdf(span['color'])
        page.insert_text(point, text, fontname='FigureArial' + style, fontsize=span['size'],
                         color=color, morph=(point, matrix), overlay=True)
        written.append(text)
    return {'replaced_spans': len(selected), 'written_runs': len(jobs),
            'covered_characters_not_reintroduced': hidden_count,
            'validation': 'Compare exported text and figures separately before accepting this conversion.',
            'title_case_changes': title_replacements or {},
            'position_rule': 'Original baseline and rotation; replacement fitted to original span advance.'}


def crop(pdf, bounds, target):
    src = fitz.open(pdf)
    clip = fitz.Rect(bounds)
    out = fitz.open()
    page = out.new_page(width=clip.width, height=clip.height)
    page.show_pdf_page(page.rect, src, 0, clip=clip, keep_proportion=False)
    save_formats(out, target)
    out.close()
    src.close()


def sentence_case_figure_one(page, font, font_file):
    # These three headings are outlined paths in the source PDF, not text spans.
    page.insert_font(fontname='NPJv32HeadingFont', fontfile=str(font_file))
    headings = [
        ([78.0, 7.7, 269.9, 16.8], 'Study design — three-cohort parallel analysis',
         (1, 1, 1), (0, 0, 0), 14.5),
        ([42.4, 30.3, 191.1, 39.1], 'Real-world EHR parallel-cohort study',
         (0.216, 0.278, 0.310), (1, 1, 1), 36.5),
        ([202.1, 30.3, 308.0, 39.1], 'Three institutional settings',
         (0.216, 0.278, 0.310), (1, 1, 1), 36.5),
    ]
    for bounds, text, background, color, baseline in headings:
        rect = fitz.Rect(bounds)
        page.draw_rect(rect, color=None, fill=background)
        size = 10
        ratio = (rect.width - 1) / font.text_length(text, fontsize=size)
        origin = fitz.Point(rect.x0 + 0.5, baseline)
        page.insert_text(origin, text, fontname='NPJv32HeadingFont', fontsize=size,
                         color=color, morph=(origin, fitz.Matrix(ratio, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--font-file', type=Path, required=True,
                    help='Path to a locally installed Arial Bold TTF.')
    ap.add_argument('--display-width', type=float, default=468.0)
    ap.add_argument('--letter-size', type=float, default=9.0)
    ap.add_argument('--layout-settings', type=Path)
    ap.add_argument('--start-at')
    ap.add_argument('--only', nargs='+')
    ap.add_argument('--no-crops', action='store_true')
    args = ap.parse_args()
    args.source_dir = args.source_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review = args.output_dir.parent / 'review'
    review.mkdir(exist_ok=True)
    font = fitz.Font(fontfile=str(args.font_file))
    layout_settings = json.loads(args.layout_settings.read_text(encoding='utf-8-sig')) if args.layout_settings else {}
    if isinstance(layout_settings, list):
        layout_settings = {item['figure']: item for item in layout_settings}
    counts_main = [5, 6, 5, 9, 4]
    counts_supp = [6, 6, 6, 2, 6, 5, 4, 2, 6, 2, 9, 9, 7, 2, 4,
                   6, 6, 4, 6, 6, 3, 2, 0, 0, 3, 2, 5, 2, 8]
    figures = [(str(n), f'Figure_{n}', c) for n, c in enumerate(counts_main, 1)]
    figures += [(f'S{n}', f'Supplementary_Figure_S{n}', c)
                for n, c in enumerate(counts_supp, 1)]
    report = {'font': 'Arial Bold', 'nominal_display_width_pt': args.display_width,
              'nominal_display_letter_size_pt': args.letter_size, 'figures': [], 'crops': []}
    report_path = review / 'panel_letter_format_changes.json'
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding='utf-8'))
    enabled = args.start_at is None
    for key, stem, expected in figures:
        if key == args.start_at:
            enabled = True
        if not enabled or (args.only and key not in args.only):
            continue
        source_path = args.source_dir / (stem + '.pdf')
        src = fitz.open(source_path)
        found = labels(src[0], key)
        expected_letters = list('abcdefghi'[:expected])
        assert [s['letter'] for s in found] == expected_letters, (key, [s['text'] for s in found])
        out = fitz.open()
        page = out.new_page(width=src[0].rect.width, height=src[0].rect.height)
        text_source = fitz.open(source_path)
        title_replacements = {
            'Study Design — Three-Cohort Parallel Analysis': 'Study design — three-cohort parallel analysis',
            'Real-World EHR Parallel-Cohort Study': 'Real-world EHR parallel-cohort study',
            'Three Institutional Settings': 'Three institutional settings',
        } if key == '1' else {}
        # S27 local legends slightly overlap the bounding boxes (not the ink) of
        # the three lower titles. Reinsert those titles with the legend text so
        # text-only redaction cannot remove their prefixes.
        font_report = normalize_fonts(text_source[0], args.font_file.parent, title_replacements,
                                      restore_overlapping_titles=key == 'S27')
        page.show_pdf_page(page.rect, text_source, 0)
        display_width = args.display_width
        setting = layout_settings.get(key, {})
        if setting:
            display_width = setting.get('width_inches', setting.get('width_in', display_width / 72)) * 72
        page.insert_font(fontname='NPJv32PanelFont', fontfile=str(args.font_file))
        if key == '1':
            sentence_case_figure_one(page, font, args.font_file)
        size = args.letter_size * page.rect.width / display_width
        changes = []
        for item in found:
            chars = item['chars'][:item['prefix_length']]
            old_box = fitz.Rect(chars[0]['bbox'])
            for char in chars[1:]:
                old_box |= fitz.Rect(char['bbox'])
            cover = old_box + (-0.4, -0.4, 0.4, 0.4)
            cover &= page.rect
            page.draw_rect(cover, color=None, fill=(1, 1, 1), overlay=True)
            label_text = item['letter'] + ')'
            width = font.text_length(label_text, fontsize=size)
            x = min(item['origin'][0], old_box.x1 - width)
            baseline = max(item['origin'][1], 0.5 + font.ascender * size)
            page.insert_text((x, baseline), label_text, fontname='NPJv32PanelFont',
                             fontsize=size, color=(0, 0, 0), overlay=True)
            changes.append({'panel': item['letter'], 'old_visible_prefix': item['text'][:item['prefix_length']],
                            'old_font': item['font'], 'old_size_pt': item['size'],
                            'new_text': label_text, 'new_native_size_pt': size, 'cover_rect': list(cover),
                            'new_origin': [x, baseline]})
        color_replacements = [
            ((0.30196, 0.67451, 0.14902), (0.30454, 0.102, 0.596)),
            ((0.29412, 0.15686, 0.52941), (0.30454, 0.102, 0.596)),
        ] if key == '2' else []
        weighted, line_count = adjust_stroke_widths(out, page.rect.width / display_width, color_replacements)
        save_formats(weighted, args.output_dir / stem, display_width)
        weighted.close()
        report['figures'] = [item for item in report['figures'] if item['figure'] != key]
        report['figures'].append({'figure': key, 'changed': True, 'labels': changes,
                                  'actual_planned_display_width_pt': display_width,
                                  'font_replacement': font_report,
                                  'colour_changes': 'MIMIC centre colours use the Figure 1 purple; matrix colormap unchanged.' if key == '2' else None,
                                  'unfilled_paths_strengthened': line_count,
                                  'filled_marks_changed': False,
                                  'plot_geometry': 'Original PDF page copied at its original dimensions; unfilled line paths retain their exact coordinates.'})
        out.close()
        text_source.close()
        src.close()
        print(f'Finished {key}', flush=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # Rebuild existing current panel crops using unchanged full-figure coordinates.
    if args.no_crops:
        return
    crop_jobs = {}
    source_root = args.source_dir.parent
    asset_map = json.loads((source_root / 'review/response_asset_sources.json').read_text(encoding='utf-8-sig'))
    for key, item in asset_map.items():
        fraction = item.get('crop_fraction')
        source = Path(item['source'])
        if fraction and source.parent != args.source_dir and 'supplement_panels' not in source.parts:
            continue
        if not fraction or not ('panel_' in source.name or 'supplement_panels' in source.parts):
            continue
        figure = item['figure']
        stem = f'Supplementary_Figure_{figure}' if str(figure).startswith('S') else f'Figure_{figure}'
        pdf = args.output_dir / (stem + '.pdf')
        page = fitz.open(pdf)[0]
        bounds = [fraction[0] * page.rect.width, fraction[1] * page.rect.height,
                  fraction[2] * page.rect.width, fraction[3] * page.rect.height]
        rel = source.relative_to(args.source_dir).with_suffix('')
        crop_jobs[str(rel)] = (pdf, bounds)
    mappings = json.loads((source_root / 'review/supplement_panel_bounds.json').read_text(encoding='utf-8-sig'))
    for number, item in mappings.items():
        for key, bounds in item['panels'].items():
            crop_jobs[f'supplement_panels/S{number}{key}'] = (
                args.output_dir / f'Supplementary_Figure_S{number}.pdf', bounds)
    main_map = json.loads((source_root / 'figure_work/main_figure_build_report.json').read_text(encoding='utf-8-sig'))
    for stem, item in main_map.items():
        if not isinstance(item, dict) or 'panel_bounds' not in item:
            continue
        for key, bounds in item['panel_bounds'].items():
            crop_jobs[f'{stem}_panel_{key}'] = (args.output_dir / (stem + '.pdf'), bounds)
    for key, bounds in {'c': [0, 205, 166.1, 421], 'd': [166.1, 205, 332.2, 421],
                        'e': [332.2, 205, 510, 421]}.items():
        crop_jobs[f'Figure_2_panel_{key}'] = (args.output_dir / 'Figure_2.pdf', bounds)
    for relative, (pdf, bounds) in sorted(crop_jobs.items()):
        crop(pdf, bounds, args.output_dir / relative)
        report['crops'].append({'path': relative, 'source': pdf.name, 'bounds': bounds})
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Updated {sum(x["changed"] for x in report["figures"])} figures and {len(crop_jobs)} panel crops.')


if __name__ == '__main__':
    main()
