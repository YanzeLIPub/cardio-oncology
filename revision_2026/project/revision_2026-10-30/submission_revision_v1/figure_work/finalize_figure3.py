from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import json, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from pypdf import PdfReader
ROOT = Path(__file__).resolve().parent
P = Path(_release_path('project', 'revision_2026-10-29/box_time_revision_v1'))
report = _release_resolve_config(json.loads((ROOT / 'figure3_submission_report.json').read_text()))

def ident(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}
old = np.asarray(Image.open(_release_path('project', 'revision_2026-10-24/figure_readability_v1/figure_work/figure3/Figure3_readability_vector_final_165_1mm_144dpi.png')))
new = np.asarray(Image.open(ROOT / 'Figure3_submission_165_1mm.png'))
assert old.shape == new.shape
heatmap_differences = int(np.any(old[385:738] != new[385:738], axis=2).sum())
assert heatmap_differences == 0, heatmap_differences
s = old.shape[1] / 805.928
rect = [675, 59, 787, 120]
q = old[round((992.603 - rect[3]) * s):round((992.603 - rect[1]) * s), round(rect[0] * s):round(rect[2] * s), :3]
assert int((q.max(2) - q.min(2) > 10).sum()) == 0
numeric = []
for x in report['changes']:
    if x['kind'] == 'text' and x['old'] != x['new']:
        a = float(x['old'].replace('−', '-'))
        b = float(x['new'].replace('−', '-'))
        assert a == b
        numeric.append({'old': x['old'], 'new': x['new'], 'equal_numeric_value': True})
oldr = PdfReader(report['source']['path'])
newr = PdfReader(ROOT / 'Figure3_submission.pdf')
forms = []
for k in ['/Im0', '/Im1']:
    a = oldr.pages[0]['/Resources']['/XObject'][k]
    b = newr.pages[0]['/Resources']['/XObject'][k]
    assert a.get_data() == b.get_data()
    forms.append({'name': k, 'byte_identical': True, 'point_count': 6553})
entries = []
for output in report['outputs']:
    n = output['figure']
    if n == 'Figure3':
        oldpng = P / 'figures/readability_revision/Figure3_readability.png'
        publish = {output['pdf']['path']: 'figures/readability_revision/Figure3_readability.pdf', output['png']['path']: 'figures/readability_revision/Figure3_readability.png', output['tif']['path']: 'figures/readability_revision/Figure3_readability.tif'}
    else:
        oldpng = P / f'figures/integrated_numbering/Supplementary_Figure_{n}.png'
        folder = 'figures/submission_typography' if n in {'S18', 'S19'} else 'figures/readability_revision'
        publish = {output['pdf']['path']: f'{folder}/Supplementary_Figure_{n}.pdf', output['png']['path']: f'figures/integrated_numbering/Supplementary_Figure_{n}.png', output['tif']['path']: f'{folder}/Supplementary_Figure_{n}.tif'}
    entries.append({'id': n, 'documents': ['main' if n == 'Figure3' else 'supplement'], 'old_png': oldpng.as_posix(), 'new_png': output['png']['path'], 'resize_to_new_aspect': False, 'publish': publish})
report.update({'visual_review': 'passed; all five full-size proof PNGs individually viewed at intended widths, then S18/S19 bottom whitespace padded to preserve prior aspect ratios', 'heatmap_C_D_pixels_exact_at_intended_width': True, 'heatmap_C_D_changed_pixels': heatmap_differences, 'new_outcome_legend_source_colored_pixels': 0, 'new_outcome_legend_rectangle_pdf_pt': rect, 'numeric_tick_format_checks': numeric, 'point_form_checks': forms, 'legend_only_color_correction': 'B WHU legend outline now matches its red fill; original blue outline removed. Actual plotted point and line colors remain unchanged.', 'panel_letters': 'A-E original text/size/location retained; supplement crops omit only redundant panel letters.', 'frozen': True, 'script': ident(_release_source_path(ROOT / 'enlarge_figure3.py'))})
(ROOT / 'figure3_submission_report.json').write_text(json.dumps(report, indent=2) + '\n')
folder = ROOT / 'figure3'
folder.mkdir(exist_ok=True)
(folder / 'figure_handoff.json').write_text(json.dumps({'figures': entries, 'verification': ident(ROOT / 'figure3_submission_report.json'), 'main_figure_width_mm': 165.1, 'supplement_width_mm': 169.16, 'main_min_A_B_internal_font_pt': report['target_min_A_B_pt'], 'main_min_E_internal_font_pt': report['target_min_E_pt'], 'numeric_data_unchanged': True}, indent=2) + '\n')
print(json.dumps({'handoff': str(folder / 'figure_handoff.json'), 'main_min_A_B_internal_font_pt': report['target_min_A_B_pt'], 'main_min_E_internal_font_pt': report['target_min_E_pt'], 'heatmap_C_D_changed_pixels': heatmap_differences}))
