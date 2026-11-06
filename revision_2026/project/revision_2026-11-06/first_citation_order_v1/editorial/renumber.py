from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import re, json, shutil
R = Path(__file__).resolve().parents[1]
ROOT = R.parents[1]
P = ROOT / 'revision_2026-11-06/panel_streamline_v1'
FIG_ORDER = [4, 1, 15, 8, 16, 17, 9, 11, 12, 20, 23, 21, 24, 18, 19, 13, 29, 14, 26, 27, 2, 7, 3, 10, 5, 6, 22, 25, 28]
TABLE_ORDER = [5, 8, 2, 3, 4, 1, 6, 7, 9, 10]
FM = {n: i + 1 for i, n in enumerate(FIG_ORDER)}
TM = {n: i + 1 for i, n in enumerate(TABLE_ORDER)}
(R / 'figures').mkdir(parents=True, exist_ok=True)
(R / 'review').mkdir(parents=True, exist_ok=True)
for p in (P / 'figures').iterdir():
    if p.suffix not in ['.pdf', '.svg', '.png']:
        continue
    name = re.sub('Supplementary_Figure_S(\\d+)', lambda m: 'Supplementary_Figure_S' + str(FM[int(m[1])]), p.name)
    shutil.copy2(p, R / 'figures' / name)
(R / 'review/numbering_map.json').write_text(json.dumps({'figures': FM, 'tables': TM, 'figure_order_old_numbers': FIG_ORDER, 'table_order_old_numbers': TABLE_ORDER}, indent=2), encoding='utf-8')
print(json.dumps({'figure_mapping': FM, 'table_mapping': TM}))
