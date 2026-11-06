from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import json, csv
R = Path(__file__).resolve().parents[1]
source = _release_resolve_config(json.loads((R / 'review/source_content.json').read_text(encoding='utf-8')))
for n, name in [(15, 'model_validation/selected_rounds.csv'), (18, 'calibration/joint_parameters.csv'), (41, 'high_coverage/results/retained_lab_counts.csv')]:
    output = R / 'aggregate_csv_exports' / name
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8', newline='') as stream:
        csv.writer(stream).writerows(source['tables'][n - 1]['rows'])
print('Exported three retained aggregate tables.')
