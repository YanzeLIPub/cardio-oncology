"""Read the eight shared-assay candidate columns; save aggregate format counts only."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
from collections import Counter, defaultdict
import json, re
import pandas as pd
P = Path(__file__).resolve().parent
source = Path(_release_path('njm', 'SCH/SCH_processed_data.csv'))
headers = _release_resolve_config(json.loads((P / 'sch_csv_headers.json').read_text(encoding='utf-8')))
header = next((x['headers'] for x in headers if x['source'] == source.name))
columns = [h for h in header if h.strip('★') in ['白蛋白', '总蛋白', '血红蛋白', '红细胞压积']]
counts = defaultdict(Counter)
rows = 0
number = '[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?'
units = re.compile('(?:mg/dL|g/dL|mg/L|g/L|mmol/L|umol/L|μmol/L|µmol/L|L/L|%)', re.I)
for chunk in pd.read_csv(source, usecols=columns, dtype='string', chunksize=50000):
    rows += len(chunk)
    for column in columns:
        series = chunk[column].dropna().str.strip()
        series = series[series != '']
        row = counts[column]
        row['nonempty'] += len(series)
        row['numeric'] += int(series.str.fullmatch(number).sum())
        row['numeric_with_comparison_symbol'] += int(series.str.fullmatch('[<>≤≥]\\s*' + number).sum())
        row['contains_recognized_unit_text'] += int(series.str.contains(units).sum())
result = {'source': source.as_posix(), 'rows': rows, 'selected_columns': columns, 'value_format_counts': dict(counts), 'scope': 'Full source CSV row pass limited to eight candidate assay columns; no patient data exported; no unit inferred from numeric range.'}
(P / 'sch_laboratory_value_formats.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
