"""Separate report-header timestamps from report-detail dates in original-cohort sources."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
from collections import Counter
import json
import pandas as pd
from scan_sources import records, key, link, datekind, ASSAYS
P = Path(__file__).resolve().parent
members = {key(v) for v in pd.read_csv(_release_path('china', 'WHU_Cancer_processed_data.csv'), usecols=['病案号'], dtype='string')['病案号'].dropna()}
all_paths = pd.read_csv(_release_path('project', 'clinical_review_2026-10-27/full_index_v1/private/whu_sources.csv'), usecols=['source_path'])['source_path'].unique()
paths = [p for p in all_paths if Path(p).suffix.lower() == '.json']
reports = Counter()
items = Counter()
contexts = Counter()
record_count = 0
for path in paths:
    obj = _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))
    if isinstance(obj, str):
        obj = _release_resolve_config(json.loads(obj))
    for visit in records(obj):
        pid, _ = link(visit)
        if pid not in members:
            continue
        record_count += 1
        for report in records(visit.get('检验列表')):
            outer = report.get('报告时间')
            reports['outer_' + datekind(outer)] += 1
            for detail in records(report.get('检验详情')):
                inner = detail.get('报告时间')
                reports['inner_' + datekind(inner)] += 1
                state = 'outer_' + datekind(outer) + '__inner_' + datekind(inner)
                reports[state] += 1
                if datekind(outer) == 'datetime' and datekind(inner) == 'date':
                    reports['outer_clock_inner_date_same_calendar_day' if outer[:10] == inner else 'outer_clock_inner_date_different_calendar_day'] += 1
                for item in records(detail.get('报告子项')):
                    name = str(item.get('项目名称', '')).strip()
                    if any((name in a for a in ASSAYS.values())):
                        items[state] += 1
                    if name in ASSAYS['albumin'] and str(item.get('单位', '')).strip() == '%':
                        contexts[str(report.get('检验名称', '<missing>'))] += 1
result = {'source_file_count': len(paths), 'original_cohort_source_records': record_count, 'report_level_counts': dict(reports), 'four_assay_item_timestamp_pairs': dict(items), 'albumin_percentage_report_names': dict(contexts), 'scope': 'Uses the original-cohort source-path index, rechecks each record patient key, and reports aggregate field precision; does not resolve report revisions or event onset.'}
(P / 'whu_report_time_precision.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
