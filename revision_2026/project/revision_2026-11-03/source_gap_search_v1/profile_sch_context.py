"""Aggregate availability of clinical context fields, without exporting clinical rows."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
from collections import Counter
import csv, io, json, re, struct, sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'revision_2026-10-12/analysis/a2_sch_archive_unit_dependencies_v1'))
import xlrd
SOURCE = Path(_release_path('njm', 'SCH'))
exports = _release_resolve_config(json.loads((HERE / 'sch_exports.json').read_text(encoding='utf-8')))
pattern = re.compile('分期|离院方式|治疗结果|^病理诊断(?:编码|名称)?$|^病理编码$|手术.*日期|主手术时间|主要手术日期')
results = []
for source in exports:
    raw = (SOURCE / source['source']).read_bytes()
    counts = Counter()
    deaths = Counter()

    def add(header, value):
        if pattern.search(header) and value is not None and str(value).strip():
            counts[header] += 1
            if '离院方式' in header or '治疗结果' in header:
                if str(value).strip() == '死亡':
                    deaths[header] += 1
    if source['format'] == 'BIFF2':
        headers = {}
        position = 0
        while position + 4 <= len(raw):
            code, size = struct.unpack_from('<HH', raw, position)
            body = raw[position + 4:position + 4 + size]
            position += 4 + size
            if code not in [1, 2, 3, 4, 5, 6] or size < 4:
                continue
            row, column = struct.unpack_from('<HH', body)
            value = None
            if code == 4 and size >= 8 and (size == 8 + body[7]):
                value = body[8:].decode('gb18030')
            elif code == 2 and size >= 9:
                value = struct.unpack_from('<H', body, 7)[0]
            elif code == 3 and size >= 15:
                value = struct.unpack_from('<d', body, 7)[0]
            if row == 0 and isinstance(value, str):
                headers[column] = value
            elif row > 0:
                add(headers.get(column, ''), value)
    else:
        book = None
        if source['format'] == 'OLE XLS':
            book = xlrd.open_workbook(file_contents=raw, encoding_override='gb18030', on_demand=True, logfile=io.StringIO())
            sheet = book.sheet_by_index(source['sheet'] - 1)
            headers = sheet.row_values(0)
            rows = (sheet.row_values(i) for i in range(1, sheet.nrows))
        else:
            rows = csv.reader(io.StringIO(raw.decode('gb18030')), delimiter='\t')
            headers = next(rows)
        selected = [(i, h) for i, h in enumerate(headers) if pattern.search(h)]
        for row in rows:
            for i, h in selected:
                if i < len(row):
                    add(h, row[i])
        if book:
            book.release_resources()
    results.append({'source': source['source'], 'source_rows': source['rows'], 'nonempty_field_occurrences': dict(counts), 'literal_death_occurrences': dict(deaths)})
(HERE / 'sch_clinical_context.json').write_text(json.dumps({'scope': 'Source-row occurrences, not unique patients or events. No linkage or clinical label inference performed. Nonempty staging fields have not been adjudicated for meaningful clinical values.', 'exports': results}, ensure_ascii=False, indent=2), encoding='utf-8')
total = Counter()
death_total = Counter()
for r in results:
    total.update(r['nonempty_field_occurrences'])
    death_total.update(r['literal_death_occurrences'])
print(json.dumps({'exports': len(results), 'staging_nonempty_field_occurrences': {k: v for k, v in total.items() if '分期' in k}, 'death_literal_field_occurrences': dict(death_total)}, ensure_ascii=False))
