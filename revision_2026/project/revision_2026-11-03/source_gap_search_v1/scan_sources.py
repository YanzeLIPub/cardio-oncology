"""Read source files and save aggregate metadata only; original data are not edited."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import csv, io, json, re, struct, sys, zipfile, platform, argparse, time
ROOT = Path(_release_path('project', ''))
OUT = Path(__file__).resolve().parent
RAW = Path(_release_path('njm', ''))
sys.stdout.reconfigure(encoding='utf-8')

def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def records(obj):
    if isinstance(obj, str) and obj.lstrip().startswith(('[', '{')):
        try:
            obj = _release_resolve_config(json.loads(obj))
        except ValueError:
            return []
    return [obj] if isinstance(obj, dict) else [v for v in obj if isinstance(v, dict)] if isinstance(obj, list) else []

def key(v):
    if v is None:
        return ''
    s = re.sub('\\.0+$', '', str(v).strip())
    return s.lstrip('0') or '0' if s.isdigit() else s

def nonempty(v):
    return v is not None and v != '' and (v != []) and (v != {})

def datekind(v):
    if not isinstance(v, str):
        return 'not_string'
    if re.fullmatch('\\d{4}[-/]\\d{1,2}[-/]\\d{1,2}(?:[ T]\\d{1,2}:\\d{2}(?::\\d{2}(?:\\.\\d+)?)?)?', v.strip()):
        return 'datetime' if ':' in v else 'date'
    return 'other'

def link(visit):
    vals = defaultdict(list)
    for block in records(visit.get('电子病历菜单及详情')):
        for section, payload in block.items():
            if section in ['个人基本信息', '住院基本信息']:
                for item in records(payload):
                    for k in ['病案号', '入院日期', '出院日期']:
                        if nonempty(item.get(k)):
                            vals[k].append(str(item[k]))
    ids = {key(v) for v in vals['病案号']}
    return (next(iter(ids)) if len(ids) == 1 else '', vals)
PAT = {'onset': re.compile('发病(?:时间|日期|时刻)|发生(?:时间|日期|时刻)|起病(?:时间|日期|时刻)|onset|event.?time', re.I), 'diagnosis_time': re.compile('诊断.*(?:时间|日期)|确诊.*(?:日期|时间)|diagnos.*(?:time|date)', re.I), 'release_time': re.compile('(?:发布|释放|签发)(?:时间|日期)|结果(?:可见|可用|返回)(?:时间|日期)|release.?time|available.?time', re.I), 'report_time': re.compile('(?:报告|审核|检验|采样|收样|检查)(?:时间|日期)', re.I), 'poa': re.compile('入院病情|present.?on.?admission|^poa$', re.I), 'unit_specimen': re.compile('^单位$|标本|^unit$|specimen|检验项目编码|^项目代码$', re.I)}
ASSAYS = {'albumin': {'白蛋白', '★白蛋白', 'ALB', 'Albumin'}, 'total_protein': {'总蛋白', '★总蛋白', 'TP', 'Total Protein'}, 'hemoglobin': {'血红蛋白', '★血红蛋白', 'HGB', 'Hb', 'Hemoglobin'}, 'hematocrit': {'红细胞压积', '★红细胞压积', '红细胞比容', '血细胞比容', 'HCT', 'Hematocrit'}}

def context(k, old):
    if old == 'infection' or re.search('感染病例登记|院内感染', k):
        return 'infection'
    if re.search('检验|化验|报告子项', k):
        return 'laboratory'
    if re.search('病案首页|诊断', k):
        return 'diagnosis'
    if re.search('检查', k):
        return 'examination'
    if re.search('入院记录|出院记录|病程', k):
        return 'clinical_note'
    return old

def whu():
    import pandas as pd
    pop = pd.read_csv(_release_path('china', 'WHU_Cancer_processed_data.csv'), usecols=['病案号'], dtype='string')
    members = {key(x) for x in pop['病案号'].dropna()}
    counts = Counter()
    fields = defaultdict(Counter)
    found = defaultdict(set)
    linked = set()
    schema = defaultdict(Counter)
    labs = defaultdict(Counter)
    labpeople = defaultdict(set)
    files = sorted((RAW / 'WUH').glob('*.json'))
    begin = time.time()
    for fi, p in enumerate(files):
        try:
            obj = _release_resolve_config(json.loads(p.read_text(encoding='utf-8-sig')))
            if isinstance(obj, str):
                obj = _release_resolve_config(json.loads(obj))
        except (ValueError, UnicodeError):
            counts['unparsed_files'] += 1
            continue
        counts['parsed_files'] += 1
        file_member = False
        for visit in records(obj):
            counts['source_records'] += 1
            pid, linkfields = link(visit)
            matched = pid in members
            if matched:
                counts['cohort_records'] += 1
                linked.add(pid)
                file_member = True
            scopes = ['all_sources'] + (['original_cancer_cohort'] if matched else [])
            for section in ['检验列表', '诊断', '电子病历菜单及详情', '检查列表']:
                if nonempty(visit.get(section)):
                    for scope in scopes:
                        counts[scope + '__records_with_' + section] += 1
            if not matched:
                continue
            scopes = ['original_cancer_cohort']
            for report in records(visit.get('检验列表')):
                schema['report'].update(report.keys())
                for detail in records(report.get('检验详情')):
                    schema['detail'].update(detail.keys())
                    for item in records(detail.get('报告子项')):
                        schema['item'].update(item.keys())
                        name = str(item.get('项目名称', '')).strip()
                        for assay, aliases in ASSAYS.items():
                            if name not in aliases:
                                continue
                            unit = str(item.get('单位', '')).strip() or '<missing>'
                            if len(unit) > 25 or re.search('\\d{6}', unit):
                                unit = '<other>'
                            numeric = bool(re.fullmatch('[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?', str(item.get('检查结果', '')).strip()))
                            for scope in scopes:
                                rec = labs[scope, assay, unit]
                                rec['items'] += 1
                                rec['numeric_results'] += numeric
                                rec['report_timestamp_items'] += int(datekind(detail.get('报告时间', report.get('报告时间'))) in ['date', 'datetime'])
                                rec['report_datetime_items'] += int(datekind(detail.get('报告时间', report.get('报告时间'))) == 'datetime')
                                if matched:
                                    labpeople[scope, assay, unit].add(pid)
            stack = [(visit, 'record')]
            while stack:
                node, role = stack.pop()
                if isinstance(node, dict):
                    for k, v in node.items():
                        if not isinstance(k, str):
                            continue
                        new = context(k, role)
                        for family, pat in PAT.items():
                            if len(k) < 70 and pat.search(k):
                                for scope in scopes:
                                    ident = (scope, family, new, k)
                                    row = fields[ident]
                                    row['occurrences'] += 1
                                    row['nonempty'] += int(nonempty(v))
                                    if nonempty(v):
                                        row[datekind(v)] += 1
                                    found[ident].add(fi)
                        if isinstance(v, (dict, list)):
                            stack.append((v, new))
                        elif isinstance(v, str) and v.lstrip().startswith(('[', '{')):
                            try:
                                decoded = _release_resolve_config(json.loads(v))
                            except ValueError:
                                continue
                            if isinstance(decoded, (dict, list)):
                                stack.append((decoded, new))
                                counts['decoded_nested_json'] += 1
                elif isinstance(node, list):
                    stack.extend(((x, role) for x in node if isinstance(x, (dict, list))))
        counts['files_linked_to_original_cohort'] += file_member
        if (fi + 1) % 1000 == 0:
            print(json.dumps({'centre': 'WHU', 'files_scanned': fi + 1, 'cohort_records': counts['cohort_records'], 'elapsed_seconds': round(time.time() - begin)}, ensure_ascii=False), flush=True)
    fieldrows = [dict(scope=k[0], family=k[1], context=k[2], field=k[3], files=len(found[k]), **v) for k, v in sorted(fields.items())]
    labrows = [dict(scope=k[0], assay=k[1], unit=k[2], cohort_patients=len(labpeople[k]), **v) for k, v in sorted(labs.items())]
    save('whu_fields.json', fieldrows)
    save('whu_common_laboratories.json', labrows)
    save('whu_summary.json', {'source': _release_path('njm', 'WUH'), 'original_cohort_patients': len(members), 'linked_original_patients': len(linked), 'counts': dict(counts), 'laboratory_schema': {k: dict(v) for k, v in schema.items()}, 'elapsed_seconds': round(time.time() - begin), 'scope': 'All JSON files parsed for source inventory and original-cohort linkage; detailed field and assay inspection limited to records linked to the original cancer cohort. Record dates are not assumed to be event onset.'})
    print(json.dumps({'centre': 'WHU', 'complete': True, 'files': len(files), 'linked_original_patients': len(linked), 'cohort_records': counts['cohort_records']}, ensure_ascii=False), flush=True)

def sch():
    sys.path.insert(0, str(ROOT / 'revision_2026-10-12/analysis/a2_sch_archive_unit_dependencies_v1'))
    import xlrd
    directory = RAW / 'SCH'
    exports = sorted(directory.rglob('*.xls'))
    rows = []
    binary_cache = {}
    seen = []
    duplicates = []
    poa = Counter()
    relevant = re.compile('单位|标本|检验|化验|报告|采样|审核|确诊|发病|发生时间|入院病情|入院时间|出院时间|住院次数|白蛋白|总蛋白|血红蛋白|压积', re.I)

    def table_info(headers, iterator, path, fmt, sheet):
        headers = [str(x) for x in headers]
        nonnull = Counter()
        category = Counter()
        n = 0
        columns = [i for i, h in enumerate(headers) if relevant.search(h)]
        for row in iterator:
            n += 1
            for i in columns:
                if i < len(row) and nonempty(row[i]):
                    nonnull[headers[i]] += 1
                    if '入院病情' in headers[i]:
                        v = str(row[i]).strip()
                        if v in ['有', '无', '不明', '情况不明', '临床未确定', '1', '2', '3', '4', '1.0', '2.0', '3.0', '4.0']:
                            category[v] += 1
                        else:
                            category['other'] += 1
        return {'source': path, 'format': fmt, 'sheet': sheet, 'rows': n, 'columns': len(headers), 'headers': headers, 'relevant_nonempty_fields': dict(nonnull), 'poa_category_counts': dict(category)}
    for p in exports:
        payload = p.read_bytes()
        rel = p.relative_to(directory).as_posix()
        same = None
        for oldpath, size in seen:
            if size == len(payload) and oldpath.read_bytes() == payload:
                same = oldpath
                break
        if same is not None:
            duplicates.append({'copy': rel, 'same_as': same.relative_to(directory).as_posix()})
            continue
        seen.append((p, len(payload)))
        if payload.startswith(b'\t\x00'):
            headers = {}
            values = {}
            n = 0
            pos = 0
            maxrow = -1
            nonemptyfields = Counter()
            cats = Counter()
            assaystrings = Counter()
            unitcells = 0
            while pos + 4 <= len(payload):
                code, size = struct.unpack_from('<HH', payload, pos)
                body = payload[pos + 4:pos + 4 + size]
                pos += size + 4
                if code not in [1, 2, 3, 4, 5, 6] or size < 4:
                    continue
                ri, ci = struct.unpack_from('<HH', body)
                maxrow = max(maxrow, ri)
                val = None
                if code == 4 and size >= 8 and (size == 8 + body[7]):
                    val = body[8:].decode('gb18030')
                elif code == 2 and size >= 9:
                    val = struct.unpack_from('<H', body, 7)[0]
                elif code == 3 and size >= 15:
                    val = struct.unpack_from('<d', body, 7)[0]
                if ri == 0 and isinstance(val, str):
                    headers[ci] = val
                elif ri > 0 and nonempty(val):
                    h = headers.get(ci, '')
                    if relevant.search(h):
                        nonemptyfields[h] += 1
                    if '入院病情' in h:
                        v = str(val).strip()
                        cats[v if v in ['有', '无', '不明', '情况不明', '临床未确定', '1', '2', '3', '4', '1.0', '2.0', '3.0', '4.0'] else 'other'] += 1
            rows.append({'source': rel, 'format': 'BIFF2', 'sheet': 1, 'rows': maxrow, 'columns': len(headers), 'headers': [headers[i] for i in sorted(headers)], 'relevant_nonempty_fields': dict(nonemptyfields), 'poa_category_counts': dict(cats), 'all_records_parsed': pos == len(payload)})
        elif payload.startswith(b'\xd0\xcf\x11\xe0'):
            book = xlrd.open_workbook(file_contents=payload, encoding_override='gb18030', on_demand=True, logfile=io.StringIO())
            for si in range(book.nsheets):
                sheet = book.sheet_by_index(si)
                rows.append(table_info(sheet.row_values(0), (sheet.row_values(i) for i in range(1, sheet.nrows)), rel, 'OLE XLS', si + 1))
            book.release_resources()
        else:
            reader = csv.reader(io.StringIO(payload.decode('gb18030')), delimiter='\t')
            rows.append(table_info(next(reader), reader, rel, 'tab-delimited text', 1))
        print(json.dumps({'centre': 'SCH', 'export': rel, 'checked': True}, ensure_ascii=False), flush=True)
    archive = []
    for p in sorted(directory.glob('*.zip')):
        with zipfile.ZipFile(p) as z:
            for member in z.infolist():
                if member.is_dir():
                    continue
                data = z.read(member)
                matches = [q.relative_to(directory).as_posix() for q in exports if q.stat().st_size == len(data) and q.read_bytes() == data]
                archive.append({'archive': p.name, 'member': member.filename, 'bytes': len(data), 'same_as_extracted': matches})
    csvrows = []
    for p in sorted(directory.glob('*.csv')):
        for enc in ['utf-8-sig', 'gb18030']:
            try:
                with p.open(encoding=enc, newline='') as f:
                    header = next(csv.reader(f))
                break
            except UnicodeError:
                continue
        csvrows.append({'source': p.name, 'bytes': p.stat().st_size, 'columns': len(header), 'headers': header, 'relevant_fields': [h for h in header if relevant.search(h)]})
    save('sch_exports.json', rows)
    save('sch_csv_headers.json', csvrows)
    save('sch_archive_contents.json', archive)
    save('sch_summary.json', {'source': directory.as_posix(), 'xls_copies': len(exports), 'distinct_export_files': len(seen), 'duplicate_copies': duplicates, 'zip_files': len(list(directory.glob('*.zip'))), 'archive_members': len(archive), 'all_archive_members_equal_extracted_files': all((x['same_as_extracted'] for x in archive)), 'export_rows_not_unique_patients': sum((r['rows'] for r in rows)), 'reader_version': xlrd.__version__, 'lab_unit_fields': [{'source': r['source'], 'fields': [h for h in r['headers'] if re.search('单位|标本|检验|化验|报告|采样|审核|白蛋白|总蛋白|血红蛋白|压积', h)]} for r in rows]})
    print(json.dumps({'centre': 'SCH', 'complete': True, 'distinct_export_files': len(seen), 'archive_members': len(archive)}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('centre', choices=['whu', 'sch'])
    args = a.parse_args()
    save(args.centre + '_execution_environment.json', {'date': '2026-11-03', 'python': platform.python_version(), 'executable': sys.executable, 'source_directories': [_release_path('njm', 'WUH'), _release_path('njm', 'SCH')], 'output_directory': OUT.as_posix(), 'patient_level_outputs': False, 'source_modifications': False})
    globals()[args.centre]()
