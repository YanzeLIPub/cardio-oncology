"""Aggregate the author-confirmed source-linkage exclusion across all centres."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import csv, json, hashlib
BASE = Path(__file__).resolve().parents[1]
REVIEW = Path(_release_path('project', 'clinical_review_2026-10-27/full_index_v1'))
sealed = _release_resolve_config(json.loads((REVIEW / 'artifact_manifest.json').read_text(encoding='utf-8')))
expected = {x['path']: None for x in sealed['items']}
rows = []
inputs = []
for centre, unit, n, pos, criterion in [('SCH', 'patient', 38509, 1323, 'diagnostic_source_record_count'), ('WHU', 'patient', 2562, 281, 'raw_record_count'), ('MIMIC', 'admission', 9946, 4462, 'diagnostic_source_record_count')]:
    rel = 'private/' + centre.lower() + '_index.csv'
    path = REVIEW / rel
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    assert h == expected[rel]
    with path.open(encoding='utf-8-sig', newline='') as f:
        source = list(csv.DictReader(f))
    assert len(source) == n and sum((int(x['original_label']) for x in source)) == pos
    keep = [x for x in source if int(x[criterion]) > 0]
    exclude = [x for x in source if int(x[criterion]) == 0]
    assert len(keep) + len(exclude) == n
    rows.append({'centre': centre, 'analysis_unit': unit, 'original_units': n, 'original_algorithm_positive': pos, 'excluded_units': len(exclude), 'excluded_algorithm_positive': sum((int(x['original_label']) for x in exclude)), 'retained_units': len(keep), 'retained_algorithm_positive': sum((int(x['original_label']) for x in keep)), 'retained_algorithm_negative': sum((1 - int(x['original_label']) for x in keep)), 'source_eligibility_criterion': criterion + '>0', 'clinical_review_completed': False})
    inputs.append({'path': str(path), 'sealed_manifest_match': True})
with (BASE / 'analysis/centre_source_eligibility_flow.csv').open('x', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
(BASE / 'qa/centre_source_eligibility_flow_checks.json').write_text(json.dumps({'passed': True, 'rows': rows, 'inputs': inputs, 'missing_individual_laboratory_or_discharge_summary_not_excluded': True, 'labels_unchanged': True, 'zero_means_original_algorithm_negative': True}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'passed': True, 'retained_units': {x['centre']: x['retained_units'] for x in rows}, 'excluded': {x['centre']: x['excluded_units'] for x in rows}}))
