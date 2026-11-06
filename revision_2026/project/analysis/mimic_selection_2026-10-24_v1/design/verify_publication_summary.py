"""Check manuscript-facing summary against independently verified CSV tables."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def main():
    dest = ROOT / 'results/publication_summary_verification.json'
    assert not dest.exists()
    metrics = ROOT / 'results/metrics'
    summary_path = metrics / 'summary.json'
    numeric_path = ROOT / 'results/independent_results_verification.json'
    numeric_identity = identity(numeric_path)
    numeric = _release_resolve_config(json.loads(numeric_path.read_text(encoding='utf-8')))
    summary = _release_resolve_config(json.loads(summary_path.read_text(encoding='utf-8')))
    assert numeric['passed']
    identities = {'summary': identity(summary_path), 'independent_numeric_verification': numeric_identity}
    checks = []
    max_error = 0.0
    specifications = [('own_cohort_metrics', 'own_cohort_performance.csv', ('target', 'model', 'metric')), ('all_common_target_metrics', 'all_performance.csv', ('target', 'model', 'metric')), ('paired_contrasts', 'paired_contrasts.csv', ('target', 'first', 'second', 'metric'))]
    for field, filename, keys in specifications:
        csv_path = metrics / filename
        identities[filename] = identity(csv_path)
        records = pd.read_csv(csv_path).to_dict('records')
        lookup = {tuple((row[k] for k in keys)): row for row in records}
        summary_lookup = {tuple((row[k] for k in keys)): row for row in summary[field]}
        assert len(lookup) == len(records) == len(summary[field]) == len(summary_lookup)
        assert set(lookup) == set(summary_lookup)
        cells = 0
        for key, row in summary_lookup.items():
            other = lookup[key]
            assert set(row) == set(other)
            for column, value in row.items():
                if isinstance(value, bool) or isinstance(value, str):
                    assert value == other[column]
                else:
                    error = abs(float(value) - float(other[column]))
                    max_error = max(max_error, error)
                    assert error <= 1e-14
                cells += 1
        checks.append({'summary_field': field, 'csv_file': filename, 'rows': len(records), 'cells_compared': cells, 'passed': True})
    assert summary['source'] == identity(ROOT / 'private/run_v1/predictions.npz') == numeric['predictions']
    assert summary['protocol'] == identity(ROOT / 'protocol.json') == numeric['protocol']
    protocol = _release_resolve_config(json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8')))
    by_target = {entry['target']: entry for entry in summary['bootstrap']}
    numeric_targets = {entry['target']: entry for entry in numeric['bootstrap_targets']}
    assert len(by_target) == 4 and set(by_target) == set(protocol['arms'])
    target_checks = []
    for index, target in enumerate(protocol['arms']):
        entry = by_target[target]
        assert entry['draws'] == entry['valid'] == numeric_targets[target]['valid_draws'] == 2000
        assert entry['invalid'] == 0 and entry['seed'] == protocol['bootstrap_seed_base'] + index
        archive_path = metrics / ('bootstrap_' + target + '.npz')
        with np.load(archive_path, allow_pickle=False) as z:
            assert len(z['valid']) == 2000 and z['valid'].all() and np.isfinite(z['draws']).all()
        identities[archive_path.name] = identity(archive_path)
        target_checks.append({'target': target, 'draws': 2000, 'valid': 2000, 'invalid': 0, 'passed': True})
    for value in identities.values():
        assert identity(value['path']) == value
    report = {'passed': True, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'script': identity(__file__), 'identities': identities, 'table_checks': checks, 'all_four_targets_have_exactly_2000_valid_draws': True, 'bootstrap_checks': target_checks, 'maximum_summary_csv_numeric_difference': max_error, 'csv_roundtrip_tolerance': 1e-14, 'independent_numeric_verification_unchanged': True, 'scope': 'Manuscript-facing summary fields and the three CSV tables agree; their numerical values were independently checked in the linked immutable report. No new model fitting or changed scientific interpretation.'}
    with dest.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True, allow_nan=False)
    print(json.dumps({'passed': True, 'tables': len(checks), 'total_cells': sum((c['cells_compared'] for c in checks)), 'all_four_targets_2000_valid': True, 'maximum_numeric_difference': max_error, 'output_identity': identity(dest)}, indent=2))
if __name__ == '__main__':
    main()
