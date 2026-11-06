"""Bind the independent preparation and pre-fit review to the train gate."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def main():
    destination = HERE / 'preparation_verification.json'
    assert not destination.exists(), 'Never overwrite a sealed gate'
    independent_path = HERE / 'independent_preparation_verification.json'
    review_path = HERE / 'prefit_code_review.json'
    independent = _release_resolve_config(json.loads(independent_path.read_text(encoding='utf-8')))
    review = _release_resolve_config(json.loads(review_path.read_text(encoding='utf-8')))
    assert independent['passed'] and review['approved'] and (not review['blocking_findings'])
    matrix = identity(independent['matrix_identity']['path'])
    assert matrix == independent['matrix_identity']
    assert identity(independent['producer_report']['path']) == independent['producer_report']
    assert identity(independent['producer_code']['path']) == independent['producer_code']
    training = identity(_release_source_path(HERE.parent / 'run_selection.py'))
    protocol = identity(HERE.parent / 'protocol.json')
    gate = {'passed': True, 'created_utc': datetime.now(timezone.utc).isoformat(), 'matrix_identity': matrix, 'independent_report_identity': identity(independent_path), 'prefit_code_review_identity': identity(review_path), 'training_script_identity': training, 'protocol_identity': protocol, 'gate_script_identity': identity(__file__), 'checks_passed': independent['checks_passed'], 'scope': 'Source preparation and pre-fit implementation approved; post-fit model and metric verification remains required', 'source_or_model_edits_by_verifier': False}
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(gate, stream, indent=2, ensure_ascii=True, allow_nan=False)
    print(json.dumps({'passed': True, 'checks_passed': gate['checks_passed'], 'gate': identity(destination), 'private_rows_or_identifiers_logged': False}, indent=2))
if __name__ == '__main__':
    main()
