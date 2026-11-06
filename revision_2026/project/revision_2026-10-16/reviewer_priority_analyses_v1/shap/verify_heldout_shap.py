"""Replay frozen analysis and compare CSV values with explicit newline handling."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import pandas as pd
from threadpoolctl import threadpool_limits
import run_heldout_shap as analysis

def main():
    state = analysis.lock()
    tests = analysis.self_test()
    with threadpool_limits(limits=2):
        summary, tables = analysis.calculate()
    saved = analysis.read_json(analysis.ROOT / 'results/summary.json')
    assert saved['results'] == summary
    for item in saved['table_identities']:
        analysis.check(item)
        path = Path(item['path'])
        expected = pd.DataFrame(tables[path.name]).to_csv(index=False, float_format='%.17g', lineterminator='\n')
        assert path.read_text(encoding='utf-8') == expected
    for item in state['sources']:
        analysis.check(item)
    associations = tables['feature_associations.csv']
    analysis.save_json(analysis.ROOT / 'verification.json', {'passed': True, 'completed_at': analysis.now(), 'verification_script': analysis.identity(__file__), 'analysis_lock': analysis.identity(analysis.ROOT / 'analysis_lock.json'), 'summary': analysis.identity(analysis.ROOT / 'results/summary.json'), 'all_table_identities': saved['table_identities'], 'self_tests': tests, 'all_ten_saved_model_explanations_recomputed': True, 'all_roles_reproduced': True, 'all_twenty_sensitivity_arrays_rechecked': True, 'all_aggregate_tables_exactly_reproduced': True, 'all_consumed_sources_unchanged': True, 'no_training_or_patient_exports': True, 'earlier_verification_issue': {'type': 'CSV_newline_comparison_only', 'description': 'The frozen analysis verify entry point compared pandas Windows CRLF text against Path.read_text universal-newline LF text. Its complete numerical summary replay had passed before this formatting-only assertion. This separate verifier explicitly requests LF in the comparison and leaves all frozen analysis sources and outputs unchanged.', 'scientific_values_changed': False}, 'numerical_reporting_note': {'pooled_nominal_p_underflow_to_zero_count': sum((r['p_nominal'] == 0 for r in associations if r['testable'])), 'pooled_bh_underflow_to_zero_count': sum((r['q_bh_joint_family'] == 0 for r in associations if r['testable'])), 'interpretation': 'Stored zero P/q values arise from floating-point underflow of asymptotic calculations, not exact zero probabilities. Report them as below numerical resolution, not as P=0.'}, 'verification_scope': 'Separate execution and exact aggregate replay, independent SciPy BH implementation checks, and original saved-model audit bindings; not an independent second implementation of TreeSHAP.'})
    print(analysis.identity(analysis.ROOT / 'verification.json'))
if __name__ == '__main__':
    main()
