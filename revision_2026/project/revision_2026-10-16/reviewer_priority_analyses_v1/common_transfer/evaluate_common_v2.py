"""Evaluation-only numeric dtype repair; frozen models and training code untouched."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from datetime import datetime, timezone
from pathlib import Path
import json
import run_common as frozen
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

class FloatingClusterMetrics(frozen.WeightedMetrics):

    def compute(self, weights):
        return super().compute(np.asarray(weights, dtype=np.float64))

def integer_weight_tests():
    rng = np.random.default_rng(640)
    cases = 0
    for n in (12, 51, 172):
        for tied in (False, True):
            y = rng.integers(0, 2, n)
            p = rng.random(n)
            if tied:
                p = np.round(p, 1)
            calc = FloatingClusterMetrics(y, p)
            for _ in range(20):
                w = rng.multinomial(n, np.full(n, 1 / n))
                if not (np.dot(w, y) and np.dot(w, 1 - y)):
                    continue
                expected = [roc_auc_score(y, p, sample_weight=w), average_precision_score(y, p, sample_weight=w), brier_score_loss(y, p, sample_weight=w)]
                assert np.allclose(calc.compute(w), expected, atol=2e-14, rtol=0)
                cases += 1
    return cases

def main():
    root = frozen.ROOT
    lock = _release_resolve_config(json.loads((root / 'analysis_lock.json').read_text(encoding='utf-8-sig')))
    for item in lock['identities']:
        assert frozen.identity(Path(item['path'])) == item
    inputs = [Path(__file__), root / 'analysis_lock.json', _release_source_path(root / 'run_common.py'), root / 'aggregate/model_manifest.json']
    inputs += [root / f'private/{centre}_predictions.npz' for centre in ('WHU', 'MIMIC')]
    before = [frozen.identity(p) for p in inputs]
    with threadpool_limits(limits=4):
        cases = integer_weight_tests()
        frozen.WeightedMetrics = FloatingClusterMetrics
        frozen.evaluate()
    after = [frozen.identity(p) for p in inputs]
    assert before == after
    frozen.write_json(root / 'aggregate/evaluation_v2_verification.json', {'passed': True, 'execution_complete': True, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'integer_weight_sklearn_metric_cases': cases, 'input_identities_before': before, 'input_identities_after': after, 'locked_training_code_and_models_unchanged': True, 'arithmetic_repair': 'Cast integer patient-cluster multiplicities to float64 before weighted cumulative precision division. No score, label, bootstrap draw, protocol or model changed.', 'entrypoint': frozen.identity(Path(__file__)), 'execution': frozen.identity(root / 'aggregate/execution.json'), 'fitting_called_by_this_entrypoint': False})
if __name__ == '__main__':
    main()
