import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from opt.artifacts import write_operation
from opt.operation import (
    OPERATION_MODEL_VERSION,
    _cached_metrics,
    _run_fingerprint,
    parameter_fingerprint,
)


V5 = "2026.08-continuous-shed-strict-balance-v5"


def _params() -> dict:
    return {
        "time": {"timestep": 5},
        "costs": {"bess_degradation_per_kwh": 0.1},
        "BESS": {
            "Emax_kwh": 1.0,
            "DoD_frac": 0.9,
            "eta_c": 1.0,
            "eta_d": 1.0,
            "E_init_kwh": 0.5,
        },
        "sizing_case_applied": {"source_sha256": "sizing-sha"},
    }


def _write_v5_case(directory: Path, residual_kw: float = 0.0) -> tuple[dict, dict]:
    params = _params()
    run_config = {"controller": "test", "n_iters": 1}
    operation = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2009-05-01")],
            "cost_total": [0.0],
            "P_bess_charge_mag_kw": [0.0],
            "P_bess_discharge_kw": [0.0],
            "Residual_kw": [residual_kw],
            "P_grid_in_kw": [0.0],
            "P_grid_out_kw": [0.0],
            "outage_active": [False],
            "E_kwh": [0.5],
        }
    )
    write_operation(operation, directory)
    metrics = {
        "operation_model_version": V5,
        "parameter_fingerprint_sha256": parameter_fingerprint(params),
        "run_fingerprint_sha256": _run_fingerprint(run_config),
        "sizing_source_sha256": "sizing-sha",
    }
    (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (directory / "parameters_used.json").write_text(
        json.dumps({"_operation_model": {"version": V5}}), encoding="utf-8"
    )
    return params, run_config


class OperationCacheTests(unittest.TestCase):
    def test_v5_cache_is_migrated_after_physical_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            params, run_config = _write_v5_case(path)

            cached = _cached_metrics(path, params, run_config)

            self.assertIsNotNone(cached)
            self.assertEqual(cached["operation_model_version"], OPERATION_MODEL_VERSION)
            self.assertEqual(cached["operation_model_compatible_from"], V5)
            stored = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
            provenance = json.loads(
                (path / "parameters_used.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stored["operation_model_version"], OPERATION_MODEL_VERSION)
            self.assertEqual(
                provenance["_operation_model"]["version"], OPERATION_MODEL_VERSION
            )
            self.assertEqual(
                provenance["_operation_model"]["cache_compatible_from"], V5
            )

    def test_v5_cache_with_failed_physics_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            params, run_config = _write_v5_case(path, residual_kw=0.01)

            cached = _cached_metrics(path, params, run_config)

            self.assertIsNone(cached)
            stored = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["operation_model_version"], V5)


if __name__ == "__main__":
    unittest.main()
