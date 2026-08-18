from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from opt.artifacts import (
    operation_artifact_path,
    read_operation,
    read_stochastic_plan,
    write_operation,
    write_stochastic_plan,
)


class ResultArtifactTests(unittest.TestCase):
    def test_operation_parquet_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            frame = pd.DataFrame({
                "timestamp": pd.date_range("2010-01-01", periods=2, freq="5min"),
                "P_bess_kw": [0.1, np.nan],
                "outage_active": [False, True],
                "clamps": [{"grid": {"kW": np.float64(0.2)}}, {}],
            }).set_index("timestamp")
            target = write_operation(frame, directory)
            restored = read_operation(directory)

            self.assertEqual(target.name, "operation_final.parquet")
            self.assertEqual(len(restored), 2)
            self.assertEqual(list(restored.columns), [
                "timestamp", "P_bess_kw", "outage_active", "clamps",
            ])
            self.assertEqual(json.loads(restored.loc[0, "clamps"])["grid"]["kW"], 0.2)

    def test_operation_reader_accepts_legacy_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            pd.DataFrame({"timestamp": ["2010-01-01"], "value": [1.0]}).to_csv(
                directory / "operation_final.csv", index=False,
            )
            self.assertEqual(operation_artifact_path(directory).suffix, ".csv")
            self.assertEqual(float(read_operation(directory).loc[0, "value"]), 1.0)

    def test_stochastic_plan_parquet_round_trip(self):
        payload = {
            "decision_time": "2010-01-01T00:00:00",
            "actions_cached": [{"timestamp": "2010-01-01T00:00:00", "gamma": 0}],
            "metadata": {"contingencies": ["c0", "c1"]},
            "scenarios": {
                "s0": {
                    "c0": [{"timestamp": "2010-01-01T00:00:00", "P_bess_kw": 0.1}],
                    "c1": [{"timestamp": "2010-01-01T00:00:00", "P_bess_kw": 0.2}],
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            target = write_stochastic_plan(payload, Path(tmp) / "plan.json")
            restored = read_stochastic_plan(target)

            self.assertEqual(target.suffix, ".parquet")
            self.assertEqual(restored, payload)


if __name__ == "__main__":
    unittest.main()
