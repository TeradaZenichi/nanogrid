import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from forecasting.causal import FORECAST_ASSEMBLY_VERSION
from forecasting.evaluate_prototype import (
    FORECAST_EVALUATION_VERSION,
    _evaluation_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "experiments" / "12_corrected_pipeline.py"
SPEC = importlib.util.spec_from_file_location("corrected_pipeline", PIPELINE_PATH)
PIPELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PIPELINE
SPEC.loader.exec_module(PIPELINE)


def _row(combo, h, t1, t2, controller, cost, solve_time, audit_pass=True):
    return {
        "case_id": f"mesh/{controller}/{combo}",
        "combo": combo,
        "h": h,
        "t1": t1,
        "t2": t2,
        "controller_name": controller,
        "audit_pass": audit_pass,
        "operation_total_cost": cost,
        "avg_solve_time_s": solve_time,
        "n_solve_fail": 0,
    }


def _complete_summary():
    meshes = {
        "mesh_a": (12, 5, 30),
        "mesh_b": (24, 10, 60),
        "mesh_c": (36, 15, 120),
    }
    costs = {
        "ideal": {"mesh_a": 100.0, "mesh_b": 110.0, "mesh_c": 120.0},
        "prototype": {"mesh_a": 120.0, "mesh_b": 100.0, "mesh_c": 110.0},
        "lstm": {"mesh_a": 130.0, "mesh_b": 110.0, "mesh_c": 100.0},
    }
    solve_times = {"mesh_a": 1.0, "mesh_b": 2.0, "mesh_c": 1.5}
    rows = []
    for controller in PIPELINE.MESH_CONTROLLERS:
        for combo, (h, t1, t2) in meshes.items():
            rows.append(
                _row(
                    combo,
                    h,
                    t1,
                    t2,
                    controller,
                    costs[controller][combo],
                    solve_times[combo],
                )
            )
    return pd.DataFrame(rows)


class CorrectedPipelineTests(unittest.TestCase):
    def test_temporal_mesh_campaign_includes_lstm(self):
        self.assertEqual(PIPELINE.DEFAULT_OUT_ROOT, Path("Results"))
        self.assertEqual(
            PIPELINE.EVALUATION_CONTROLLERS,
            ("ideal", "prototype", "lstm"),
        )
        self.assertEqual(
            PIPELINE.SELECTION_CONTROLLERS,
            ("prototype", "lstm"),
        )
        mesh_runs = (
            len(PIPELINE.MESH_H)
            * len(PIPELINE.MESH_T1)
            * len(PIPELINE.MESH_T2)
            * len(PIPELINE.EVALUATION_CONTROLLERS)
        )
        annual_configurations = mesh_runs
        annual_monthly_runs = annual_configurations * len(
            PIPELINE.MONTHLY_VALIDATION_WINDOWS
        )
        incremental_runs = annual_configurations * (
            len(PIPELINE.MONTHLY_VALIDATION_WINDOWS) - 1
        )
        self.assertEqual(mesh_runs, 81)
        self.assertEqual(annual_configurations, 81)
        self.assertEqual(len(PIPELINE.MONTHLY_VALIDATION_WINDOWS), 12)
        self.assertEqual(annual_monthly_runs, 972)
        self.assertEqual(incremental_runs, 891)
        self.assertEqual(
            set(PIPELINE.MONTHLY_VALIDATION_WINDOWS),
            set(PIPELINE.MONTHS),
        )
        self.assertEqual(
            set(PIPELINE.MONTHLY_OUTAGE_SEEDS),
            set(PIPELINE.MONTHS),
        )
        self.assertEqual(
            list(PIPELINE.MONTHLY_OUTAGE_SEEDS.values()),
            list(range(42, 54)),
        )
        self.assertEqual(
            len(PIPELINE.CAUSAL_PILOT_MESHES)
            * len(PIPELINE.MONTHLY_VALIDATION_WINDOWS)
            * len(PIPELINE.SELECTION_CONTROLLERS),
            72,
        )

    def test_final_forecast_uses_month_specific_outage_seeds(self):
        args = SimpleNamespace(
            out_root=Path("Results"),
            forecast_controllers="ideal,prototype,lstm",
            n_iters=3,
            fresh=False,
            workers=1,
        )
        captured = {}

        def capture(tasks, workers, summary_path):
            captured["tasks"] = tasks
            return pd.DataFrame()

        with (
            patch.object(PIPELINE, "_champion", return_value={"h": 12, "t1": 5, "t2": 30}),
            patch.object(PIPELINE, "run_tasks", side_effect=capture),
        ):
            PIPELINE.stage_forecast(args)

        tasks = captured["tasks"]
        self.assertEqual(len(tasks), 36)
        for task in tasks:
            month = task["labels"]["month"]
            expected_seed = PIPELINE.MONTHLY_OUTAGE_SEEDS[month]
            self.assertEqual(task["seed"], expected_seed)
            self.assertEqual(task["labels"]["outage_seed"], expected_seed)

    def test_forecast_evaluation_manifest_tracks_causal_configuration(self):
        params = {"sizing_case_applied": {"source_sha256": "sizing-hash"}}
        manifest = _evaluation_manifest(
            params,
            days=365,
            every_min=60,
            start=None,
            with_lstm=True,
        )

        self.assertEqual(
            manifest["forecast_evaluation_version"],
            FORECAST_EVALUATION_VERSION,
        )
        self.assertEqual(
            manifest["forecast_assembly_version"],
            FORECAST_ASSEMBLY_VERSION,
        )
        self.assertEqual(manifest["sizing_source_sha256"], "sizing-hash")
        changed = _evaluation_manifest(
            params,
            days=28,
            every_min=60,
            start=None,
            with_lstm=True,
        )
        self.assertNotEqual(manifest, changed)

    def test_expanded_robustness_uses_four_seasonal_windows(self):
        args = SimpleNamespace(
            out_root=Path("Results"),
            n_iters=3,
            fresh=False,
            workers=4,
        )
        captured = {}

        def capture(tasks, workers, summary_path):
            captured["tasks"] = tasks
            captured["workers"] = workers
            return pd.DataFrame()

        with (
            patch.object(PIPELINE, "_champion", return_value={"h": 12, "t1": 5, "t2": 30}),
            patch.object(PIPELINE, "run_tasks", side_effect=capture),
        ):
            PIPELINE.stage_robustness(args)

        tasks = captured["tasks"]
        expected = (
            len(PIPELINE.ROBUSTNESS_WINDOWS)
            * len(PIPELINE.ROBUSTNESS_VARIANTS)
            * 2
        )
        self.assertEqual(expected, 56)
        self.assertEqual(len(tasks), expected)
        self.assertEqual(len({task["case_id"] for task in tasks}), expected)
        self.assertEqual(captured["workers"], 2)
        for window, _, window_seed in PIPELINE.ROBUSTNESS_WINDOWS:
            window_tasks = [task for task in tasks if task["labels"]["window"] == window]
            self.assertEqual(len(window_tasks), 14)
            base_tasks = [task for task in window_tasks if task["labels"]["variant"] == "base"]
            self.assertEqual({task["seed"] for task in base_tasks}, {window_seed})
            self.assertEqual(
                {task["controller"] for task in base_tasks},
                {"prototype", "stochastic"},
            )

    def test_strict_audit_rejects_a_power_balance_residual(self):
        audited = PIPELINE._audit_metrics(
            {
                "n_solve_fail": 0,
                "max_simultaneous_ch_dis_kw": 0.0,
                "max_plant_simultaneous_ch_dis_kw": 0.0,
                "max_abs_power_balance_residual_kw": 0.01,
                "max_grid_exchange_during_outage_kw": 0.0,
                "max_bess_energy_bound_violation_kwh": 0.0,
                "max_bess_energy_transition_residual_kwh": 0.0,
            }
        )

        self.assertFalse(audited["audit_pass"])
        self.assertEqual(audited["audit_failures"], 1)
        self.assertIn("power-balance residual", audited["audit_reasons"])

    def test_selection_uses_per_forecaster_regret(self):
        candidates = PIPELINE._candidate_table(_complete_summary())
        champion = candidates.iloc[0]
        self.assertEqual(champion["combo"], "mesh_c")
        self.assertEqual(int(champion["controllers"]), 2)
        for column in (
            "regret_prototype",
            "regret_lstm",
            "mean_regret",
            "max_regret",
            "selection_score",
        ):
            self.assertIn(column, candidates.columns)
        self.assertNotIn("regret_ideal", candidates.columns)
        self.assertAlmostEqual(float(champion["regret_prototype"]), 0.10)
        self.assertAlmostEqual(float(champion["regret_lstm"]), 0.0)

    def test_ideal_target_does_not_affect_mesh_selection(self):
        summary = _complete_summary()
        original = PIPELINE._candidate_table(summary)["combo"].tolist()
        summary.loc[summary["controller_name"] == "ideal", "operation_total_cost"] = [
            1e9, 0.0, -1e9
        ]
        changed = PIPELINE._candidate_table(summary)["combo"].tolist()
        self.assertEqual(changed, original)

    def test_incomplete_forecaster_results_cannot_select_a_mesh(self):
        complete = [
            _row("complete", 12, 5, 30, controller, 100.0, 1.0)
            for controller in PIPELINE.SELECTION_CONTROLLERS
        ]
        incomplete = [
            _row("incomplete", 24, 10, 60, "ideal", 1.0, 0.1),
            _row("incomplete", 24, 10, 60, "prototype", 1.0, 0.1),
        ]
        candidates = PIPELINE._candidate_table(pd.DataFrame(complete + incomplete))
        self.assertEqual(candidates["combo"].tolist(), ["complete"])

        without_lstm = pd.DataFrame(complete + incomplete)
        without_lstm = without_lstm[without_lstm["controller_name"] != "lstm"]
        with self.assertRaisesRegex(RuntimeError, "No audited cases for: lstm"):
            PIPELINE._candidate_table(without_lstm)

    def test_monthly_selection_requires_complete_coverage(self):
        summary = _complete_summary()
        with self.assertRaisesRegex(
            RuntimeError,
            r"No mesh has 2 audited successful run\(s\)",
        ):
            PIPELINE._candidate_table(
                summary,
                expected_runs_per_controller=2,
            )

        repeated = pd.concat([summary, summary.assign(case_id=summary["case_id"] + "/m2")])
        candidates = PIPELINE._candidate_table(
            repeated,
            expected_runs_per_controller=2,
        )
        self.assertEqual(set(candidates["runs"]), {4})

    def test_selection_reports_audit_rejections_separately_from_simulation_failure(self):
        summary = _complete_summary()
        prototype = summary["controller_name"] == "prototype"
        summary.loc[prototype, "audit_pass"] = False
        summary.loc[prototype, "max_simultaneous_ch_dis_kw"] = 0.0799356

        with self.assertRaisesRegex(
            RuntimeError,
            r"Mesh simulations completed.*prototype \(3 rejected by audit;"
            r" maximum charge/discharge overlap 0\.0799356 kW\)",
        ):
            PIPELINE._candidate_table(summary)

    def test_selection_artifact_records_and_validates_method(self):
        summary = _complete_summary()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_path = tmp_path / "summary.csv"
            output_path = tmp_path / "selected_meshes.json"
            summary.to_csv(summary_path, index=False)
            selected = PIPELINE.select_meshes(summary_path, output_path, 1)
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(selected, [{"h": 36, "t1": 15, "t2": 120}])
        self.assertTrue(PIPELINE._selection_artifact_is_current(payload))
        self.assertEqual(
            payload["mesh_selection_version"],
            PIPELINE.MESH_SELECTION_VERSION,
        )
        self.assertEqual(
            payload["required_controllers"],
            list(PIPELINE.SELECTION_CONTROLLERS),
        )
        old_payload = dict(payload, required_controllers=["ideal", "prototype", "lstm"])
        self.assertFalse(PIPELINE._selection_artifact_is_current(old_payload))


if __name__ == "__main__":
    unittest.main()
