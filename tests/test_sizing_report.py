import importlib.util
import tempfile
import unittest
from pathlib import Path


def _load_sizing_experiment():
    path = Path(__file__).resolve().parents[1] / "experiments" / "01_sizing.py"
    spec = importlib.util.spec_from_file_location("sizing_experiment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SizingReportTests(unittest.TestCase):
    def test_gulliver_pdf_report_is_generated(self):
        sizing_experiment = _load_sizing_experiment()
        self.assertTrue(sizing_experiment.GULLIVER_FONT_PATH.exists())
        self.assertIn("gulliver", sizing_experiment.GULLIVER_FONT_FAMILY.lower())

        def payload(pv, bess, capex, objective):
            years = {1: bess, 2: 0.98 * bess}
            initial = {1: 0.6 * bess, 2: 0.6 * 0.98 * bess}
            return {
                "decision_variables": {
                    "P_hat_PV_kw": pv,
                    "E_hat_BESS_kwh": bess,
                },
                "bess_capacity_by_year_kwh": years,
                "bess_initial_energy_by_year_kwh": initial,
                "objective_breakdown": {
                    "CAPEX": capex,
                    "Objective": objective,
                    "OPEX_annual_by_year": {1: 100.0, 2: 105.0},
                },
            }

        comparison = {
            "alpha_eq_0": payload(1.5, 3.0, 7000.0, 12000.0),
            "alpha_gt_0": payload(1.6, 3.3, 7500.0, 12500.0),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = sizing_experiment._save_comparison_report(
                root,
                comparison,
                discount_rate=0.08,
            )
            self.assertTrue(report.exists())
            self.assertGreater(report.stat().st_size, 1000)
            self.assertEqual(report.read_bytes()[:4], b"%PDF")


if __name__ == "__main__":
    unittest.main()
