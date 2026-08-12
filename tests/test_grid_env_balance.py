import unittest

import pandas as pd

from env.grid_env import GridEnv


def _offgrid_env(load_kw: float, pv_kw: float, energy_kwh: float) -> GridEnv:
    env = GridEnv.__new__(GridEnv)
    env.mode = "offgrid"
    env.debug = False
    env.clamp_soc_pct = True
    env.tol_kw = 1e-6
    env.dt_h = 5.0 / 60.0
    env.timestamp = pd.Timestamp("2009-05-02 07:00:00")
    env.load_kw_s = pd.Series({env.timestamp: load_kw})
    env.pv_kw_s = pd.Series({env.timestamp: pv_kw})
    env.bess = {
        "P_max": 0.2,
        "ramp": 0.2,
        "eta_c": 1.0,
        "eta_d": 1.0,
        "E_min": 0.0,
        "E_max": 1.0,
        "E_nom": 1.0,
    }
    env.grid_caps = {"P_import_max": 2.0, "P_export_max": 2.0}
    env.costs = {
        "TOU": {},
        "c_shed": 5.0,
        "c_curt": 0.001,
    }
    env.noise = {
        "enabled": False,
        "std_kw": 0.0,
        "std_frac": 0.0,
    }
    env._rng = None
    env.E_meas = energy_kwh
    env._prev_Pb = 0.0
    env._rows = []
    env.iter_k = 0
    env.n_iters = 1
    env.outage_active = True
    env.outage_end_time = env.timestamp + pd.Timedelta(hours=1)
    env._outage_calendar = [
        {
            "start": env.timestamp,
            "end": env.outage_end_time,
            "duration_h": 1.0,
        }
    ]
    return env


class GridEnvBalanceTests(unittest.TestCase):
    def test_offgrid_deficit_uses_continuous_exact_shedding(self):
        env = _offgrid_env(load_kw=1.0, pv_kw=0.0, energy_kwh=1.0)

        row, done = env.step(0.0, None, None)

        self.assertTrue(done)
        self.assertAlmostEqual(row["P_bess_kw"], 0.2)
        self.assertAlmostEqual(row["Shedding_kw"], 0.8)
        self.assertAlmostEqual(row["Load_served_kw"], 0.2)
        self.assertAlmostEqual(row["Residual_kw"], 0.0)
        self.assertEqual(row["P_grid_in_kw"], 0.0)
        self.assertEqual(row["P_grid_out_kw"], 0.0)
        self.assertAlmostEqual(
            row["clamps"]["offgrid_autofix_shed"]["new_XL_pct"],
            80.0,
        )

    def test_offgrid_surplus_uses_continuous_exact_curtailment(self):
        env = _offgrid_env(load_kw=0.0, pv_kw=1.0, energy_kwh=1.0)

        row, done = env.step(0.0, None, None)

        self.assertTrue(done)
        self.assertAlmostEqual(row["P_bess_kw"], 0.0)
        self.assertAlmostEqual(row["Curtailment_kw"], 1.0)
        self.assertAlmostEqual(row["PV_used_kw"], 0.0)
        self.assertAlmostEqual(row["Residual_kw"], 0.0)
        self.assertEqual(row["P_grid_in_kw"], 0.0)
        self.assertEqual(row["P_grid_out_kw"], 0.0)


if __name__ == "__main__":
    unittest.main()
