from __future__ import annotations

import unittest

import pandas as pd

from opt.closed_loop import ControlAction, run_mpc_closed_loop


class _FakeEnvironment:
    def __init__(self):
        self.modes = ("ongrid", "offgrid", "ongrid")
        self.realized_powers = (0.4, 0.2, -0.1)
        self.iter_k = 0
        self.dt_min = 5
        self.E_meas = 0.5
        self.steps = []

    @property
    def timestamp(self):
        return pd.Timestamp("2020-01-01") + pd.Timedelta(minutes=5 * self.iter_k)

    @property
    def mode(self):
        return self.modes[self.iter_k]

    def done(self):
        return self.iter_k >= len(self.modes)

    def step(self, **command):
        self.steps.append(command)
        realized = {"P_bess_kw": self.realized_powers[self.iter_k]}
        self.iter_k += 1
        return realized, self.done()


class _FakeForecaster:
    def __init__(self):
        self.calls = []

    def get_forecasts(self, **request):
        self.calls.append(request)
        return {"load_kw": {}, "pv_kw": {}}


class _FakeMPC:
    def __init__(self):
        self.builds = []
        self.solves = []
        self.actions = iter((
            {"P_bess_kw": 1.0, "X_L": 0.1, "X_PV": 0.2, "obj": 3.0},
            {"P_bess_kw": -0.5, "X_L": 0.0, "X_PV": 0.0, "obj": 4.0},
        ))

    def build(self, **values):
        self.builds.append(values)

    def solve(self, **options):
        self.solves.append(options)

    def max_simultaneous_charge_discharge_kw(self):
        return 0.0

    def extract_first_step(self, scenario):
        if scenario != "c0":
            raise AssertionError("The executable action must use the base case")
        return next(self.actions)


class ClosedLoopTests(unittest.TestCase):
    def test_control_action_preserves_historical_safe_fallback(self):
        action = ControlAction.from_mpc_step(None)
        self.assertEqual(action.p_bess_kw, 0.0)
        self.assertIsNone(action.load_shed_fraction)
        self.assertIsNone(action.pv_curtail_fraction)
        self.assertIsNone(action.objective)

    def test_loop_is_observe_optimize_apply_and_feedback(self):
        env = _FakeEnvironment()
        forecaster = _FakeForecaster()
        mpc = _FakeMPC()

        result = run_mpc_closed_loop(
            env=env, mpc=mpc, forecaster=forecaster, solver_opts={"threads": 1},
            n_iters=3, controller_label="test", progress_every=100,
        )

        self.assertEqual(result.n_ongrid_steps, 2)
        self.assertEqual(result.n_offgrid_steps, 1)
        self.assertEqual(result.n_solve_ok, 2)
        self.assertEqual(result.n_solve_fail, 0)
        self.assertEqual(len(forecaster.calls), 2)
        self.assertEqual(len(mpc.builds), 2)
        self.assertEqual(mpc.builds[0]["P_bess_hat_kw"], 0.0)
        self.assertEqual(mpc.builds[1]["P_bess_hat_kw"], 0.2)
        self.assertEqual(mpc.solves, [{"tee": False, "threads": 1}] * 2)

        self.assertEqual(env.steps[0]["P_bess_kw"], 1.0)
        self.assertEqual(env.steps[0]["X_L"], 0.1)
        self.assertEqual(env.steps[0]["X_PV"], 0.2)
        self.assertEqual(env.steps[1]["P_bess_kw"], 0.0)
        self.assertIsNone(env.steps[1]["X_L"])
        self.assertIsNone(env.steps[1]["X_PV"])
        self.assertEqual(env.steps[2]["P_bess_kw"], -0.5)


if __name__ == "__main__":
    unittest.main()
