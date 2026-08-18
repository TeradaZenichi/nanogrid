from __future__ import annotations

import unittest

import pandas as pd

from opt.baseline import BaselineObservation, run_baseline_closed_loop
from opt.closed_loop import ControlAction
from opt.load_shifting import LoadShiftingController
from opt.peak_shaving import PeakShavingController
from opt.reserve_only import ReserveOnlyController
from opt.self_consumption import SelfConsumptionController


def observation(
    *, load: float = 0.8, pv: float = 0.2, energy: float = 0.8,
    tariff: float = 0.19,
) -> BaselineObservation:
    return BaselineObservation(
        timestamp=pd.Timestamp("2010-01-01 12:00:00"),
        load_kw=load,
        pv_kw=pv,
        energy_kwh=energy,
        energy_min_kwh=0.1,
        energy_max_kwh=1.0,
        power_max_kw=0.4,
        previous_bess_kw=0.0,
        tariff_per_kwh=tariff,
    )


class BaselineControllerTests(unittest.TestCase):
    def test_reserve_only_never_dispatches_on_grid(self):
        action = ReserveOnlyController().action(observation())
        self.assertEqual(action.p_bess_kw, 0.0)

    def test_self_consumption_tracks_local_mismatch_and_power_limit(self):
        controller = SelfConsumptionController()
        self.assertEqual(controller.action(observation()).p_bess_kw, 0.4)
        self.assertAlmostEqual(
            controller.action(observation(load=0.1, pv=0.35)).p_bess_kw,
            -0.25,
        )

    def test_self_consumption_respects_fixed_energy_reserve(self):
        controller = SelfConsumptionController(reserve_fraction=0.5)
        reserve = 0.1 + 0.5 * (1.0 - 0.1)
        action = controller.action(observation(energy=reserve))
        self.assertEqual(action.p_bess_kw, 0.0)

    def test_load_shifting_charges_off_peak_and_discharges_at_peak(self):
        controller = LoadShiftingController(0.125, 0.34)
        self.assertEqual(
            controller.action(observation(tariff=0.125)).p_bess_kw,
            -0.4,
        )
        self.assertAlmostEqual(
            controller.action(observation(load=0.45, pv=0.15, tariff=0.34)).p_bess_kw,
            0.3,
        )
        self.assertEqual(
            controller.action(observation(tariff=0.19)).p_bess_kw,
            0.0,
        )

    def test_peak_shaving_targets_the_fixed_import_limit(self):
        controller = PeakShavingController(import_limit_kw=0.5)
        action = controller.action(observation(load=1.0, pv=0.1))
        self.assertEqual(action.p_bess_kw, 0.4)
        recharge = controller.action(observation(load=0.1, pv=0.3))
        self.assertAlmostEqual(recharge.p_bess_kw, -0.2)

    def test_controller_parameters_are_validated(self):
        with self.assertRaises(ValueError):
            SelfConsumptionController(reserve_fraction=1.1)
        with self.assertRaises(ValueError):
            LoadShiftingController(0.34, 0.125)
        with self.assertRaises(ValueError):
            PeakShavingController(import_limit_kw=-0.1)

    def test_common_loop_leaves_islanded_dispatch_to_the_environment(self):
        class Controller:
            name = "test"
            version = "test-v1"

            def __init__(self):
                self.calls = 0

            def action(self, _observation):
                self.calls += 1
                return ControlAction(p_bess_kw=0.25)

            def configuration(self):
                return {}

        class Environment:
            def __init__(self):
                self.iter_k = 0
                self.modes = ("ongrid", "offgrid")
                self.load_kw_s = pd.Series({
                    pd.Timestamp("2010-01-01 00:00"): 0.5,
                    pd.Timestamp("2010-01-01 00:05"): 0.5,
                })
                self.pv_kw_s = self.load_kw_s * 0.0
                self.bess = {"E_min": 0.1, "E_max": 1.0, "P_max": 0.4}
                self.costs = {"TOU": {"00:00": 0.125}}
                self.E_meas = 0.8
                self.commands = []

            @property
            def timestamp(self):
                return pd.Timestamp("2010-01-01 00:00") + pd.Timedelta(minutes=5 * self.iter_k)

            @property
            def mode(self):
                return self.modes[self.iter_k]

            def done(self):
                return self.iter_k >= len(self.modes)

            def step(self, **command):
                self.commands.append(command)
                self.iter_k += 1
                return {"P_bess_kw": command["P_bess_kw"]}, self.done()

        controller = Controller()
        env = Environment()
        result = run_baseline_closed_loop(
            env=env, controller=controller, n_iters=2, progress_every=100
        )

        self.assertEqual(controller.calls, 1)
        self.assertEqual(env.commands[0]["P_bess_kw"], 0.25)
        self.assertEqual(env.commands[1]["P_bess_kw"], 0.0)
        self.assertEqual(result.n_offgrid_steps, 1)


if __name__ == "__main__":
    unittest.main()
