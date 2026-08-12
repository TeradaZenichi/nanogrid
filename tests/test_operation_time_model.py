import json
import math
import re
import unittest
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd

from pyomo.core.expr.visitor import identify_variables

from opt.ongrid import OnGridMPC, Parameters as MPCParameters
from opt.operation import (
    _operation_cost_metrics,
    _operation_physical_metrics,
    _run_fingerprint,
)
from opt.stochastic import OnGridStochasticOperation, Parameters as StochasticParameters
from opt.utils import outage_hazard_rate_per_hour


ROOT = Path(__file__).resolve().parents[1]


def _params(h=12, t1=5, t2=60):
    params = json.loads((ROOT / "data" / "parameters.json").read_text(encoding="utf-8"))
    params["time"].update(
        horizon_hours=int(h),
        timestep_1_min=int(t1),
        timestep_2_min=int(t2),
    )
    return params


def _mpc_time_data(h=12, t1=5, t2=60):
    return MPCParameters(_params(h, t1, t2)).build_time_data(datetime(2009, 5, 1))


def _offsets_and_weights(data):
    start = data["times"][0]
    return [
        (
            (c - start).total_seconds() / 3600.0,
            float(data["pi_c"][c]),
        )
        for c in data["contingencies"]
    ]


def _assert_close(actual, expected, tol=1e-12):
    assert math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol)


def _check_outage_support_and_weights_do_not_depend_on_mesh():
    reference = _mpc_time_data(12, 5, 30)
    expected = _offsets_and_weights(reference)
    assert [x[0] for x in expected] == [0.0, 2.0, 4.0, 6.0, 8.0]

    for t1 in (5, 10, 15):
        for t2 in (30, 60, 120):
            data = _mpc_time_data(12, t1, t2)
            actual = _offsets_and_weights(data)
            assert [x[0] for x in actual] == [x[0] for x in expected]
            for actual_weight, expected_weight in zip(
                [x[1] for x in actual], [x[1] for x in expected]
            ):
                _assert_close(actual_weight, expected_weight)
            _assert_close(sum(float(data["pi_c"][c]) for c in data["scenarios"]), 1.0)
            for c in data["contingencies"]:
                duration = sum(float(data["dt_h_map"][t]) for t in data["windows"][c])
                _assert_close(duration, 2.0)


def _check_daily_probability_is_calibrated_and_horizon_mass_increases():
    hazard = outage_hazard_rate_per_hour(10.0, 24.0)
    _assert_close(1.0 - math.exp(-hazard * 24.0), 0.10)

    masses = []
    for horizon in (12, 24, 36):
        data = _mpc_time_data(horizon, 5, 60)
        masses.append(1.0 - float(data["pi_c"]["c0"]))
        expected = 1.0 - math.exp(-hazard * (horizon - 2.0))
        _assert_close(masses[-1], expected)
    assert masses[0] < masses[1] < masses[2]


def _check_ramp_reference_is_physical_not_timestep_1():
    refs = [_mpc_time_data(12, t1, 60)["dt_ref_h"] for t1 in (5, 10, 15)]
    for actual, expected in zip(refs, [5.0 / 60.0] * 3):
        _assert_close(actual, expected)


def _check_run_fingerprint_invalidates_changed_campaign_settings():
    base = {
        "controller": "mpc",
        "start_ts": "2009-05-01T00:00:00",
        "n_iters": 2880,
        "forecaster": "ideal",
    }
    changed = dict(base, n_iters=5760)
    assert _run_fingerprint(base) == _run_fingerprint(dict(base))
    assert _run_fingerprint(base) != _run_fingerprint(changed)


def _check_mesh_that_cannot_represent_fixed_outages_is_rejected():
    # 50 min divides the 10 h coarse region, but not the 2 h physical support.
    try:
        _mpc_time_data(12, 10, 50)
    except ValueError as exc:
        assert re.search("not aligned|not represented exactly", str(exc))
    else:
        raise AssertionError("Invalid contingency support was not rejected")


def _check_mpc_has_terminal_energy_closure_including_last_interval():
    params = _params(12, 5, 60)
    mpc = OnGridMPC(params, relaxation=True)
    time_data = mpc.param.build_time_data(datetime(2009, 5, 1))
    forecasts = {
        "load_kw": {t: 1.0 for t in time_data["times"]},
        "pv_kw": {t: 0.5 for t in time_data["times"]},
    }
    model = mpc.build(
        datetime(2009, 5, 1),
        forecasts,
        E_hat_kwh=float(params["BESS"]["E_init_kwh"]),
    )
    assert hasattr(model, "TerminalEnergy")
    last_t = time_data["times"][-1]
    terminal_var_ids = {id(var) for var in identify_variables(model.TerminalEnergy.body)}
    assert id(model.E[last_t, "c0"]) in terminal_var_ids
    assert id(model.P_ch[last_t, "c0"]) in terminal_var_ids
    assert id(model.P_dis[last_t, "c0"]) in terminal_var_ids
    for component in (
        "PchargeEqualPre",
        "PdischargeEqualPre",
        "EequalPre",
        "XLEequalPre",
        "XPVEequalPre",
        "PgridInEqualPre",
        "PgridOutEqualPre",
    ):
        assert hasattr(model, component)
    assert not hasattr(model, "Pbess_abs")
    assert not hasattr(model, "AbsPos")
    assert not hasattr(model, "AbsNeg")
    objective_var_ids = {id(var) for var in identify_variables(model.Objective.expr)}
    first_t = time_data["times"][0]
    assert id(model.P_ch[first_t, "c0"]) in objective_var_ids
    assert id(model.P_dis[first_t, "c0"]) in objective_var_ids


def _check_stochastic_and_mpc_use_same_outage_mass():
    params = _params(24, 10, 60)
    start = datetime(2009, 5, 1)
    mpc_data = MPCParameters(deepcopy(params)).build_time_data(start)
    stochastic_data = StochasticParameters(deepcopy(params)).build_time_data(start)
    _assert_close(sum(stochastic_data["pi_c"].values()), 1.0)
    _assert_close(stochastic_data["pi_c"]["c0"], mpc_data["pi_c"]["c0"])
    assert len(stochastic_data["contingencies"]) == len(mpc_data["scenarios"])


def _check_stochastic_objective_uses_bidirectional_throughput():
    params = _params(12, 5, 60)
    operation = OnGridStochasticOperation(params, relaxation=True)
    model = operation.build(
        datetime(2009, 5, 1),
        E_hat_kwh=float(params["BESS"]["E_init_kwh"]),
    )
    assert not hasattr(model, "Pbess_abs")
    assert not hasattr(model, "AbsPos")
    assert not hasattr(model, "AbsNeg")
    objective_var_ids = {id(var) for var in identify_variables(model.Objective.expr)}
    first_t = next(iter(model.T))
    assert id(model.P_ch[first_t]) in objective_var_ids
    assert id(model.P_dis[first_t]) in objective_var_ids


class OperationTimeModelTests(unittest.TestCase):
    def test_outage_support_and_weights_do_not_depend_on_mesh(self):
        _check_outage_support_and_weights_do_not_depend_on_mesh()

    def test_daily_probability_is_calibrated_and_horizon_mass_increases(self):
        _check_daily_probability_is_calibrated_and_horizon_mass_increases()

    def test_ramp_reference_is_physical_not_timestep_1(self):
        _check_ramp_reference_is_physical_not_timestep_1()

    def test_run_fingerprint_invalidates_changed_campaign_settings(self):
        _check_run_fingerprint_invalidates_changed_campaign_settings()

    def test_mesh_that_cannot_represent_fixed_outages_is_rejected(self):
        _check_mesh_that_cannot_represent_fixed_outages_is_rejected()

    def test_mpc_has_terminal_energy_closure_including_last_interval(self):
        _check_mpc_has_terminal_energy_closure_including_last_interval()

    def test_stochastic_and_mpc_use_same_outage_mass(self):
        _check_stochastic_and_mpc_use_same_outage_mass()

    def test_stochastic_objective_uses_bidirectional_throughput(self):
        _check_stochastic_objective_uses_bidirectional_throughput()

    def test_realized_cost_includes_bidirectional_bess_wear(self):
        params = _params()
        params["time"]["timestep"] = 5
        params["costs"]["bess_degradation_per_kwh"] = 0.1
        operation = pd.DataFrame(
            {
                "cost_total": [1.0, 2.0],
                "P_bess_charge_mag_kw": [1.0, 0.0],
                "P_bess_discharge_kw": [0.0, 2.0],
            }
        )
        metrics = _operation_cost_metrics(operation, params)
        self.assertAlmostEqual(metrics["operation_grid_reliability_cost"], 3.0)
        self.assertAlmostEqual(metrics["operation_bess_throughput_kwh"], 0.25)
        self.assertAlmostEqual(metrics["operation_wear_cost"], 0.025)
        self.assertAlmostEqual(metrics["operation_total_cost"], 3.025)

    def test_realized_physical_audit_reconstructs_exact_balance_and_energy(self):
        params = _params()
        params["time"]["timestep"] = 5
        params["BESS"].update(
            Emax_kwh=1.0,
            E_init_kwh=0.5,
            DoD_frac=1.0,
            eta_c=1.0,
            eta_d=1.0,
        )
        operation = pd.DataFrame(
            {
                "Residual_kw": [0.0, 0.0],
                "P_grid_in_kw": [0.0, 1.0],
                "P_grid_out_kw": [0.0, 0.0],
                "outage_active": [True, False],
                "E_kwh": [0.5 + 1.0 / 12.0, 0.5],
                "P_bess_charge_mag_kw": [1.0, 0.0],
                "P_bess_discharge_kw": [0.0, 1.0],
            }
        )

        metrics = _operation_physical_metrics(operation, params)

        self.assertAlmostEqual(metrics["max_abs_power_balance_residual_kw"], 0.0)
        self.assertAlmostEqual(metrics["max_grid_exchange_during_outage_kw"], 0.0)
        self.assertAlmostEqual(
            metrics["max_bess_energy_transition_residual_kwh"],
            0.0,
        )
        self.assertAlmostEqual(
            metrics["max_bess_energy_bound_violation_kwh"],
            0.0,
        )
        self.assertAlmostEqual(metrics["max_plant_simultaneous_ch_dis_kw"], 0.0)


if __name__ == "__main__":
    unittest.main()
