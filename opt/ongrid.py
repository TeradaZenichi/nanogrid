# -*- coding: utf-8 -*-
"""
On-grid stochastic MPC.

This module keeps the public API of `OnGridMPC` intact while internally
organizing the model in component classes (`Parameters`, `Load`, `PV`,
`Grid`, `BESS`, `Scenarios`) similar to the structure used in `sizing`.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeIntegers,
    NonNegativeReals,
    Objective,
    Param,
    Reals,
    Set,
    SolverFactory,
    UnitInterval,
    Var,
    minimize,
    value,
)

from .utils import build_time_and_contingencies_from_params, predecessor_pairs


class Parameters:
    """Loads and prepares scalar optimization parameters from JSON."""

    def __init__(self, params_path: Path):
        self.params_path = Path(params_path)
        self.raw = self._load_raw()
        self.data = self._build_data()

    def _load_raw(self) -> Dict[str, Any]:
        with open(self.params_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for sec in ["time", "costs", "BESS", "PV", "Load", "EDS"]:
            if sec not in raw:
                raise KeyError(f"Mandatory section missing from JSON: '{sec}'")
        return raw

    def _build_data(self) -> Dict[str, Any]:
        raw = self.raw
        time = raw["time"]
        costs = raw["costs"]
        bess = raw["BESS"]
        pv = raw["PV"]
        load = raw["Load"]
        eds = raw["EDS"] or {}

        out: Dict[str, Any] = {}
        out["horizon_hours"] = int(time["horizon_hours"])
        out["timestep_1_min"] = int(time["timestep_1_min"])
        out["timestep_2_min"] = int(time["timestep_2_min"])

        out["c_shed_per_kwh"] = float(costs["c_shed_per_kwh"])
        out["c_pv_curt_per_kwh"] = float(costs["c_pv_curt_per_kwh"])
        out["tou_map"] = dict(costs.get("EDS", {}))

        if "bess_degradation_per_kwh" in costs:
            out["c_bess_deg_per_kwh"] = float(costs["bess_degradation_per_kwh"])
        else:
            rep = bess.get("replacement_cost_per_kwh", None)
            ncy = bess.get("cycle_life_full", None)
            if rep is not None and ncy is not None and float(ncy) > 0:
                out["c_bess_deg_per_kwh"] = float(rep) / (2.0 * float(ncy))
            else:
                out["c_bess_deg_per_kwh"] = 0.0

        out["P_ch_max_kw"] = float(bess["Pmax_kw"])
        out["P_dis_max_kw"] = float(bess["Pmax_kw"])
        out["E_nom_kwh"] = float(bess["Emax_kwh"])
        dod = float(bess.get("DoD_frac", 0.9))
        out["soc_min_frac"] = max(0.0, 1.0 - dod)
        out["soc_max_frac"] = float(bess.get("soc_max_frac", 1.0))
        out["eta_c"] = float(bess["eta_c"])
        out["eta_d"] = float(bess["eta_d"])
        out["R_bess_kw_per_step"] = float(bess.get("ramp_kw_per_step", 1.0))

        out["P_PV_nom_kw"] = float(pv["Pmax_kw"])
        out["P_L_nom_kw"] = float(load["Pmax_kw"])

        out["P_grid_import_cap_kw"] = float(eds.get("Pmax_kw", eds.get("Pmax", 0.0)))
        out["P_grid_export_cap_kw"] = float(abs(eds.get("Pmin", 0.0)))
        out["outage_probability_pct"] = float(eds.get("outage_probability_pct", 0.0))
        out["outage_duration_hours"] = float(eds.get("outage_duration_hours", 0.0))
        return out

    def build_time_data(self, start_dt: datetime) -> Dict[str, Any]:
        times, contingencies = build_time_and_contingencies_from_params(self.data, start_dt)
        times = list(times)
        contingencies = list(contingencies)
        trans_pairs = predecessor_pairs(times)

        dt_h_map: Dict[datetime, float] = {}
        for t0, t1 in trans_pairs:
            dt_h_map[t0] = (t1 - t0).total_seconds() / 3600.0
        dt_h_map[times[-1]] = dt_h_map[trans_pairs[-1][0]]

        tou_map = self.data["tou_map"]
        price_map = {t: float(tou_map.get(f"{t.hour:02d}:00", 0.0)) for t in times}
        return {
            "times": times,
            "contingencies": contingencies,
            "trans_pairs": trans_pairs,
            "dt_h_map": dt_h_map,
            "price_map": price_map,
            "dt_ref_h": max(1e-9, self.data["timestep_1_min"] / 60.0),
        }


class Load:
    def __init__(self, params: Parameters, relaxation: bool):
        self.relaxation = relaxation
        self.c_shed_per_kwh = float(params.data["c_shed_per_kwh"])

    def build(self, model, forecasts: Dict[str, Dict[datetime, float]]) -> None:
        model.c_shed = Param(initialize=self.c_shed_per_kwh)
        model.Load_kw = Param(model.T, initialize=lambda _, t: float(forecasts["load_kw"][t]))
        if self.relaxation:
            model.X_L = Var(model.T, model.C, domain=UnitInterval)
        else:
            model.X_L = Var(model.T, model.C, domain=UnitInterval)
            model.n_shed = Var(model.T, model.C, domain=NonNegativeIntegers, bounds=(0, 10))
            model.ShedDiscretization = Constraint(
                model.T, model.C, rule=lambda m, t, c: m.X_L[t, c] == 0.1 * m.n_shed[t, c]
            )


class PV:
    def __init__(self, params: Parameters):
        self.c_pv_curt_per_kwh = float(params.data["c_pv_curt_per_kwh"])

    def build(self, model, forecasts: Dict[str, Dict[datetime, float]]) -> None:
        model.c_pv_curt = Param(initialize=self.c_pv_curt_per_kwh)
        model.PV_kw = Param(model.T, initialize=lambda _, t: float(forecasts["pv_kw"][t]))
        model.X_PV = Var(model.T, model.C, domain=UnitInterval)


class Grid:
    def __init__(self, params: Parameters):
        self.p_imp_cap_kw = max(0.0, float(params.data["P_grid_import_cap_kw"]))
        self.p_exp_cap_kw = max(0.0, float(params.data["P_grid_export_cap_kw"]))

    def build(self, model, price_map: Dict[datetime, float]) -> None:
        model.P_imp_cap = Param(initialize=self.p_imp_cap_kw)
        model.P_exp_cap = Param(initialize=self.p_exp_cap_kw)
        model.c_grid = Param(model.T, initialize=lambda _, t: price_map[t])
        model.P_gin = Var(model.T, model.C, domain=NonNegativeReals)
        model.P_gout = Var(model.T, model.C, domain=NonNegativeReals)

        model.GridImportCap = Constraint(
            model.T,
            model.C,
            rule=lambda m, t, c: (m.P_gin[t, c] <= m.P_imp_cap)
            if (c == "c0" or t not in m.W[c])
            else Constraint.Skip,
        )
        model.GridExportCap = Constraint(
            model.T,
            model.C,
            rule=lambda m, t, c: (m.P_gout[t, c] <= m.P_exp_cap)
            if (c == "c0" or t not in m.W[c])
            else Constraint.Skip,
        )
        model.GridZeroIn = Constraint(
            model.T,
            model.C,
            rule=lambda m, t, c: (m.P_gin[t, c] == 0.0)
            if (c != "c0" and t in m.W[c])
            else Constraint.Skip,
        )
        model.GridZeroOut = Constraint(
            model.T,
            model.C,
            rule=lambda m, t, c: (m.P_gout[t, c] == 0.0)
            if (c != "c0" and t in m.W[c])
            else Constraint.Skip,
        )


class BESS:
    def __init__(self, params: Parameters):
        p = params.data
        self.c_deg_per_kwh = float(p["c_bess_deg_per_kwh"])
        self.e_nom_kwh = float(p["E_nom_kwh"])
        self.soc_min_frac = float(p["soc_min_frac"])
        self.soc_max_frac = float(p["soc_max_frac"])
        self.p_ch_max_kw = float(p["P_ch_max_kw"])
        self.p_dis_max_kw = float(p["P_dis_max_kw"])
        self.eta_c = float(p["eta_c"])
        self.eta_d = float(p["eta_d"])
        self.r_bess_kw_per_step = float(p["R_bess_kw_per_step"])

    def build(self, model, first_t: datetime, e_hat_kwh: float, p_bess_hat_kw: float, dt_ref_h: float) -> None:
        model.c_deg = Param(initialize=self.c_deg_per_kwh)
        model.E_nom = Param(initialize=self.e_nom_kwh)
        model.f_soc_min = Param(initialize=self.soc_min_frac)
        model.f_soc_max = Param(initialize=self.soc_max_frac)
        model.P_ch_max = Param(initialize=self.p_ch_max_kw)
        model.P_dis_max = Param(initialize=self.p_dis_max_kw)
        model.eta_c = Param(initialize=self.eta_c)
        model.eta_d = Param(initialize=self.eta_d)
        model.R_bess = Param(initialize=self.r_bess_kw_per_step)

        model.P_ch = Var(model.T, model.C, domain=NonNegativeReals)
        model.P_dis = Var(model.T, model.C, domain=NonNegativeReals)
        model.P_bess = Var(model.T, model.C, domain=Reals)
        model.gamma = Var(model.T, model.C, domain=Binary)
        model.E = Var(model.T, model.C, domain=NonNegativeReals)
        model.Pbess_abs = Var(model.T, model.C, domain=NonNegativeReals)

        model.AbsPos = Constraint(model.T, model.C, rule=lambda m, t, c: m.Pbess_abs[t, c] >= m.P_bess[t, c])
        model.AbsNeg = Constraint(model.T, model.C, rule=lambda m, t, c: m.Pbess_abs[t, c] >= -m.P_bess[t, c])
        model.PbessLink = Constraint(
            model.T, model.C, rule=lambda m, t, c: m.P_bess[t, c] == m.P_dis[t, c] - m.P_ch[t, c]
        )
        model.Dynamics = Constraint(
            model.TRANS,
            model.C,
            rule=lambda m, t0, t1, c: m.E[t1, c]
            == m.E[t0, c] + m.dt_h[t0] * (m.eta_c * m.P_ch[t0, c] - (1.0 / m.eta_d) * m.P_dis[t0, c]),
        )
        model.SoC_Lo = Constraint(model.T, model.C, rule=lambda m, t, c: m.E[t, c] >= m.f_soc_min * m.E_nom)
        model.SoC_Hi = Constraint(model.T, model.C, rule=lambda m, t, c: m.E[t, c] <= m.f_soc_max * m.E_nom)

        e_min = self.soc_min_frac * self.e_nom_kwh
        e_max = self.soc_max_frac * self.e_nom_kwh
        model.DischargeEnergyCap = Constraint(
            model.T,
            model.C,
            rule=lambda m, t, c: m.P_dis[t, c] <= m.eta_d * (m.E[t, c] - e_min) / m.dt_h[t],
        )
        model.ChargeEnergyCap = Constraint(
            model.T,
            model.C,
            rule=lambda m, t, c: m.P_ch[t, c] <= (e_max - m.E[t, c]) / (m.eta_c * m.dt_h[t]),
        )
        model.ChargeLimit = Constraint(
            model.T, model.C, rule=lambda m, t, c: m.P_ch[t, c] <= m.P_ch_max * m.gamma[t, c]
        )
        model.DischargeLimit = Constraint(
            model.T, model.C, rule=lambda m, t, c: m.P_dis[t, c] <= m.P_dis_max * (1 - m.gamma[t, c])
        )

        model.RampLo = Constraint(
            model.TRANS,
            model.C,
            rule=lambda m, t0, t1, c: m.P_bess[t1, c] - m.P_bess[t0, c] >= -m.R_bess * (m.dt_h[t0] / dt_ref_h),
        )
        model.RampHi = Constraint(
            model.TRANS,
            model.C,
            rule=lambda m, t0, t1, c: m.P_bess[t1, c] - m.P_bess[t0, c] <= m.R_bess * (m.dt_h[t0] / dt_ref_h),
        )

        model.Pbess_hat = Param(initialize=float(p_bess_hat_kw))
        model.RampFromPrevLo = Constraint(
            model.C,
            rule=lambda m, c: m.P_bess[first_t, c] - m.Pbess_hat >= -m.R_bess * (m.dt_h[first_t] / dt_ref_h),
        )
        model.RampFromPrevHi = Constraint(
            model.C,
            rule=lambda m, c: m.P_bess[first_t, c] - m.Pbess_hat <= m.R_bess * (m.dt_h[first_t] / dt_ref_h),
        )

        model.E_hat = Param(initialize=float(e_hat_kwh))
        model.InitialCond = Constraint(model.C, rule=lambda m, c: m.E[first_t, c] == m.E_hat)


class Scenarios:
    def __init__(self, params: Parameters):
        p = params.data
        self.outage_duration_hours = float(p["outage_duration_hours"])
        self.outage_probability_pct = float(p["outage_probability_pct"])

    def build(self, model, times: List[datetime], contingencies: List[datetime]) -> List[Any]:
        scenarios = ["c0"] + contingencies
        model.C = Set(initialize=scenarios, ordered=True)

        horizon_hours = self.outage_duration_hours
        windows: Dict[Any, List[datetime]] = {}
        for c in contingencies:
            end_c = c + timedelta(hours=horizon_hours)
            windows[c] = [t for t in times if (t >= c and t < end_c)]
        windows["c0"] = []
        model.W = Set(model.C, within=model.T, ordered=True, initialize=lambda _, c: windows[c])

        idx = {t: i for i, t in enumerate(times)}
        before: Dict[Any, List[datetime]] = {"c0": []}
        for c in contingencies:
            before[c] = times[: idx[c]]
        model.Before = Set(model.C, within=model.T, ordered=True, initialize=lambda _, c: before[c])

        out_pct = self.outage_probability_pct / 100.0
        p_each = (out_pct / float(len(contingencies))) if contingencies else 0.0
        model.piC = Param(
            model.C,
            initialize=lambda _, c: max(0.0, 1.0 - out_pct) if c == "c0" else p_each,
            within=NonNegativeReals,
        )
        sum_pi = sum(value(model.piC[c]) for c in model.C)
        assert abs(sum_pi - 1.0) < 1e-8, f"Sum of scenario probabilities is {sum_pi}, expected 1.0"
        return scenarios

    @staticmethod
    def build_non_anticipativity(model) -> None:
        model.EequalPre = Constraint(
            model.T,
            model.C,
            rule=lambda m, t, c: (m.E[t, c] == m.E[t, "c0"]) if (c != "c0" and t in m.Before[c]) else Constraint.Skip,
        )
        model.XLEequalPre = Constraint(
            model.T,
            model.C,
            rule=lambda m, t, c: (m.X_L[t, c] == m.X_L[t, "c0"])
            if (c != "c0" and t in m.Before[c])
            else Constraint.Skip,
        )
        model.XPVEequalPre = Constraint(
            model.T,
            model.C,
            rule=lambda m, t, c: (m.X_PV[t, c] == m.X_PV[t, "c0"])
            if (c != "c0" and t in m.Before[c])
            else Constraint.Skip,
        )


class OnGridMPC:
    """Public on-grid MPC class. API compatible with previous implementation."""

    def __init__(self, params_path: str = "data/parameters.json", relaxation: bool = False):
        self.params_path = Path(params_path)
        self.relaxation = relaxation
        self.param = Parameters(self.params_path)
        self.bess = BESS(self.param)
        self.load = Load(self.param, relaxation=self.relaxation)
        self.pv = PV(self.param)
        self.grid = Grid(self.param)
        self.scenario = Scenarios(self.param)

        self.model = None
        self.results = None
        self.params = self.param.data  # backwards-compatible attribute
        self._times: List[datetime] | None = None
        self._contingencies: List[datetime] | None = None
        self._scenarios: List[Any] | None = None

    def build(
        self,
        start_dt: datetime,
        forecasts: Dict[str, Dict[datetime, float]],
        E_hat_kwh: float,
        P_bess_hat_kw: float = 0.0,
    ):
        time_data = self.param.build_time_data(start_dt)
        self._times = list(time_data["times"])
        self._contingencies = list(time_data["contingencies"])

        model = ConcreteModel(name="MPC_OnGrid_Stochastic")
        model.T = Set(initialize=self._times, ordered=True)
        model.TRANS = Set(initialize=time_data["trans_pairs"], dimen=2, ordered=True)
        model.dt_h = Param(model.T, initialize=lambda _, t: float(time_data["dt_h_map"][t]))
        model.H = Param(initialize=float(self.param.data["outage_duration_hours"]))

        self._scenarios = self.scenario.build(model, self._times, self._contingencies)
        self.load.build(model, forecasts)
        self.pv.build(model, forecasts)
        self.grid.build(model, time_data["price_map"])
        self.bess.build(
            model=model,
            first_t=self._times[0],
            e_hat_kwh=E_hat_kwh,
            p_bess_hat_kw=P_bess_hat_kw,
            dt_ref_h=time_data["dt_ref_h"],
        )

        model.Balance = Constraint(
            model.T,
            model.C,
            rule=lambda m, t, c: m.PV_kw[t] * (1 - m.X_PV[t, c])
            + m.P_dis[t, c]
            - m.P_ch[t, c]
            + m.P_gin[t, c]
            - m.P_gout[t, c]
            == m.Load_kw[t] * (1 - m.X_L[t, c]),
        )

        def objective_rule(m):
            eps = 1e-12
            return sum(
                m.piC[c]
                * m.dt_h[t]
                * (
                    m.c_shed * m.Load_kw[t] * m.X_L[t, c]
                    + m.c_pv_curt * m.PV_kw[t] * m.X_PV[t, c]
                    + m.c_grid[t] * m.P_gin[t, c]
                    + m.c_deg * m.Pbess_abs[t, c]
                )
                + eps * (m.X_L[t, c] + m.X_PV[t, c])
                for t in m.T
                for c in m.C
            )

        model.Objective = Objective(rule=objective_rule, sense=minimize)
        self.scenario.build_non_anticipativity(model)

        self.model = model
        return model

    def solve(self, solver_name: str = "gurobi", tee: bool = True, **solver_opts):
        solver = SolverFactory(solver_name)
        for k, v in solver_opts.items():
            solver.options[k] = v
        self.results = solver.solve(self.model, tee=tee)
        return self.results

    def extract_first_step(self, scenario: Any = "c0") -> Dict[str, float]:
        t0 = self._times[0]
        m = self.model
        c = scenario
        return {
            "scenario": str(c),
            "P_bess_kw": value(m.P_bess[t0, c]),
            "P_ch_kw": value(m.P_ch[t0, c]),
            "P_dis_kw": value(m.P_dis[t0, c]),
            "gamma": int(round(value(m.gamma[t0, c]))),
            "X_L": value(m.X_L[t0, c]),
            "X_PV": value(m.X_PV[t0, c]),
            "E_kwh": value(m.E[t0, c]),
            "P_grid_in_kw": value(m.P_gin[t0, c]),
            "P_grid_out_kw": value(m.P_gout[t0, c]),
            "obj": value(m.Objective),
        }

    def extract_first_step_all(self) -> Dict[Any, Dict[str, float]]:
        return {c: self.extract_first_step(scenario=c) for c in (self._scenarios or ["c0"])}

    def extract_full_solution(self) -> Dict[str, Any]:
        if self.model is None or self._times is None or self._scenarios is None:
            return {}
        m = self.model
        solution: Dict[str, Any] = {"decision_time": self._times[0].isoformat(), "scenarios": {}}
        for c in self._scenarios:
            scenario_data = []
            for t in self._times:
                scenario_data.append(
                    {
                        "timestamp": t.isoformat(),
                        "P_bess_kw": value(m.P_bess[t, c]),
                        "P_load_kw": value(m.Load_kw[t]),
                        "P_pv_kw": value(m.PV_kw[t]),
                        "X_L": value(m.X_L[t, c]),
                        "X_PV": value(m.X_PV[t, c]),
                        "P_grid_in_kw": value(m.P_gin[t, c]),
                        "P_grid_out_kw": value(m.P_gout[t, c]),
                        "E_kwh": value(m.E[t, c]),
                    }
                )
            scenario_key = "c0_base_case" if c == "c0" else f"contingency_{c.isoformat()}"
            solution["scenarios"][scenario_key] = scenario_data
        return solution
