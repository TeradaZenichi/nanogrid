# -*- coding: utf-8 -*-
"""
Online stochastic operation using train clusters from sizing.

Key idea:
- Build a stochastic model over (time, scenario, contingency) for operation only.
- Use one shared control trajectory over time (P_bess, X_L, X_PV) for all scenarios.
- Solve once, cache actions internally, then serve controls online without rebuilding.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
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
    UnitInterval,
    Var,
    minimize,
    value,
)

from .utils import build_dt_vector, build_time_grid, predecessor_pairs, solve_model


# Scenario inputs (same prototypes/probabilities used by the sizing model)
PV_PROTOTYPES_CSV = "data/sizing/prototypes_pv_dtw_train.csv"
LOAD_PROTOTYPES_CSV = "data/sizing/prototypes_load_dtw_all_train.csv"
JOINT_PROB_CSV = "data/sizing/prob_joint_load_pv.csv"

SCENARIO_REQUIRED_COLS = ("group", "split", "cluster", "slot", "value")
JOINT_PROB_REQUIRED_COLS = (
    "split",
    "group_load",
    "group_pv",
    "cluster_load",
    "cluster_pv",
    "probability",
)


def _validate_scenario_df(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in SCENARIO_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


class Parameters:
    """Prepares scalar parameters for stochastic operation from an already-loaded dict."""

    def __init__(self, params: Dict[str, Any]):
        for sec in ["time", "costs", "BESS", "PV", "Load", "EDS"]:
            if sec not in params:
                raise KeyError(f"Mandatory section missing from parameters: '{sec}'")
        self.raw = params
        self.data = self._build_data()

    def _build_data(self) -> Dict[str, Any]:
        raw = self.raw
        time = raw["time"]
        costs = raw["costs"]
        bess = raw["BESS"]
        pv = raw["PV"]
        load = raw["Load"]
        eds = raw["EDS"] or {}
        sizing = raw.get("sizing", {}) or {}

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

        out["split"] = str(sizing.get("split", "train"))
        out["contingency_step_slots"] = int(sizing.get("contingency_step_slots", 4))
        out["max_contingency_starts"] = sizing.get("max_contingency_starts")
        out["include_zero_prob_scenarios"] = bool(sizing.get("include_zero_prob_scenarios", True))
        return out

    def build_time_data(self, start_dt: datetime) -> Dict[str, Any]:
        p = self.data
        dt_min = build_dt_vector(
            horizon_hours=int(p["horizon_hours"]),
            outage_duration_hours=int(p["outage_duration_hours"]),
            dt1_min=int(p["timestep_1_min"]),
            dt2_min=int(p["timestep_2_min"]),
        )
        times = build_time_grid(start_dt, dt_min)
        trans_pairs = predecessor_pairs(times)
        dt_h_map: Dict[datetime, float] = {}
        for t0, t1 in trans_pairs:
            dt_h_map[t0] = (t1 - t0).total_seconds() / 3600.0
        dt_h_map[times[-1]] = dt_h_map[trans_pairs[-1][0]]

        tou_map = p["tou_map"]
        price_map = {t: float(tou_map.get(f"{t.hour:02d}:00", 0.0)) for t in times}

        contingencies, windows, before, pi_c = self.build_contingencies(times)
        return {
            "times": times,
            "trans_pairs": trans_pairs,
            "dt_h_map": dt_h_map,
            "price_map": price_map,
            "dt_ref_h": max(1e-9, p["timestep_1_min"] / 60.0),
            "contingencies": contingencies,
            "windows": windows,
            "before": before,
            "pi_c": pi_c,
        }

    def build_contingencies(
        self, times: List[datetime]
    ) -> Tuple[List[str], Dict[str, List[datetime]], Dict[str, List[datetime]], Dict[str, float]]:
        contingencies = ["c0"]
        windows: Dict[str, List[datetime]] = {"c0": []}
        before: Dict[str, List[datetime]] = {"c0": []}

        out_prob = max(0.0, min(1.0, float(self.data["outage_probability_pct"]) / 100.0))
        out_h = max(0.0, float(self.data["outage_duration_hours"]))
        step = max(1, int(self.data.get("contingency_step_slots", 4)))
        max_starts = self.data.get("max_contingency_starts")

        if out_prob > 0.0 and out_h > 0.0 and times:
            starts_idx = list(range(0, len(times), step))
            if max_starts is not None:
                starts_idx = starts_idx[: int(max_starts)]

            for i in starts_idx:
                t0 = times[i]
                c = f"c_{i}"
                tend = t0 + timedelta(hours=out_h)
                contingencies.append(c)
                windows[c] = [t for t in times if (t >= t0 and t < tend)]
                before[c] = times[:i]

        n_out = len(contingencies) - 1
        if n_out <= 0 or out_prob <= 0.0:
            pi_c = {"c0": 1.0}
        else:
            p_each = out_prob / float(n_out)
            pi_c = {"c0": max(0.0, 1.0 - out_prob)}
            for c in contingencies:
                if c != "c0":
                    pi_c[c] = p_each

        # Guard against tiny numerical drift.
        s = sum(pi_c.values())
        if s <= 0.0:
            pi_c = {"c0": 1.0}
        else:
            for c in list(pi_c.keys()):
                pi_c[c] = float(pi_c[c]) / float(s)

        return contingencies, windows, before, pi_c


class ScenarioLibrary:
    """Loads train cluster prototypes and joint probabilities from sizing data."""

    def __init__(
        self,
        split: str = "train",
        include_zero_prob_scenarios: bool = True,
        df_pv: pd.DataFrame | None = None,
        df_load: pd.DataFrame | None = None,
        df_prob_joint: pd.DataFrame | None = None,
    ):
        self.split = str(split)
        self.include_zero_prob_scenarios = bool(include_zero_prob_scenarios)

        self.df_pv = df_pv.copy() if df_pv is not None else pd.read_csv(Path(PV_PROTOTYPES_CSV))
        self.df_load = df_load.copy() if df_load is not None else pd.read_csv(Path(LOAD_PROTOTYPES_CSV))
        self.df_prob_joint = (
            df_prob_joint.copy() if df_prob_joint is not None else pd.read_csv(Path(JOINT_PROB_CSV))
        )

    def build(self, times: List[datetime], p_l_nom_kw: float, p_pv_nom_kw: float) -> Dict[str, Any]:
        _validate_scenario_df(self.df_pv, "df_pv")
        _validate_scenario_df(self.df_load, "df_load")
        missing_joint = [c for c in JOINT_PROB_REQUIRED_COLS if c not in self.df_prob_joint.columns]
        if missing_joint:
            raise ValueError(f"df_prob_joint is missing required columns: {missing_joint}")

        pv = self.df_pv.copy()
        load = self.df_load.copy()
        joint = self.df_prob_joint.copy()

        if "split" in pv.columns:
            pv = pv[pv["split"] == self.split].copy()
        if "split" in load.columns:
            load = load[load["split"] == self.split].copy()
        if "split" in joint.columns:
            joint = joint[joint["split"] == self.split].copy()

        if pv.empty or load.empty or joint.empty:
            raise ValueError(f"No rows found for split='{self.split}' in stochastic inputs.")

        pv["cluster"] = pv["cluster"].astype(int)
        load["cluster"] = load["cluster"].astype(int)
        pv["slot"] = pv["slot"].astype(int)
        load["slot"] = load["slot"].astype(int)
        pv["value"] = pv["value"].astype(float).clip(lower=0.0)
        load["value"] = load["value"].astype(float).clip(lower=0.0)

        joint["cluster_load"] = joint["cluster_load"].astype(int)
        joint["cluster_pv"] = joint["cluster_pv"].astype(int)
        joint["probability"] = joint["probability"].astype(float)
        if not self.include_zero_prob_scenarios:
            joint = joint[joint["probability"] > 0.0].copy()
        if joint.empty:
            raise ValueError("No scenario rows available after filtering probabilities.")

        slots_load = sorted(load["slot"].unique().tolist())
        slots_pv = sorted(pv["slot"].unique().tolist())
        if not slots_load or not slots_pv:
            raise ValueError("Prototype slots are empty.")
        if len(slots_load) != len(slots_pv):
            raise ValueError(
                "Load/PV prototype slot dimensions differ. Please align files in data/sizing first."
            )

        nslots = len(slots_load)
        slot_minutes = (24.0 * 60.0) / float(nslots)

        load_map = {(int(r.cluster), int(r.slot)): float(r.value) for r in load.itertuples(index=False)}
        pv_map = {(int(r.cluster), int(r.slot)): float(r.value) for r in pv.itertuples(index=False)}

        scenarios: List[str] = []
        cluster_load_of_s: Dict[str, int] = {}
        cluster_pv_of_s: Dict[str, int] = {}
        pi_s: Dict[str, float] = {}
        for idx, row in enumerate(joint.itertuples(index=False)):
            s = f"s{idx}"
            scenarios.append(s)
            cluster_load_of_s[s] = int(row.cluster_load)
            cluster_pv_of_s[s] = int(row.cluster_pv)
            pi_s[s] = float(row.probability)

        sp = sum(pi_s.values())
        if sp <= 0.0:
            raise ValueError("Scenario probabilities sum to zero.")
        for s in scenarios:
            pi_s[s] = float(pi_s[s]) / float(sp)

        p_load_kw: Dict[Tuple[datetime, str], float] = {}
        p_pv_kw: Dict[Tuple[datetime, str], float] = {}
        for t in times:
            minute_of_day = float(t.hour * 60 + t.minute)
            slot_idx = int(minute_of_day // slot_minutes) % nslots
            slot = slots_load[slot_idx]
            for s in scenarios:
                cl = cluster_load_of_s[s]
                cp = cluster_pv_of_s[s]
                if (cl, slot) not in load_map:
                    raise ValueError(f"Missing load prototype for cluster={cl}, slot={slot}")
                if (cp, slot) not in pv_map:
                    raise ValueError(f"Missing PV prototype for cluster={cp}, slot={slot}")
                p_load_kw[(t, s)] = max(0.0, load_map[(cl, slot)] * p_l_nom_kw)
                p_pv_kw[(t, s)] = max(0.0, pv_map[(cp, slot)] * p_pv_nom_kw)

        return {
            "scenarios": scenarios,
            "cluster_load_of_s": cluster_load_of_s,
            "cluster_pv_of_s": cluster_pv_of_s,
            "pi_s": pi_s,
            "p_load_kw": p_load_kw,
            "p_pv_kw": p_pv_kw,
            "prototype_slot_count": nslots,
        }


class OnGridStochasticOperation:
    """
    Online stochastic operation model (single shared control trajectory).

    Intended use:
    1) build(...)
    2) solve(...)
    3) consume cached actions through get_control()
    """

    def __init__(
        self,
        params: Dict[str, Any],
        relaxation: bool = True,
        split: str | None = None,
        df_pv: pd.DataFrame | None = None,
        df_load: pd.DataFrame | None = None,
        df_prob_joint: pd.DataFrame | None = None,
    ):
        self.relaxation = bool(relaxation)

        self.param = Parameters(params)
        if split is None:
            split = str(self.param.data.get("split", "train"))
        self.library = ScenarioLibrary(
            split=split,
            include_zero_prob_scenarios=bool(self.param.data.get("include_zero_prob_scenarios", True)),
            df_pv=df_pv,
            df_load=df_load,
            df_prob_joint=df_prob_joint,
        )

        self.model = None
        self.results = None
        self.params = self.param.data

        self._times: List[datetime] | None = None
        self._scenarios: List[str] | None = None
        self._contingencies: List[str] | None = None
        self._meta: Dict[str, Any] = {}

        self._actions: List[Dict[str, float]] = []
        self._action_cursor: int = 0
        self._has_loaded_solution = False

    def build(
        self,
        start_dt: datetime,
        forecasts: Dict[str, Dict[datetime, float]] | None = None,
        E_hat_kwh: float = 0.0,
        P_bess_hat_kw: float = 0.0,
    ):
        # forecasts is intentionally unused here; operation is driven by train clusters.
        _ = forecasts

        tdata = self.param.build_time_data(start_dt)
        self._times = list(tdata["times"])
        self._contingencies = list(tdata["contingencies"])

        sdata = self.library.build(
            times=self._times,
            p_l_nom_kw=float(self.param.data["P_L_nom_kw"]),
            p_pv_nom_kw=float(self.param.data["P_PV_nom_kw"]),
        )
        self._scenarios = list(sdata["scenarios"])

        m = ConcreteModel(name="OnGrid_Cluster_Stochastic_Operation")
        m.T = Set(initialize=self._times, ordered=True)
        m.S = Set(initialize=self._scenarios, ordered=True)
        m.C = Set(initialize=self._contingencies, ordered=True)
        m.TRANS = Set(initialize=tdata["trans_pairs"], dimen=2, ordered=True)

        m.W = Set(m.C, within=m.T, ordered=True, initialize=lambda _, c: tdata["windows"][c])
        m.Before = Set(m.C, within=m.T, ordered=True, initialize=lambda _, c: tdata["before"][c])

        m.dt_h = Param(m.T, initialize=lambda _, t: float(tdata["dt_h_map"][t]))
        m.piS = Param(m.S, initialize=lambda _, s: float(sdata["pi_s"][s]), within=NonNegativeReals)
        m.piC = Param(m.C, initialize=lambda _, c: float(tdata["pi_c"][c]), within=NonNegativeReals)

        m.c_shed = Param(initialize=float(self.param.data["c_shed_per_kwh"]))
        m.c_pv_curt = Param(initialize=float(self.param.data["c_pv_curt_per_kwh"]))
        m.c_deg = Param(initialize=float(self.param.data["c_bess_deg_per_kwh"]))
        m.c_grid = Param(m.T, initialize=lambda _, t: float(tdata["price_map"][t]))

        m.Load_kw = Param(m.T, m.S, initialize=lambda _, t, s: float(sdata["p_load_kw"][(t, s)]))
        m.PV_kw = Param(m.T, m.S, initialize=lambda _, t, s: float(sdata["p_pv_kw"][(t, s)]))

        # Shared battery controls across all scenarios/contingencies.
        if self.relaxation:
            m.X_L = Var(m.T, m.S, m.C, domain=UnitInterval)
            m.X_PV = Var(m.T, m.S, m.C, domain=UnitInterval)
        else:
            m.X_L = Var(m.T, m.S, m.C, domain=UnitInterval)
            m.X_PV = Var(m.T, m.S, m.C, domain=UnitInterval)
            m.n_shed = Var(m.T, m.S, m.C, domain=NonNegativeIntegers, bounds=(0, 10))
            m.n_pvcurt = Var(m.T, m.S, m.C, domain=NonNegativeIntegers, bounds=(0, 10))
            m.ShedDiscretization = Constraint(
                m.T, m.S, m.C, rule=lambda _m, t, s, c: _m.X_L[t, s, c] == 0.1 * _m.n_shed[t, s, c]
            )
            m.PVCurtDiscretization = Constraint(
                m.T, m.S, m.C, rule=lambda _m, t, s, c: _m.X_PV[t, s, c] == 0.1 * _m.n_pvcurt[t, s, c]
            )

        m.P_ch = Var(m.T, domain=NonNegativeReals)
        m.P_dis = Var(m.T, domain=NonNegativeReals)
        m.P_bess = Var(m.T, domain=Reals)
        m.E = Var(m.T, domain=NonNegativeReals)
        m.Pbess_abs = Var(m.T, domain=NonNegativeReals)

        m.P_gin = Var(m.T, m.S, m.C, domain=NonNegativeReals)
        m.P_gout = Var(m.T, m.S, m.C, domain=NonNegativeReals)

        m.P_imp_cap = Param(initialize=max(0.0, float(self.param.data["P_grid_import_cap_kw"])))
        m.P_exp_cap = Param(initialize=max(0.0, float(self.param.data["P_grid_export_cap_kw"])))

        m.E_nom = Param(initialize=float(self.param.data["E_nom_kwh"]))
        m.f_soc_min = Param(initialize=float(self.param.data["soc_min_frac"]))
        m.f_soc_max = Param(initialize=float(self.param.data["soc_max_frac"]))
        m.P_ch_max = Param(initialize=float(self.param.data["P_ch_max_kw"]))
        m.P_dis_max = Param(initialize=float(self.param.data["P_dis_max_kw"]))
        m.eta_c = Param(initialize=float(self.param.data["eta_c"]))
        m.eta_d = Param(initialize=float(self.param.data["eta_d"]))
        m.R_bess = Param(initialize=float(self.param.data["R_bess_kw_per_step"]))

        # BESS constraints
        m.AbsPos = Constraint(m.T, rule=lambda _m, t: _m.Pbess_abs[t] >= _m.P_bess[t])
        m.AbsNeg = Constraint(m.T, rule=lambda _m, t: _m.Pbess_abs[t] >= -_m.P_bess[t])
        m.PbessLink = Constraint(m.T, rule=lambda _m, t: _m.P_bess[t] == _m.P_dis[t] - _m.P_ch[t])
        m.Dynamics = Constraint(
            m.TRANS,
            rule=lambda _m, t0, t1: _m.E[t1]
            == _m.E[t0] + _m.dt_h[t0] * (_m.eta_c * _m.P_ch[t0] - (1.0 / _m.eta_d) * _m.P_dis[t0]),
        )
        m.SoC_Lo = Constraint(m.T, rule=lambda _m, t: _m.E[t] >= _m.f_soc_min * _m.E_nom)
        m.SoC_Hi = Constraint(m.T, rule=lambda _m, t: _m.E[t] <= _m.f_soc_max * _m.E_nom)

        e_min = float(self.param.data["soc_min_frac"]) * float(self.param.data["E_nom_kwh"])
        e_max = float(self.param.data["soc_max_frac"]) * float(self.param.data["E_nom_kwh"])
        m.DischargeEnergyCap = Constraint(
            m.T, rule=lambda _m, t: _m.P_dis[t] <= _m.eta_d * (_m.E[t] - e_min) / _m.dt_h[t]
        )
        m.ChargeEnergyCap = Constraint(
            m.T, rule=lambda _m, t: _m.P_ch[t] <= (e_max - _m.E[t]) / (_m.eta_c * _m.dt_h[t])
        )
        if self.relaxation:
            # Pozo et al. Extn-LP (Formulation 5), same as the sizing model:
            # affine coupling P_ch + P_dis <= P_max replaces the binary mode.
            m.ChargeLimit = Constraint(m.T, rule=lambda _m, t: _m.P_ch[t] <= _m.P_ch_max)
            m.DischargeLimit = Constraint(m.T, rule=lambda _m, t: _m.P_dis[t] <= _m.P_dis_max)
            m.RelaxCoupling = Constraint(
                m.T, rule=lambda _m, t: _m.P_dis[t] <= _m.P_dis_max - _m.P_ch[t]
            )
        else:
            m.gamma = Var(m.T, domain=Binary)
            m.ChargeLimit = Constraint(m.T, rule=lambda _m, t: _m.P_ch[t] <= _m.P_ch_max * _m.gamma[t])
            m.DischargeLimit = Constraint(
                m.T, rule=lambda _m, t: _m.P_dis[t] <= _m.P_dis_max * (1 - _m.gamma[t])
            )

        dt_ref_h = float(tdata["dt_ref_h"])
        m.RampLo = Constraint(
            m.TRANS,
            rule=lambda _m, t0, t1: _m.P_bess[t1] - _m.P_bess[t0] >= -_m.R_bess * (_m.dt_h[t0] / dt_ref_h),
        )
        m.RampHi = Constraint(
            m.TRANS,
            rule=lambda _m, t0, t1: _m.P_bess[t1] - _m.P_bess[t0] <= _m.R_bess * (_m.dt_h[t0] / dt_ref_h),
        )

        first_t = self._times[0]
        m.Pbess_hat = Param(initialize=float(P_bess_hat_kw))
        m.RampFromPrevLo = Constraint(
            rule=lambda _m: _m.P_bess[first_t] - _m.Pbess_hat >= -_m.R_bess * (_m.dt_h[first_t] / dt_ref_h)
        )
        m.RampFromPrevHi = Constraint(
            rule=lambda _m: _m.P_bess[first_t] - _m.Pbess_hat <= _m.R_bess * (_m.dt_h[first_t] / dt_ref_h)
        )
        m.E_hat = Param(initialize=float(E_hat_kwh))
        m.InitialCond = Constraint(rule=lambda _m: _m.E[first_t] == _m.E_hat)

        # Grid and outage constraints
        m.GridImportCap = Constraint(
            m.T,
            m.S,
            m.C,
            rule=lambda _m, t, s, c: (_m.P_gin[t, s, c] <= _m.P_imp_cap)
            if (c == "c0" or t not in _m.W[c])
            else Constraint.Skip,
        )
        m.GridExportCap = Constraint(
            m.T,
            m.S,
            m.C,
            rule=lambda _m, t, s, c: (_m.P_gout[t, s, c] <= _m.P_exp_cap)
            if (c == "c0" or t not in _m.W[c])
            else Constraint.Skip,
        )
        m.GridZeroIn = Constraint(
            m.T,
            m.S,
            m.C,
            rule=lambda _m, t, s, c: (_m.P_gin[t, s, c] == 0.0)
            if (c != "c0" and t in _m.W[c])
            else Constraint.Skip,
        )
        m.GridZeroOut = Constraint(
            m.T,
            m.S,
            m.C,
            rule=lambda _m, t, s, c: (_m.P_gout[t, s, c] == 0.0)
            if (c != "c0" and t in _m.W[c])
            else Constraint.Skip,
        )

        # Non-anticipativity of recourse before outage realization.
        m.NonAnt_xl = Constraint(
            m.T,
            m.S,
            m.C,
            rule=lambda _m, t, s, c: (_m.X_L[t, s, c] == _m.X_L[t, s, "c0"])
            if (c != "c0" and t in _m.Before[c])
            else Constraint.Skip,
        )
        m.NonAnt_xpv = Constraint(
            m.T,
            m.S,
            m.C,
            rule=lambda _m, t, s, c: (_m.X_PV[t, s, c] == _m.X_PV[t, s, "c0"])
            if (c != "c0" and t in _m.Before[c])
            else Constraint.Skip,
        )
        m.NonAnt_gin = Constraint(
            m.T,
            m.S,
            m.C,
            rule=lambda _m, t, s, c: (_m.P_gin[t, s, c] == _m.P_gin[t, s, "c0"])
            if (c != "c0" and t in _m.Before[c])
            else Constraint.Skip,
        )
        m.NonAnt_gout = Constraint(
            m.T,
            m.S,
            m.C,
            rule=lambda _m, t, s, c: (_m.P_gout[t, s, c] == _m.P_gout[t, s, "c0"])
            if (c != "c0" and t in _m.Before[c])
            else Constraint.Skip,
        )

        # Balance for all scenario/contingency combinations with shared controls.
        m.Balance = Constraint(
            m.T,
            m.S,
            m.C,
            rule=lambda _m, t, s, c: _m.PV_kw[t, s] * (1 - _m.X_PV[t, s, c])
            + _m.P_dis[t]
            - _m.P_ch[t]
            + _m.P_gin[t, s, c]
            - _m.P_gout[t, s, c]
            == _m.Load_kw[t, s] * (1 - _m.X_L[t, s, c]),
        )

        def objective_rule(_m):
            eps = 1e-12
            return sum(
                _m.piS[s]
                * _m.piC[c]
                * _m.dt_h[t]
                * (
                    _m.c_shed * _m.Load_kw[t, s] * _m.X_L[t, s, c]
                    + _m.c_pv_curt * _m.PV_kw[t, s] * _m.X_PV[t, s, c]
                    + _m.c_grid[t] * _m.P_gin[t, s, c]
                    + _m.c_deg * _m.Pbess_abs[t]
                )
                + eps * (_m.X_L[t, s, c] + _m.X_PV[t, s, c])
                for t in _m.T
                for s in _m.S
                for c in _m.C
            )

        m.Objective = Objective(rule=objective_rule, sense=minimize)

        self.model = m
        self._actions = []
        self._action_cursor = 0
        self._has_loaded_solution = False
        self._meta = {
            "split": self.library.split,
            "cluster_load_of_s": sdata["cluster_load_of_s"],
            "cluster_pv_of_s": sdata["cluster_pv_of_s"],
            "prototype_slot_count": sdata["prototype_slot_count"],
            "contingencies": list(self._contingencies),
        }
        return m

    def solve(self, tee: bool = False,
              time_limit: float | None = 1200,
              threads: int | None = None,
              mip_gap: float | None = 0.01,
              solver_name: str | None = None):
        """Solve with Gurobi if licensed, else HiGHS (see opt.utils.solve_model)."""
        if self.model is None:
            raise RuntimeError("Build the model before solving.")

        self.results = solve_model(
            self.model, tee=tee, time_limit=time_limit, threads=threads,
            mip_gap=mip_gap, solver_name=solver_name, load_solutions=False,
        )
        status = str(self.results.solver.status).lower()
        term = str(self.results.solver.termination_condition).lower()
        self._has_loaded_solution = status == "ok" and term in {"optimal", "locallyoptimal", "feasible"}
        if self._has_loaded_solution:
            self.model.solutions.load_from(self.results)
            self._cache_actions()
        else:
            self._actions = []
            self._action_cursor = 0
        return self.results

    def _cache_actions(self) -> None:
        if self.model is None or not self._has_loaded_solution or not self._times:
            self._actions = []
            self._action_cursor = 0
            return

        m = self.model
        self._actions = []
        for t in self._times:
            expected_gin = 0.0
            expected_gout = 0.0
            expected_xl = 0.0
            expected_xpv = 0.0
            for s in m.S:
                for c in m.C:
                    w = float(value(m.piS[s])) * float(value(m.piC[c]))
                    expected_gin += w * float(value(m.P_gin[t, s, c]))
                    expected_gout += w * float(value(m.P_gout[t, s, c]))
                    expected_xl += w * float(value(m.X_L[t, s, c]))
                    expected_xpv += w * float(value(m.X_PV[t, s, c]))

            self._actions.append(
                {
                    "timestamp": t.isoformat(),
                    "P_bess_kw": float(value(m.P_bess[t])),
                    "P_ch_kw": float(value(m.P_ch[t])),
                    "P_dis_kw": float(value(m.P_dis[t])),
                    "gamma": (int(round(float(value(m.gamma[t])))) if hasattr(m, "gamma")
                              else int(float(value(m.P_ch[t])) > 1e-6)),
                    "X_L": float(expected_xl),
                    "X_PV": float(expected_xpv),
                    "X_L_expected": float(expected_xl),
                    "X_PV_expected": float(expected_xpv),
                    "E_kwh": float(value(m.E[t])),
                    "P_grid_in_kw_expected": float(expected_gin),
                    "P_grid_out_kw_expected": float(expected_gout),
                    "obj": float(value(m.Objective)),
                }
            )
        self._action_cursor = 0

    def reset_cursor(self) -> None:
        self._action_cursor = 0

    def get_control(self, step: int | None = None, advance: bool = True) -> Dict[str, float]:
        if not self._actions:
            return {}
        if step is None:
            idx = self._action_cursor
            if idx >= len(self._actions):
                idx = len(self._actions) - 1
            action = dict(self._actions[idx])
            if advance and self._action_cursor < len(self._actions):
                self._action_cursor += 1
            return action

        idx = max(0, min(int(step), len(self._actions) - 1))
        return dict(self._actions[idx])

    def extract_first_step(self, scenario: Any | None = None, contingency: str = "c0") -> Dict[str, float]:
        if self.model is None or not self._times or not self._scenarios:
            return {}
        if not self._has_loaded_solution:
            return {}

        m = self.model
        t0 = self._times[0]
        s = str(scenario) if scenario is not None else self._scenarios[0]
        c = str(contingency)
        if s not in list(m.S):
            s = self._scenarios[0]
        if c not in list(m.C):
            c = "c0"
        return {
            "scenario": s,
            "contingency": c,
            "P_bess_kw": float(value(m.P_bess[t0])),
            "P_ch_kw": float(value(m.P_ch[t0])),
            "P_dis_kw": float(value(m.P_dis[t0])),
            "gamma": (int(round(float(value(m.gamma[t0])))) if hasattr(m, "gamma")
                      else int(float(value(m.P_ch[t0])) > 1e-6)),
            "X_L": float(value(m.X_L[t0, s, c])),
            "X_PV": float(value(m.X_PV[t0, s, c])),
            "E_kwh": float(value(m.E[t0])),
            "P_grid_in_kw": float(value(m.P_gin[t0, s, c])),
            "P_grid_out_kw": float(value(m.P_gout[t0, s, c])),
            "obj": float(value(m.Objective)),
        }

    def extract_first_step_all(self) -> Dict[str, Dict[str, float]]:
        if self.model is None or not self._scenarios or not self._contingencies or not self._has_loaded_solution:
            return {}
        out: Dict[str, Dict[str, float]] = {}
        for s in self._scenarios:
            for c in self._contingencies:
                k = f"{s}|{c}"
                out[k] = self.extract_first_step(scenario=s, contingency=c)
        return out

    def extract_full_solution(self) -> Dict[str, Any]:
        if (
            self.model is None
            or self._times is None
            or self._scenarios is None
            or self._contingencies is None
            or not self._has_loaded_solution
        ):
            return {}

        m = self.model
        solution: Dict[str, Any] = {
            "decision_time": self._times[0].isoformat(),
            "actions_cached": list(self._actions),
            "metadata": dict(self._meta),
            "scenarios": {},
        }
        for s in self._scenarios:
            scenario_rows = {}
            for c in self._contingencies:
                rows = []
                for t in self._times:
                    rows.append(
                        {
                            "timestamp": t.isoformat(),
                            "P_bess_kw": float(value(m.P_bess[t])),
                            "X_L": float(value(m.X_L[t, s, c])),
                            "X_PV": float(value(m.X_PV[t, s, c])),
                            "E_kwh": float(value(m.E[t])),
                            "P_load_kw": float(value(m.Load_kw[t, s])),
                            "P_pv_kw": float(value(m.PV_kw[t, s])),
                            "P_grid_in_kw": float(value(m.P_gin[t, s, c])),
                            "P_grid_out_kw": float(value(m.P_gout[t, s, c])),
                        }
                    )
                scenario_rows[c] = rows
            solution["scenarios"][s] = scenario_rows
        return solution
