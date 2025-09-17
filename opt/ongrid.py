# -*- coding: utf-8 -*-
"""
opt/ongrid_stochastic.py

OnGridMPC (ON-GRID) with a stochastic outage component.

Key modelling choices implemented here:
1) Only import from the grid is paid (export has no cost/revenue).
2) Export capacity uses abs(Pmin) from EDS (in case Pmin is negative).
3) Grid capacity limits apply only OUTSIDE the outage window; INSIDE it, grid is forced to zero.
4) Objective regularization epsilon is applied only to fractional curtailments (X_L, X_PV),
   not to P_gin / P_gout, to avoid biasing toward curtailment.
5) Scenario probability π depends ONLY on the scenario c (outage starts at time c),
   i.e., π = π(c), not π(t,c). The whole horizon of each scenario is weighted by π(c).
6) BESS ramp at the first step is anchored to the previous-step net BESS power Pbess_hat,
   analogous to how SoC uses E_hat for the initial condition.
7) BESS degradation cost uses |P_bess| (linearized) instead of (P_ch + P_dis).
8) NEW (A): Ramp limits are proportional to the step duration Δt (rate-based),
   i.e., |ΔP| ≤ R_bess * (Δt / Δt_ref), with Δt_ref = timestep_1_min.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Reals, NonNegativeReals, UnitInterval, Binary,
    Objective, Constraint, minimize, value, SolverFactory, NonNegativeIntegers
)

from .utils import (
    predecessor_pairs,
    build_time_and_contingencies_from_params,
)


class OnGridMPC:
    def __init__(self, params_path: str = "data/parameters.json", relaxation: bool = False):
        self.params_path = Path(params_path)
        self.params = self._load_params()
        self.relaxation = relaxation
        self.model = None
        self.results = None
        self._times: List[datetime] = None
        self._contingencies: List[datetime] = None
        self._scenarios: List[Any] = None  # ["c0"] + contingencies

    # --------------------- params ---------------------
    def _load_params(self) -> Dict[str, Any]:
        with open(self.params_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for sec in ["time", "costs", "BESS", "PV", "Load", "EDS"]:
            if sec not in raw:
                raise KeyError(f"Mandatory section missing from JSON: '{sec}'")

        time  = raw["time"]
        costs = raw["costs"]
        BESS  = raw["BESS"]
        PV    = raw["PV"]
        Load  = raw["Load"]
        EDS   = raw["EDS"] or {}

        p: Dict[str, Any] = {}
        # time
        p["horizon_hours"]  = int(time["horizon_hours"])
        p["timestep_1_min"] = int(time["timestep_1_min"])
        p["timestep_2_min"] = int(time["timestep_2_min"])

        # costs
        p["c_shed_per_kwh"]    = float(costs["c_shed_per_kwh"])
        p["c_pv_curt_per_kwh"] = float(costs["c_pv_curt_per_kwh"])
        p["tou_map"]           = dict(costs.get("EDS", {}))  # "HH:00" -> price

        # BESS degradation cost ($/kWh cycled). If not given, derive from replacement and cycle life.
        if "bess_degradation_per_kwh" in costs:
            p["c_bess_deg_per_kwh"] = float(costs["bess_degradation_per_kwh"])
        else:
            rep = BESS.get("replacement_cost_per_kwh", None)
            ncy = BESS.get("cycle_life_full", None)
            if rep is not None and ncy is not None and float(ncy) > 0:
                p["c_bess_deg_per_kwh"] = float(rep) / (2.0 * float(ncy))
            else:
                p["c_bess_deg_per_kwh"] = 0.0

        # BESS
        p["P_ch_max_kw"]  = float(BESS["Pmax_kw"])
        p["P_dis_max_kw"] = float(BESS["Pmax_kw"])
        p["E_nom_kwh"]    = float(BESS["Emax_kwh"])
        DoD               = float(BESS.get("DoD_frac", 0.9))
        p["soc_min_frac"] = max(0.0, 1.0 - DoD)
        p["soc_max_frac"] = float(BESS.get("soc_max_frac", 1.0))
        p["eta_c"]        = float(BESS["eta_c"])
        p["eta_d"]        = float(BESS["eta_d"])
        p["R_bess_kw_per_step"] = float(BESS.get("ramp_kw_per_step", 1.0))

        # Informative scales
        p["P_PV_nom_kw"] = float(PV["Pmax_kw"])
        p["P_L_nom_kw"]  = float(Load["Pmax_kw"])

        # Grid + stochastic
        p["P_grid_import_cap_kw"] = float(EDS.get("Pmax_kw", EDS.get("Pmax", 0.0)))
        p["P_grid_export_cap_kw"] = float(abs(EDS.get("Pmin", 0.0)))
        p["outage_probability_pct"] = float(EDS.get("outage_probability_pct", 0.0))
        p["outage_duration_hours"]  = float(EDS.get("outage_duration_hours", 0.0))

        return p

    # --------------------- build ---------------------
    def build(self,
              start_dt: datetime,
              forecasts: Dict[str, Dict[datetime, float]],
              E_hat_kwh: float,
              P_bess_hat_kw: float = 0.0):

        p = self.params

        # Build time grid and set of contingencies (c are datetime instants)
        times, contingencies = build_time_and_contingencies_from_params(p, start_dt)
        self._times = list(times)
        self._contingencies = list(contingencies)

        m = ConcreteModel(name="MPC_OnGrid_Stochastic")
        m.T = Set(initialize=self._times, ordered=True)
        trans_pairs = predecessor_pairs(self._times)
        m.TRANS = Set(initialize=trans_pairs, dimen=2, ordered=True)

        # First step in the horizon (used by initial conditions and first-step ramp)
        first_t = self._times[0]

        # Main parameters
        m.c_shed    = Param(initialize=p["c_shed_per_kwh"])
        m.c_pv_curt = Param(initialize=p["c_pv_curt_per_kwh"])
        m.c_deg     = Param(initialize=p["c_bess_deg_per_kwh"])
        m.E_nom     = Param(initialize=p["E_nom_kwh"])
        m.f_soc_min = Param(initialize=p["soc_min_frac"])
        m.f_soc_max = Param(initialize=p["soc_max_frac"])
        m.P_ch_max  = Param(initialize=p["P_ch_max_kw"])
        m.P_dis_max = Param(initialize=p["P_dis_max_kw"])
        m.eta_c     = Param(initialize=p["eta_c"])
        m.eta_d     = Param(initialize=p["eta_d"])
        m.R_bess    = Param(initialize=p["R_bess_kw_per_step"])
        m.P_imp_cap = Param(initialize=max(0.0, p["P_grid_import_cap_kw"]))
        m.P_exp_cap = Param(initialize=max(0.0, p["P_grid_export_cap_kw"]))

        # Outage duration H (hours)
        m.H = Param(initialize=float(p["outage_duration_hours"]))

        # dt_h per step (hours)
        dt_h_map: Dict[datetime, float] = {}
        for (t0, t1) in trans_pairs:
            dt_h_map[t0] = (t1 - t0).total_seconds() / 3600.0
        dt_h_map[self._times[-1]] = dt_h_map[trans_pairs[-1][0]]
        m.dt_h = Param(m.T, initialize=lambda _, t: float(dt_h_map[t]))

        # --- NEW (A): reference step (hours) to scale ramp by time, not by step count ---
        dt_ref_h = max(1e-9, p["timestep_1_min"] / 60.0)  # e.g., 5 min -> 0.0833 h

        # Profiles
        m.Load_kw = Param(m.T, initialize=lambda _, t: float(forecasts["load_kw"][t]))
        m.PV_kw   = Param(m.T, initialize=lambda _, t: float(forecasts["pv_kw"][t]))

        # Time-of-use (import) price map c_grid[t]
        tou_map = p["tou_map"]
        price_map: Dict[datetime, float] = {}
        for t in self._times:
            key = f"{t.hour:02d}:00"
            price_map[t] = float(tou_map.get(key, 0.0))
        m.c_grid = Param(m.T, initialize=lambda _, t: price_map[t])

        # ---------- Scenarios ----------
        scenario_list = ["c0"] + self._contingencies  # c0 = no outage
        self._scenarios = scenario_list
        m.C = Set(initialize=scenario_list, ordered=True)

        # Outage windows W[c] = {t in T: c <= t < c+H}  (applied with strict end)
        H = float(p["outage_duration_hours"])
        W: Dict[Any, List[datetime]] = {}
        for c in self._contingencies:
            end_c = c + timedelta(hours=H)
            members = [t for t in self._times if (t >= c and t < end_c)]
            W[c] = members
        W["c0"] = []
        m.W = Set(m.C, within=m.T, ordered=True, initialize=lambda mm, c: W[c])

        # Times strictly before the start of the outage (for non-anticipativity)
        idx = {t: i for i, t in enumerate(self._times)}
        Before: Dict[Any, List[datetime]] = {"c0": []}
        for c in self._contingencies:
            i_c = idx[c]
            Before[c] = self._times[:i_c]
        m.Before = Set(m.C, within=m.T, ordered=True, initialize=lambda mm, c: Before[c])

        # ---------- Scenario probabilities π(c) ----------
        out_pct = float(p["outage_probability_pct"]) / 100.0
        num_c = len(self._contingencies)
        p_each = (out_pct / float(num_c)) if num_c > 0 else 0.0

        def piC_init(mm, c):
            if c == "c0":
                return max(0.0, 1.0 - out_pct)
            else:
                return p_each
        m.piC = Param(m.C, initialize=piC_init, within=NonNegativeReals)
        _sum_pi = sum(value(m.piC[c]) for c in m.C)
        assert abs(_sum_pi - 1.0) < 1e-8, f"Sum of scenario probabilities is {_sum_pi}, expected 1.0"

        # ---------- Variables ----------
        if self.relaxation:
            m.X_L = Var(m.T, m.C, domain=UnitInterval)
        else:
            m.X_L    = Var(m.T, m.C, domain=UnitInterval)
            m.n_shed = Var(m.T, m.C, domain=NonNegativeIntegers, bounds=(0, 10))
            m.ShedDiscretization = Constraint(
                m.T, m.C, rule=lambda m, t, c: m.X_L[t, c] == 0.1 * m.n_shed[t, c]
            )

        m.X_PV   = Var(m.T, m.C, domain=UnitInterval)
        m.P_ch   = Var(m.T, m.C, domain=NonNegativeReals)
        m.P_dis  = Var(m.T, m.C, domain=NonNegativeReals)
        m.P_bess = Var(m.T, m.C, domain=Reals)
        m.gamma  = Var(m.T, m.C, domain=Binary)

        # E non-negative
        m.E      = Var(m.T, m.C, domain=NonNegativeReals)

        m.P_gin  = Var(m.T, m.C, domain=NonNegativeReals)
        m.P_gout = Var(m.T, m.C, domain=NonNegativeReals)

        # Linearize |P_bess|
        m.Pbess_abs = Var(m.T, m.C, domain=NonNegativeReals)
        m.AbsPos = Constraint(m.T, m.C, rule=lambda mm, t, c: mm.Pbess_abs[t, c] >=  mm.P_bess[t, c])
        m.AbsNeg = Constraint(m.T, m.C, rule=lambda mm, t, c: mm.Pbess_abs[t, c] >= -mm.P_bess[t, c])

        # ---------- Objective ----------
        def obj_rule(mm):
            eps = 1e-12
            return sum(
                mm.piC[c] * mm.dt_h[t] * (
                    mm.c_shed    * mm.Load_kw[t] * mm.X_L[t, c] +
                    mm.c_pv_curt * mm.PV_kw[t]   * mm.X_PV[t, c] +
                    mm.c_grid[t] * mm.P_gin[t, c] +
                    mm.c_deg     * mm.Pbess_abs[t, c]
                )
                + eps * (mm.X_L[t, c] + mm.X_PV[t, c])
                for t in mm.T for c in mm.C
            )
        m.Objective = Objective(rule=obj_rule, sense=minimize)

        # ---------- Constraints ----------
        # Power balance
        m.Balance = Constraint(m.T, m.C, rule=lambda mm, t, c:
            mm.PV_kw[t] * (1 - mm.X_PV[t, c]) + mm.P_dis[t, c] - mm.P_ch[t, c] + mm.P_gin[t, c] - mm.P_gout[t, c]
            == mm.Load_kw[t] * (1 - mm.X_L[t, c])
        )

        # Link P_bess = P_dis - P_ch
        m.PbessLink = Constraint(m.T, m.C, rule=lambda mm, t, c:
            mm.P_bess[t, c] == mm.P_dis[t, c] - mm.P_ch[t, c]
        )

        # BESS dynamics
        m.Dynamics = Constraint(m.TRANS, m.C, rule=lambda mm, t0, t1, c:
            mm.E[t1, c] == mm.E[t0, c] + mm.dt_h[t0] * (mm.eta_c * mm.P_ch[t0, c] - (1.0 / mm.eta_d) * mm.P_dis[t0, c])
        )

        # SoC bounds
        m.SoC_Lo = Constraint(m.T, m.C, rule=lambda mm, t, c: mm.E[t, c] >= mm.f_soc_min * mm.E_nom)
        m.SoC_Hi = Constraint(m.T, m.C, rule=lambda mm, t, c: mm.E[t, c] <= mm.f_soc_max * mm.E_nom)

        # Energy-to-power limits
        E_min = p["soc_min_frac"] * p["E_nom_kwh"]
        E_max = p["soc_max_frac"] * p["E_nom_kwh"]
        m.DischargeEnergyCap = Constraint(m.T, m.C, rule=lambda mm, t, c:
            mm.P_dis[t, c] <= mm.eta_d * (mm.E[t, c] - E_min) / mm.dt_h[t]
        )
        m.ChargeEnergyCap = Constraint(m.T, m.C, rule=lambda mm, t, c:
            mm.P_ch[t, c] <= (E_max - mm.E[t, c]) / (mm.eta_c * mm.dt_h[t])
        )

        # Converter limits (charge vs discharge)
        m.ChargeLimit    = Constraint(m.T, m.C, rule=lambda mm, t, c: mm.P_ch[t, c] <= mm.P_ch_max * mm.gamma[t, c])
        m.DischargeLimit = Constraint(m.T, m.C, rule=lambda mm, t, c: mm.P_dis[t, c] <= mm.P_dis_max * (1 - mm.gamma[t, c]))

        # ----------------- Ramp limits (NEW A: rate-based, scaled by Δt/dt_ref) -----------------
        m.RampLo = Constraint(m.TRANS, m.C, rule=lambda mm, t0, t1, c:
            mm.P_bess[t1, c] - mm.P_bess[t0, c] >= -mm.R_bess * (mm.dt_h[t0] / dt_ref_h)
        )
        m.RampHi = Constraint(m.TRANS, m.C, rule=lambda mm, t0, t1, c:
            mm.P_bess[t1, c] - mm.P_bess[t0, c] <=  mm.R_bess * (mm.dt_h[t0] / dt_ref_h)
        )

        # NEW (A): first-step ramp vs previous-step net power (scaled by Δt of first step)
        m.Pbess_hat = Param(initialize=float(P_bess_hat_kw))
        m.RampFromPrevLo = Constraint(m.C, rule=lambda mm, c:
            mm.P_bess[first_t, c] - mm.Pbess_hat >= -mm.R_bess * (mm.dt_h[first_t] / dt_ref_h)
        )
        m.RampFromPrevHi = Constraint(m.C, rule=lambda mm, c:
            mm.P_bess[first_t, c] - mm.Pbess_hat <=  mm.R_bess * (mm.dt_h[first_t] / dt_ref_h)
        )

        # Grid caps outside outage
        m.GridImportCap = Constraint(m.T, m.C, rule=lambda mm, t, c:
            (mm.P_gin[t, c]  <= mm.P_imp_cap) if (c == "c0" or t not in mm.W[c]) else Constraint.Skip
        )
        m.GridExportCap = Constraint(m.T, m.C, rule=lambda mm, t, c:
            (mm.P_gout[t, c] <= mm.P_exp_cap) if (c == "c0" or t not in mm.W[c]) else Constraint.Skip
        )
        # Grid dead during outage
        m.GridZeroIn  = Constraint(m.T, m.C, rule=lambda mm, t, c:
            (mm.P_gin[t, c]  == 0.0) if (c != "c0" and t in mm.W[c]) else Constraint.Skip
        )
        m.GridZeroOut = Constraint(m.T, m.C, rule=lambda mm, t, c:
            (mm.P_gout[t, c] == 0.0) if (c != "c0" and t in mm.W[c]) else Constraint.Skip
        )

        # Initial SoC equal across scenarios
        m.E_hat = Param(initialize=float(E_hat_kwh))
        m.InitialCond = Constraint(m.C, rule=lambda mm, c: mm.E[first_t, c] == mm.E_hat)

        # Partial non-anticipativity BEFORE outage start
        m.EequalPre = Constraint(m.T, m.C, rule=lambda mm, t, c:
            (mm.E[t, c] == mm.E[t, "c0"]) if (c != "c0" and t in mm.Before[c]) else Constraint.Skip
        )
        m.XLEequalPre = Constraint(m.T, m.C, rule=lambda mm, t, c:
            (mm.X_L[t, c] == mm.X_L[t, "c0"]) if (c != "c0" and t in mm.Before[c]) else Constraint.Skip
        )
        m.XPVEequalPre = Constraint(m.T, m.C, rule=lambda mm, t, c:
            (mm.X_PV[t, c] == mm.X_PV[t, "c0"]) if (c != "c0" and t in mm.Before[c]) else Constraint.Skip
        )

        self.model = m
        return m

    # --------------------- solve & extract ---------------------
    def solve(self, solver_name="gurobi", tee=True, **solver_opts):
        solver = SolverFactory(solver_name)
        for k, v in solver_opts.items():
            solver.options[k] = v
        self.results = solver.solve(self.model, tee=tee)
        return self.results

    def extract_first_step(self, scenario="c0") -> Dict[str, float]:
        t0 = self._times[0]
        m = self.model
        c = scenario
        return {
            "scenario":        str(c),
            "P_bess_kw":       value(m.P_bess[t0, c]),
            "P_ch_kw":         value(m.P_ch[t0, c]),
            "P_dis_kw":        value(m.P_dis[t0, c]),
            "gamma":           int(round(value(m.gamma[t0, c]))),
            "X_L":             value(m.X_L[t0, c]),
            "X_PV":            value(m.X_PV[t0, c]),
            "E_kwh":           value(m.E[t0, c]),
            "P_grid_in_kw":    value(m.P_gin[t0, c]),
            "P_grid_out_kw":   value(m.P_gout[t0, c]),
            "obj":             value(m.Objective),
        }

    def extract_first_step_all(self) -> Dict[Any, Dict[str, float]]:
        return {c: self.extract_first_step(scenario=c) for c in (self._scenarios or ["c0"])}

    def extract_full_solution(self) -> Dict[str, Any]:
        if self.model is None or self._times is None or self._scenarios is None:
            return {}
        m = self.model
        solution = {"decision_time": self._times[0].isoformat(), "scenarios": {}}
        for c in self._scenarios:
            scenario_data = []
            for t in self._times:
                step_data = {
                    "timestamp": t.isoformat(),
                    "P_bess_kw": value(m.P_bess[t, c]),
                    "P_load_kw": value(m.Load_kw[t]),
                    "P_pv_kw": value(m.PV_kw[t]),
                    "X_L": value(m.X_L[t, c]),
                    "X_PV": value(m.X_PV[t, c]),
                    "P_grid_in_kw": value(m.P_gin[t, c]),
                    "P_grid_out_kw": value(m.P_gout[t, c]),
                    "E_kwh": value(m.E[t, c])
                }
                scenario_data.append(step_data)
            scenario_key = "c0_base_case" if c == "c0" else f"contingency_{c.isoformat()}"
            solution["scenarios"][scenario_key] = scenario_data
        return solution
