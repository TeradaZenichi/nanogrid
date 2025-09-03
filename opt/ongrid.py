# -*- coding: utf-8 -*-
"""
opt/ongrid_stochastic.py

OnGridMPC (ON-GRID) com componente estocástica de outage.

Simplificação: em cada cenário c != "c0", tanto importação quanto exportação são zeradas
para todo t que satisfaz c <= t <= c + timedelta(hours=H), onde
H = params["outage_duration_hours"] (em horas).
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Reals, NonNegativeReals, UnitInterval, Binary,
    Objective, Constraint, minimize, value, SolverFactory
)

from .utils import (
    predecessor_pairs,
    build_time_and_contingencies_from_params,
)


class OnGridMPC:
    def __init__(self, params_path: str = "data/parameters.json"):
        self.params_path = Path(params_path)
        self.params = self._load_params()
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
                raise KeyError(f"Seção obrigatória ausente no JSON: '{sec}'")

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
        p["tou_map"]           = dict(costs.get("EDS", {}))  # "HH:00" -> preço

        # Degradação da BESS
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

        # Escalas informativas
        p["P_PV_nom_kw"] = float(PV["Pmax_kw"])
        p["P_L_nom_kw"]  = float(Load["Pmax_kw"])

        # Rede + estocástico
        p["P_grid_import_cap_kw"] = float(EDS.get("Pmax_kw", EDS.get("Pmax", 0.0)))
        p["P_grid_export_cap_kw"] = float(EDS.get("Pmin", 0.0))
        p["outage_probability_pct"] = float(EDS.get("outage_probability_pct", 0.0))
        p["outage_duration_hours"]  = float(EDS.get("outage_duration_hours", 0.0))

        return p

    # --------------------- build ---------------------
    def build(self,
              start_dt: datetime,
              forecasts: Dict[str, Dict[datetime, float]],
              E_hat_kwh: float):
        
        p = self.params

        # Monta grade temporal e conjunto de contingências
        times, contingencies = build_time_and_contingencies_from_params(p, start_dt)
        self._times = list(times)
        self._contingencies = list(contingencies)

        m = ConcreteModel(name="MPC_OnGrid_Stochastic")
        m.T = Set(initialize=self._times, ordered=True)
        trans_pairs = predecessor_pairs(self._times)
        m.TRANS = Set(initialize=trans_pairs, dimen=2, ordered=True)

        # Parâmetros principais
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

        # Duração do outage (H)
        m.H = Param(initialize=float(p["outage_duration_hours"]))

        # dt_h por passo
        dt_h_map: Dict[datetime, float] = {}
        for (t0, t1) in trans_pairs:
            dt_h_map[t0] = (t1 - t0).total_seconds() / 3600.0
        dt_h_map[self._times[-1]] = dt_h_map[trans_pairs[-1][0]]
        m.dt_h = Param(m.T, initialize=lambda _, t: float(dt_h_map[t]))

        # profiles
        m.Load_kw = Param(m.T, initialize=lambda _, t: float(forecasts["load_kw"][t]))
        m.PV_kw   = Param(m.T, initialize=lambda _, t: float(forecasts["pv_kw"][t]))

        # TOU price map
        tou_map = p["tou_map"]
        price_map: Dict[datetime, float] = {}
        for t in self._times:
            key = f"{t.hour:02d}:00"
            price_map[t] = float(tou_map.get(key, 0.0))
        m.c_grid = Param(m.T, initialize=lambda _, t: price_map[t])

        # ---------- Scenarios ----------
        scenario_list = ["c0"] + self._contingencies
        self._scenarios = scenario_list
        m.C = Set(initialize=scenario_list, ordered=True)

        H = float(p["outage_duration_hours"])
        W: Dict[Any, List[datetime]] = {}
        for c in self._contingencies:
            end_c = c + timedelta(hours=H)
            members = [t for t in self._times if (t >= c and t <= end_c)]
            W[c] = members
        W["c0"] = []
        m.W = Set(m.C, within=m.T, ordered=True, initialize=lambda mm, c: W[c])

        # Before[c] = {t in T: t < c}
        idx = {t: i for i, t in enumerate(self._times)}
        Before: Dict[Any, List[datetime]] = {"c0": []}
        for c in self._contingencies:
            i_c = idx[c]
            Before[c] = self._times[:i_c]
        m.Before = Set(m.C, within=m.T, ordered=True, initialize=lambda mm, c: Before[c])

        # Probabilidades π(t,c)
        out_pct = float(p["outage_probability_pct"]) / 100.0
        dt_min_map = {t: m.dt_h[t] * 60.0 for t in self._times}
        p_t = {t: out_pct * (value(dt_min_map[t]) / 1440.0) for t in self._times}

        def pi_init(mm, t, c):
            if c == "c0":
                s = 0.0
                for cc in self._contingencies:
                    if t in W[cc]:
                        s += p_t[t]
                return max(0.0, 1.0 - s)
            else:
                return p_t[t] if (c in self._contingencies and t in W[c]) else 0.0

        m.pi = Param(m.T, m.C, initialize=pi_init, within=NonNegativeReals)

        # ---------- Variables
        m.X_L    = Var(m.T, m.C, domain=UnitInterval)
        m.X_PV   = Var(m.T, m.C, domain=UnitInterval)
        m.P_ch   = Var(m.T, m.C, domain=NonNegativeReals)
        m.P_dis  = Var(m.T, m.C, domain=NonNegativeReals)
        m.P_bess = Var(m.T, m.C, domain=Reals)
        m.gamma  = Var(m.T, m.C, domain=Binary)
        m.E      = Var(m.T, m.C, domain=Reals)
        m.P_gin  = Var(m.T, m.C, domain=NonNegativeReals)  # import
        m.P_gout = Var(m.T, m.C, domain=NonNegativeReals)  # export

        # ---------- Objective
        def obj_rule(mm):
            eps = 1e-12
            return sum(
                mm.pi[t, c] * mm.dt_h[t] * (
                    mm.c_shed * mm.Load_kw[t] * mm.X_L[t, c] +
                    mm.c_pv_curt * mm.PV_kw[t] * mm.X_PV[t, c] +
                    mm.c_grid[t] * mm.P_gin[t, c] +
                    mm.c_deg * (mm.P_ch[t, c] + mm.P_dis[t, c])
                )
                + eps * (mm.X_L[t, c] + mm.X_PV[t, c] + mm.P_gin[t, c] + mm.P_gout[t, c])
                for t in mm.T for c in mm.C
            )
        m.Objective = Objective(rule=obj_rule, sense=minimize)

        # ---------- Constraints
        m.Balance = Constraint(m.T, m.C, rule=lambda mm, t, c:
            mm.PV_kw[t] * (1 - mm.X_PV[t, c]) + mm.P_dis[t, c] - mm.P_ch[t, c] + mm.P_gin[t, c] - mm.P_gout[t, c]
            == mm.Load_kw[t] * (1 - mm.X_L[t, c])
        )

        m.PbessLink = Constraint(m.T, m.C, rule=lambda mm, t, c:
            mm.P_bess[t, c] == mm.P_dis[t, c] - mm.P_ch[t, c]
        )

        m.Dynamics = Constraint(m.TRANS, m.C, rule=lambda mm, t0, t1, c:
            mm.E[t1, c] == mm.E[t0, c] + mm.dt_h[t0] * (mm.eta_c * mm.P_ch[t0, c] - (1.0 / mm.eta_d) * mm.P_dis[t0, c])
        )

        m.SoC_Lo = Constraint(m.T, m.C, rule=lambda mm, t, c: mm.E[t, c] >= mm.f_soc_min * mm.E_nom)
        m.SoC_Hi = Constraint(m.T, m.C, rule=lambda mm, t, c: mm.E[t, c] <= mm.f_soc_max * mm.E_nom)

        E_min = p["soc_min_frac"] * p["E_nom_kwh"]
        E_max = p["soc_max_frac"] * p["E_nom_kwh"]
        m.DischargeEnergyCap = Constraint(m.T, m.C, rule=lambda mm, t, c:
            mm.P_dis[t, c] <= mm.eta_d * (mm.E[t, c] - E_min) / mm.dt_h[t]
        )
        m.ChargeEnergyCap = Constraint(m.T, m.C, rule=lambda mm, t, c:
            mm.P_ch[t, c] <= (E_max - mm.E[t, c]) / (mm.eta_c * mm.dt_h[t]
        ))

        m.ChargeLimit    = Constraint(m.T, m.C, rule=lambda mm, t, c: mm.P_ch[t, c] <= mm.P_ch_max * mm.gamma[t, c])
        m.DischargeLimit = Constraint(m.T, m.C, rule=lambda mm, t, c: mm.P_dis[t, c] <= mm.P_dis_max * (1 - mm.gamma[t, c]))

        m.RampLo = Constraint(m.TRANS, m.C, rule=lambda mm, t0, t1, c: mm.P_bess[t1, c] - mm.P_bess[t0, c] >= -mm.R_bess)
        m.RampHi = Constraint(m.TRANS, m.C, rule=lambda mm, t0, t1, c: mm.P_bess[t1, c] - mm.P_bess[t0, c] <=  mm.R_bess)

        m.GridImportCap = Constraint(m.T, m.C, rule=lambda mm, t, c:
            (mm.P_gin[t, c] <= mm.P_imp_cap) if (c == "c0" or (c in mm.T and t not in mm.W[c])) else Constraint.Skip
        )
        m.GridExportCap = Constraint(m.T, m.C, rule=lambda mm, t, c:
            (mm.P_gout[t, c] <= mm.P_exp_cap) if (c == "c0" or (c in mm.T and t not in mm.W[c])) else Constraint.Skip
        )

        m.GridZeroIn  = Constraint(m.T, m.C, rule=lambda mm, t, c:
            (mm.P_gin[t, c]  == 0.0) if (c != "c0" and t in mm.W[c]) else Constraint.Skip
        )
        m.GridZeroOut = Constraint(m.T, m.C, rule=lambda mm, t, c:
            (mm.P_gout[t, c] == 0.0) if (c != "c0" and t in mm.W[c]) else Constraint.Skip
        )

        first_t = self._times[0]
        m.E_hat = Param(initialize=float(E_hat_kwh))
        m.InitialCond = Constraint(m.C, rule=lambda mm, c: mm.E[first_t, c] == mm.E_hat)

        m.EequalPre = Constraint(m.T, m.C, rule=lambda mm, t, c:
            (mm.E[t, c] == mm.E[t, "c0"]) if (c != "c0" and t in mm.Before[c]) else Constraint.Skip
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
