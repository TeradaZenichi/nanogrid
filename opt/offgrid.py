# -*- coding: utf-8 -*-
"""
opt/offgrid.py
Classe OffGridMPC isolada para ser importada em main_offgrid.py
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Reals, NonNegativeReals, UnitInterval, Binary,
    Objective, Constraint, minimize, value, SolverFactory
)

from .utils import predecessor_pairs, build_time_and_contingencies_from_params


class OffGridMPC:
    def __init__(self, params_path: str = "data/parameters.json"):
        self.params_path = Path(params_path)
        self.params = self._load_params()
        self.model = None
        self.results = None

    def _load_params(self) -> Dict[str, Any]:
        with open(self.params_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # checagens mínimas
        for sec in ["time", "costs", "BESS", "PV", "Load"]:
            if sec not in raw or not isinstance(raw[sec], dict):
                raise KeyError(f"Seção obrigatória ausente/no formato errado no JSON: '{sec}'")

        time  = raw["time"]
        costs = raw["costs"]
        BESS  = raw["BESS"]
        PV    = raw["PV"]
        Load  = raw["Load"]
        EDS   = raw.get("EDS", {}) if isinstance(raw.get("EDS", {}), dict) else {}

        p: Dict[str, Any] = {}

        # Tempo (o GridEnv monta 'times'; aqui só guardamos para referência)
        p["horizon_hours"]  = int(time["horizon_hours"])
        p["timestep_1_min"] = int(time["timestep_1_min"])
        p["timestep_2_min"] = int(time["timestep_2_min"])

        # Custos
        p["c_shed_per_kwh"]    = float(costs["c_shed_per_kwh"])
        p["c_pv_curt_per_kwh"] = float(costs["c_pv_curt_per_kwh"])

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

        # Escalas de planta (aceita Pmax_kw ou Pmax)
        P_PV_max  = PV.get("Pmax_kw", PV.get("Pmax", None))
        P_Load_max = Load.get("Pmax_kw", Load.get("Pmax", None))
        if P_PV_max is None or P_Load_max is None:
            raise KeyError("PV.Pmax_kw/PV.Pmax ou Load.Pmax_kw/Load.Pmax ausentes no JSON.")
        p["P_PV_nom_kw"] = float(P_PV_max)
        p["P_L_nom_kw"]  = float(P_Load_max)

        # Fonte auxiliar opcional (ex.: EDS/gerador)
        p["P_EDS_max_kw"] = float(EDS.get("Pmax_kw", EDS.get("Pmax", 0.0))) if EDS else 0.0
        p["use_EDS"]      = bool(EDS.get("enabled", False)) if EDS else False

        return p

    def build(self, start_dt: datetime, forecasts: Dict[str, Dict[datetime, float]], E_hat_kwh: float):
        p = self.params

        times, _ = build_time_and_contingencies_from_params(p, start_dt)
        self.times = times
        model = ConcreteModel(name="MPC_OffGrid")
        model.T = Set(initialize=times, ordered=True)
        trans_pairs = predecessor_pairs(times)
        model.TRANS = Set(initialize=trans_pairs, dimen=2, ordered=True)

        model.c_shed    = Param(initialize=p["c_shed_per_kwh"])
        model.c_pv_curt = Param(initialize=p["c_pv_curt_per_kwh"])
        model.E_nom     = Param(initialize=p["E_nom_kwh"])
        model.f_soc_min = Param(initialize=p["soc_min_frac"])
        model.f_soc_max = Param(initialize=p["soc_max_frac"])
        model.P_ch_max  = Param(initialize=p["P_ch_max_kw"])
        model.P_dis_max = Param(initialize=p["P_dis_max_kw"])
        model.eta_c     = Param(initialize=p["eta_c"])
        model.eta_d     = Param(initialize=p["eta_d"])
        model.R_bess    = Param(initialize=p["R_bess_kw_per_step"])

        # dt_h por passo
        dt_h_map: Dict[datetime, float] = {}
        for (t_prev, t_curr) in trans_pairs:
            dt_h_map[t_prev] = (t_curr - t_prev).total_seconds() / 3600.0
        dt_h_map[times[-1]] = dt_h_map[trans_pairs[-1][0]]
        model.dt_h = Param(model.T, initialize=lambda m, t: float(dt_h_map[t]))

        # perfis (kW)
        model.Load_kw = Param(model.T, initialize=lambda m, t: float(forecasts['load_kw'][t]))
        model.PV_kw   = Param(model.T, initialize=lambda m, t: float(forecasts['pv_kw'][t]))

        # variáveis
        model.X_L    = Var(model.T, domain=UnitInterval)
        model.X_PV   = Var(model.T, domain=UnitInterval)
        model.P_ch   = Var(model.T, domain=NonNegativeReals)
        model.P_dis  = Var(model.T, domain=NonNegativeReals)
        model.P_bess = Var(model.T, domain=Reals)
        model.gamma  = Var(model.T, domain=Binary)
        model.E      = Var(model.T, domain=Reals)

        # opcional: fonte auxiliar limitada
        if p.get("use_EDS", False) and p.get("P_EDS_max_kw", 0.0) > 0.0:
            model.P_EDS_ub = Param(initialize=p["P_EDS_max_kw"])
            model.P_EDS    = Var(model.T, domain=NonNegativeReals)
            model.EDS_Limit = Constraint(model.T, rule=lambda m, t: m.P_EDS[t] <= m.P_EDS_ub)
        else:
            model.P_EDS = None

        # objetivo
        def obj_rule(model):
            eps = 1e-12
            return sum(
                model.dt_h[t] * (
                    model.c_shed * model.Load_kw[t] * model.X_L[t] +
                    model.c_pv_curt * model.PV_kw[t] * model.X_PV[t]
                ) + eps * (model.X_L[t] + model.X_PV[t])
                for t in model.T
            )
        model.Objective = Objective(rule=obj_rule, sense=minimize)

        # balanço de potência
        def balance_rule(model, t):
            lhs = model.PV_kw[t] * (1 - model.X_PV[t]) + model.P_bess[t]
            if model.P_EDS is not None:
                lhs += model.P_EDS[t]
            return lhs == model.Load_kw[t] * (1 - model.X_L[t])
        model.Balance = Constraint(model.T, rule=balance_rule)

        # ligações e dinâmica
        model.PbessLink = Constraint(model.T, rule=lambda m, t: m.P_bess[t] == m.P_dis[t] - m.P_ch[t])
        model.Dynamics = Constraint(
            model.TRANS,
            rule=lambda m, t0, t1: m.E[t1] == m.E[t0] + m.dt_h[t0] * (
                m.eta_c * m.P_ch[t0] - (1.0 / m.eta_d) * m.P_dis[t0]
            )
        )

        # limites energéticos por passo (caps de janela)
        E_min = p["soc_min_frac"] * p["E_nom_kwh"]
        E_max = p["soc_max_frac"] * p["E_nom_kwh"]
        model.DischargeEnergyCap = Constraint(model.T, rule=lambda m, t: m.P_dis[t] <= m.eta_d * (m.E[t] - E_min) / m.dt_h[t])
        model.ChargeEnergyCap    = Constraint(model.T, rule=lambda m, t: m.P_ch[t] <= (E_max - m.E[t]) / (m.eta_c * m.dt_h[t]))

        # condição inicial
        first_t = times[0]
        model.E_hat = Param(initialize=float(E_hat_kwh))
        model.InitialCond = Constraint(expr=model.E[first_t] == model.E_hat)

        # limites de SoC
        model.SoC_Lo = Constraint(model.T, rule=lambda m, t: m.E[t] >= m.f_soc_min * m.E_nom)
        model.SoC_Hi = Constraint(model.T, rule=lambda m, t: m.E[t] <= m.f_soc_max * m.E_nom)

        # potência BESS + exclusão de carga/descarga simultâneas
        model.ChargeLimit    = Constraint(model.T, rule=lambda m, t: m.P_ch[t] <= m.P_ch_max * m.gamma[t])
        model.DischargeLimit = Constraint(model.T, rule=lambda m, t: m.P_dis[t] <= m.P_dis_max * (1 - m.gamma[t]))

        # rampa BESS
        model.RampLo = Constraint(model.TRANS, rule=lambda m, t0, t1: m.P_bess[t1] - m.P_bess[t0] >= -m.R_bess)
        model.RampHi = Constraint(model.TRANS, rule=lambda m, t0, t1: m.P_bess[t1] - m.P_bess[t0] <=  m.R_bess)

        self.model = model
        return model

    def solve(self, solver_name="gurobi", tee=True, **solver_opts):
        solver = SolverFactory(solver_name)
        for k, v in solver_opts.items():
            solver.options[k] = v
        self.results = solver.solve(self.model, tee=tee)
        return self.results

    def extract_first_step(self) -> Dict[str, float]:
        t0 = self.times[0]
        m = self.model
        out = {
            "P_bess_kw":  value(m.P_bess[t0]),
            "P_ch_kw":    value(m.P_ch[t0]),
            "P_dis_kw":   value(m.P_dis[t0]),
            "gamma":      int(round(value(m.gamma[t0]))),
            "X_L":        value(m.X_L[t0]),
            "X_PV":       value(m.X_PV[t0]),
            "E_kwh":      value(m.E[t0]),
            "obj":        value(m.Objective)
        }
        if getattr(m, "P_EDS", None) is not None:
            out["P_EDS_kw"] = value(m.P_EDS[t0])
        return out
