from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo


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


def normalize_scenario_values(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    _validate_scenario_df(df, "scenario_df")
    out = df.copy()
    out[value_col] = out[value_col].astype(float)
    vmin = float(out[value_col].min())
    vmax = float(out[value_col].max())
    if np.isclose(vmax, vmin):
        out[value_col] = 0.0
        return out
    out[value_col] = (out[value_col] - vmin) / (vmax - vmin)
    return out


def interpolate_pv_scenarios(
    df_pv: pd.DataFrame,
    target_slots: int,
    source_slots: int | None = None,
    group_cols: tuple[str, str, str] = ("group", "split", "cluster"),
    slot_col: str = "slot",
    value_col: str = "value",
) -> pd.DataFrame:
    _validate_scenario_df(df_pv, "df_pv")
    if target_slots <= 0:
        raise ValueError("target_slots must be > 0")

    pv = df_pv.copy()
    pv[slot_col] = pv[slot_col].astype(int)
    pv[value_col] = pv[value_col].astype(float)

    nslots = int(pv[slot_col].nunique())
    if nslots == target_slots:
        return pv.sort_values(list(group_cols) + [slot_col]).reset_index(drop=True)

    if source_slots is None:
        source_slots = nslots
    if source_slots <= 0:
        raise ValueError("source_slots must be > 0")

    source_step_min = (24.0 * 60.0) / float(source_slots)
    target_step_min = (24.0 * 60.0) / float(target_slots)
    t_new = np.arange(target_slots, dtype=float) * target_step_min

    out_rows: list[dict[str, Any]] = []
    grouped = pv.groupby(list(group_cols), sort=False)
    for key, g in grouped:
        g = g.sort_values(slot_col)
        x_old_slots = g[slot_col].to_numpy(dtype=float)
        y_old = g[value_col].to_numpy(dtype=float)
        t_old = x_old_slots * source_step_min
        y_new = np.interp(t_new, t_old, y_old, left=y_old[0], right=y_old[-1])
        y_new = np.clip(y_new, 0.0, 1.0)

        if not isinstance(key, tuple):
            key = (key,)
        key_dict = {col: val for col, val in zip(group_cols, key)}
        for s in range(target_slots):
            out_rows.append({**key_dict, slot_col: int(s), value_col: float(y_new[s])})

    out = pd.DataFrame(out_rows, columns=[*group_cols, slot_col, value_col])
    return out.sort_values([*group_cols, slot_col]).reset_index(drop=True)


class Parameters:
    def __init__(self, config: dict[str, Any]):
        self.config = config or {}

        self.time_cfg = self.config.get("time", {})
        self.cost_cfg = self.config.get("costs", {})
        self.eds_cfg = self.config.get("EDS", {})
        self.bess_cfg = self.config.get("BESS", {})
        self.pv_cfg = self.config.get("PV", {})
        self.load_cfg = self.config.get("Load", {})
        self.sizing_cfg = self.config.get("sizing", {})

        self.c_shed_per_kwh = float(self.cost_cfg.get("c_shed_per_kwh", 1000.0))
        self.c_pv_curt_per_kwh = float(self.cost_cfg.get("c_pv_curt_per_kwh", 0.0))
        self.tou_map = dict(self.cost_cfg.get("EDS", {}))

        if "bess_degradation_per_kwh" in self.cost_cfg:
            self.c_bess_deg_per_kwh = float(self.cost_cfg["bess_degradation_per_kwh"])
        else:
            rep = float(self.bess_cfg.get("replacement_cost_per_kwh", 0.0))
            ncy = float(self.bess_cfg.get("cycle_life_full", 0.0))
            self.c_bess_deg_per_kwh = (rep / (2.0 * ncy)) if ncy > 0 else 0.0

        self.c_pv_capex_per_kw = float(
            self.pv_cfg.get(
                "capex_per_kw",
                self.sizing_cfg.get("capex_pv_per_kw", 0.0),
            )
        )
        self.c_bess_capex_per_kwh = float(
            self.bess_cfg.get(
                "capex_per_kwh",
                self.sizing_cfg.get(
                    "capex_bess_per_kwh",
                    self.bess_cfg.get("replacement_cost_per_kwh", 0.0),
                ),
            )
        )

        self.p_d_bar_kw = float(self.load_cfg.get("Pmax_kw", 0.0))
        self.p_eds_in_max_kw = float(self.eds_cfg.get("Pmax", self.eds_cfg.get("Pmax_kw", 0.0)))
        self.p_eds_out_max_kw = float(abs(self.eds_cfg.get("Pmin", self.eds_cfg.get("Pmin_kw", 0.0))))

        self.pv_size_max_kw = float(
            self.sizing_cfg.get("P_PV_size_max_kw", self.pv_cfg.get("Pmax_kw", 0.0))
        )
        self.bess_size_max_kwh = float(
            self.sizing_cfg.get("E_BESS_size_max_kwh", self.bess_cfg.get("Emax_kwh", 0.0))
        )

        self.dod_frac = float(self.bess_cfg.get("DoD_frac", 0.9))
        self.eta_bess_c = float(self.bess_cfg.get("eta_c", 0.95))
        self.eta_bess_d = float(self.bess_cfg.get("eta_d", 0.95))

        if "crate_per_h" in self.bess_cfg:
            self.kappa_bess_per_h = float(self.bess_cfg["crate_per_h"])
        else:
            pmax = float(self.bess_cfg.get("Pmax_kw", 0.0))
            emax = float(self.bess_cfg.get("Emax_kwh", 0.0))
            self.kappa_bess_per_h = (pmax / emax) if emax > 0 else 0.0

        e_init_kwh = float(self.bess_cfg.get("E_init_kwh", self.bess_cfg.get("Emax_kwh", 0.0)))
        e_nom_kwh = float(self.bess_cfg.get("Emax_kwh", 0.0))
        self.e_init_frac = (e_init_kwh / e_nom_kwh) if e_nom_kwh > 0 else 0.5
        self.e_init_frac = min(max(self.e_init_frac, 0.0), 1.0)

        self.outage_probability = float(self.eds_cfg.get("outage_probability_pct", 0.0)) / 100.0
        self.outage_duration_h = float(self.eds_cfg.get("outage_duration_hours", 0.0))
        self.contingency_step_slots = int(self.sizing_cfg.get("contingency_step_slots", 1))
        self.max_contingency_starts = self.sizing_cfg.get("max_contingency_starts")

        self.npv_years = int(self.sizing_cfg.get("npv_years", 10))
        self.discount_rate = float(self.sizing_cfg.get("discount_rate", 0.08))
        self.days_per_year = float(self.sizing_cfg.get("days_per_year", 365.0))

        self.default_split = str(self.sizing_cfg.get("split", "train"))
        self.max_time_slots = self.sizing_cfg.get("max_time_slots")
        self.include_zero_prob_scenarios = bool(self.sizing_cfg.get("include_zero_prob_scenarios", True))

    def c_eds_at_slot(self, slot: int, dt_h: float) -> float:
        hour = int((slot * dt_h) % 24)
        return float(self.tou_map.get(f"{hour:02d}:00", 0.0))

    def build_contingencies(self, slots: list[int], dt_h: float) -> dict[str, Any]:
        contingencies = ["c0"]
        windows: dict[str, list[int]] = {"c0": []}
        before: dict[str, list[int]] = {"c0": []}

        if self.outage_probability > 0.0 and self.outage_duration_h > 0.0 and slots:
            duration_slots = max(1, int(round(self.outage_duration_h / max(1e-9, dt_h))))
            step = max(1, self.contingency_step_slots)
            starts = slots[::step]
            if self.max_contingency_starts is not None:
                starts = starts[: int(self.max_contingency_starts)]

            slot_index = {s: i for i, s in enumerate(slots)}
            for start in starts:
                c_name = f"c_{start}"
                contingencies.append(c_name)
                idx0 = slot_index[start]
                windows[c_name] = slots[idx0 : idx0 + duration_slots]
                before[c_name] = slots[:idx0]

        n_out = len(contingencies) - 1
        pi_c: dict[str, float] = {}
        if n_out == 0 or self.outage_probability <= 0.0:
            pi_c["c0"] = 1.0
        else:
            pi_c["c0"] = max(0.0, 1.0 - self.outage_probability)
            p_each = self.outage_probability / float(n_out)
            for c in contingencies:
                if c != "c0":
                    pi_c[c] = p_each

        return {
            "contingencies": contingencies,
            "windows": windows,
            "before": before,
            "pi_c": pi_c,
        }

    def build(self, model: pyo.ConcreteModel, bundle: dict[str, Any]) -> None:
        model.T = pyo.Set(initialize=bundle["slots"], ordered=True)
        model.S = pyo.Set(initialize=bundle["scenarios"], ordered=True)
        model.C = pyo.Set(initialize=bundle["contingencies"], ordered=True)
        model.TRANS = pyo.Set(initialize=bundle["transitions"], dimen=2, ordered=True)
        model.Y = pyo.Set(initialize=list(range(1, self.npv_years + 1)), ordered=True)
        model.W = pyo.Set(model.C, within=model.T, ordered=True, initialize=lambda _, c: bundle["windows"][c])
        model.Before = pyo.Set(
            model.C,
            within=model.T,
            ordered=True,
            initialize=lambda _, c: bundle["before"][c],
        )

        model.dt_h = pyo.Param(model.T, initialize=lambda _, t: float(bundle["dt_h_map"][t]))
        model.pi_s = pyo.Param(model.S, initialize=lambda _, s: float(bundle["pi_s"][s]))
        model.pi_c = pyo.Param(
            model.C,
            initialize=lambda _, c: float(bundle["pi_c"][c]),
            within=pyo.NonNegativeReals,
        )

        model.P_D_bar = pyo.Param(initialize=self.p_d_bar_kw)
        model.P_EDS_in_max = pyo.Param(initialize=max(0.0, self.p_eds_in_max_kw))
        model.P_EDS_out_max = pyo.Param(initialize=max(0.0, self.p_eds_out_max_kw))

        model.c_EDS = pyo.Param(model.T, initialize=lambda _, t: self.c_eds_at_slot(int(t), bundle["dt_h"]))
        model.c_L_shed = pyo.Param(initialize=self.c_shed_per_kwh)
        model.c_PV_curt = pyo.Param(initialize=self.c_pv_curt_per_kwh)
        model.c_BESS_deg = pyo.Param(initialize=self.c_bess_deg_per_kwh)

        model.c_CAPEX_PV = pyo.Param(initialize=self.c_pv_capex_per_kw)
        model.c_CAPEX_BESS = pyo.Param(initialize=self.c_bess_capex_per_kwh)

        model.eta_BESS_c = pyo.Param(initialize=self.eta_bess_c)
        model.eta_BESS_d = pyo.Param(initialize=self.eta_bess_d)
        model.DoD_BESS = pyo.Param(initialize=self.dod_frac)
        model.kappa_BESS = pyo.Param(initialize=max(0.0, self.kappa_bess_per_h))
        model.E_init_frac = pyo.Param(initialize=self.e_init_frac)

        model.days_per_year = pyo.Param(initialize=self.days_per_year)
        model.discount_rate = pyo.Param(initialize=max(0.0, self.discount_rate))


class Load:
    def __init__(self, config: dict[str, Any], parameters: Parameters):
        self.config = config or {}
        self.param = parameters

    def build(self, model: pyo.ConcreteModel, bundle: dict[str, Any]) -> None:
        model.f_L = pyo.Param(
            model.T,
            model.S,
            initialize=lambda _, t, s: float(bundle["f_load"][(int(t), str(s))]),
        )
        model.P_L = pyo.Expression(model.T, model.S, rule=lambda m, t, s: m.f_L[t, s] * m.P_D_bar)
        model.P_L_shed = pyo.Var(model.T, model.S, model.C, domain=pyo.NonNegativeReals)

        model.LoadShedHi = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: m.P_L_shed[t, s, c] <= m.P_L[t, s],
        )


class PV:
    def __init__(self, config: dict[str, Any], parameters: Parameters):
        self.config = config or {}
        self.param = parameters

    def build(self, model: pyo.ConcreteModel, bundle: dict[str, Any]) -> None:
        model.f_PV = pyo.Param(
            model.T,
            model.S,
            initialize=lambda _, t, s: float(bundle["f_pv"][(int(t), str(s))]),
        )
        model.P_hat_PV = pyo.Var(
            domain=pyo.NonNegativeReals,
            bounds=(0.0, max(0.0, self.param.pv_size_max_kw)),
        )
        model.P_PV_avail = pyo.Expression(model.T, model.S, rule=lambda m, t, s: m.f_PV[t, s] * m.P_hat_PV)
        model.P_PV_curt = pyo.Var(model.T, model.S, model.C, domain=pyo.NonNegativeReals)

        model.PVCurtailHi = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: m.P_PV_curt[t, s, c] <= m.P_PV_avail[t, s],
        )


class Grid:
    def __init__(self, config: dict[str, Any], parameters: Parameters):
        self.config = config or {}
        self.param = parameters

    def build(self, model: pyo.ConcreteModel) -> None:
        model.P_EDS_in = pyo.Var(model.T, model.S, model.C, domain=pyo.NonNegativeReals)
        model.P_EDS_out = pyo.Var(model.T, model.S, model.C, domain=pyo.NonNegativeReals)

        model.EDSImportCap = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: m.P_EDS_in[t, s, c] <= m.P_EDS_in_max,
        )
        model.EDSExportCap = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: m.P_EDS_out[t, s, c] <= m.P_EDS_out_max,
        )
        model.EDSOutageImportZero = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: (m.P_EDS_in[t, s, c] == 0.0)
            if (c != "c0" and t in m.W[c])
            else pyo.Constraint.Skip,
        )
        model.EDSOutageExportZero = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: (m.P_EDS_out[t, s, c] == 0.0)
            if (c != "c0" and t in m.W[c])
            else pyo.Constraint.Skip,
        )


class BESS:
    def __init__(self, config: dict[str, Any], parameters: Parameters):
        self.config = config or {}
        self.param = parameters

    def build(self, model: pyo.ConcreteModel) -> None:
        model.E_hat_BESS = pyo.Var(
            domain=pyo.NonNegativeReals,
            bounds=(0.0, max(0.0, self.param.bess_size_max_kwh)),
        )

        model.P_BESS_c = pyo.Var(model.T, model.S, model.C, domain=pyo.NonNegativeReals)
        model.P_BESS_d = pyo.Var(model.T, model.S, model.C, domain=pyo.NonNegativeReals)
        model.E_BESS = pyo.Var(model.T, model.S, model.C, domain=pyo.NonNegativeReals)

        model.gamma_BESS_c = pyo.Var(model.T, model.S, model.C, domain=pyo.Binary)
        model.gamma_BESS_d = pyo.Var(model.T, model.S, model.C, domain=pyo.Binary)

        first_t = model.T.first()

        model.BESSDynamics = pyo.Constraint(
            model.TRANS,
            model.S,
            model.C,
            rule=lambda m, t0, t1, s, c: m.E_BESS[t1, s, c]
            == m.E_BESS[t0, s, c]
            + m.dt_h[t0]
            * (
                m.eta_BESS_c * m.P_BESS_c[t0, s, c]
                - (1.0 / m.eta_BESS_d) * m.P_BESS_d[t0, s, c]
            ),
        )
        model.BESSInit = pyo.Constraint(
            model.S,
            model.C,
            rule=lambda m, s, c: m.E_BESS[first_t, s, c] == m.E_init_frac * m.E_hat_BESS,
        )

        model.BESSMode = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: m.gamma_BESS_c[t, s, c] + m.gamma_BESS_d[t, s, c] <= 1,
        )

        model.BESSChargeCRate = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: m.P_BESS_c[t, s, c] <= m.kappa_BESS * m.E_hat_BESS,
        )
        model.BESSDischargeCRate = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: m.P_BESS_d[t, s, c] <= m.kappa_BESS * m.E_hat_BESS,
        )
        model.BESSChargeMode = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: m.P_BESS_c[t, s, c]
            <= m.kappa_BESS * max(0.0, self.param.bess_size_max_kwh) * m.gamma_BESS_c[t, s, c],
        )
        model.BESSDischargeMode = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: m.P_BESS_d[t, s, c]
            <= m.kappa_BESS * max(0.0, self.param.bess_size_max_kwh) * m.gamma_BESS_d[t, s, c],
        )

        model.BESSSoCLo = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: m.E_BESS[t, s, c] >= (1.0 - m.DoD_BESS) * m.E_hat_BESS,
        )
        model.BESSSoCHi = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: m.E_BESS[t, s, c] <= m.E_hat_BESS,
        )


class EV:
    def __init__(self, config: dict[str, Any], parameters: Parameters):
        self.config = config or {}
        self.param = parameters

    def build(self, model: pyo.ConcreteModel) -> None:
        # Placeholder for future EV sizing/operation coupling.
        _ = model


class MicrogridDesign:
    def __init__(
        self,
        config: dict[str, Any],
        df_pv: pd.DataFrame | None = None,
        df_load: pd.DataFrame | None = None,
        df_prob_joint: pd.DataFrame | None = None,
    ):
        self.config = config if isinstance(config, dict) else {}

        if df_pv is None:
            df_pv = pd.read_csv(Path("data/sizing/prototypes_pv_dtw_train.csv"))
        if df_load is None:
            df_load = pd.read_csv(Path("data/sizing/prototypes_load_dtw_all_train.csv"))
        if df_prob_joint is None:
            df_prob_joint = pd.read_csv(Path("data/sizing/prob_joint_load_pv.csv"))

        self.df_pv = df_pv
        self.df_load = df_load
        self.df_prob_joint = df_prob_joint

        self.param = Parameters(self.config)
        self.bess = BESS(self.config.get("BESS", {}), self.param)
        self.load = Load(self.config.get("Load", {}), self.param)
        self.grid = Grid(self.config.get("EDS", {}), self.param)
        self.pv = PV(self.config.get("PV", {}), self.param)
        self.ev = EV(self.config.get("EV", {}), self.param)

        self.model: pyo.ConcreteModel | None = None
        self.results = None
        self._meta: dict[str, Any] = {}

    def _prepare_bundle(self) -> dict[str, Any]:
        _validate_scenario_df(self.df_pv, "df_pv")
        _validate_scenario_df(self.df_load, "df_load")
        missing_joint = [c for c in JOINT_PROB_REQUIRED_COLS if c not in self.df_prob_joint.columns]
        if missing_joint:
            raise ValueError(f"df_prob_joint is missing required columns: {missing_joint}")

        split = self.param.default_split
        pv = self.df_pv.copy()
        load = self.df_load.copy()
        joint = self.df_prob_joint.copy()

        if "split" in pv.columns:
            pv = pv[pv["split"] == split].copy()
        if "split" in load.columns:
            load = load[load["split"] == split].copy()
        if "split" in joint.columns:
            joint = joint[joint["split"] == split].copy()

        if pv.empty or load.empty or joint.empty:
            raise ValueError(f"No rows found for split='{split}' in sizing inputs.")

        pv["cluster"] = pv["cluster"].astype(int)
        load["cluster"] = load["cluster"].astype(int)
        pv["slot"] = pv["slot"].astype(int)
        load["slot"] = load["slot"].astype(int)
        pv["value"] = pv["value"].astype(float)
        load["value"] = load["value"].astype(float)

        joint["cluster_load"] = joint["cluster_load"].astype(int)
        joint["cluster_pv"] = joint["cluster_pv"].astype(int)
        joint["probability"] = joint["probability"].astype(float)
        if not self.param.include_zero_prob_scenarios:
            joint = joint[joint["probability"] > 0.0].copy()
        if joint.empty:
            raise ValueError("No scenario rows available after filtering probabilities.")

        full_slot_count = int(load["slot"].nunique())
        slots = sorted(load["slot"].unique().tolist())
        if self.param.max_time_slots is not None:
            slots = slots[: int(self.param.max_time_slots)]
            load = load[load["slot"].isin(slots)].copy()
            pv = pv[pv["slot"].isin(slots)].copy()
        if len(slots) < 2:
            raise ValueError("Need at least 2 time slots for dynamics.")

        dt_h = 24.0 / float(max(1, full_slot_count))
        transitions = [(slots[i], slots[i + 1]) for i in range(len(slots) - 1)]
        dt_h_map = {t0: dt_h for (t0, _t1) in transitions}
        dt_h_map[slots[-1]] = dt_h

        load_map = {(int(r.cluster), int(r.slot)): float(r.value) for r in load.itertuples(index=False)}
        pv_map = {(int(r.cluster), int(r.slot)): float(r.value) for r in pv.itertuples(index=False)}

        scenarios: list[str] = []
        cluster_load_of_s: dict[str, int] = {}
        cluster_pv_of_s: dict[str, int] = {}
        pi_s: dict[str, float] = {}
        for idx, row in enumerate(joint.itertuples(index=False)):
            s_name = f"s{idx}"
            scenarios.append(s_name)
            cluster_load_of_s[s_name] = int(row.cluster_load)
            cluster_pv_of_s[s_name] = int(row.cluster_pv)
            pi_s[s_name] = float(row.probability)

        sum_pi = sum(pi_s.values())
        if sum_pi <= 0:
            raise ValueError("Scenario probabilities sum to zero.")
        for s in scenarios:
            pi_s[s] = pi_s[s] / sum_pi

        f_load: dict[tuple[int, str], float] = {}
        f_pv: dict[tuple[int, str], float] = {}
        for s in scenarios:
            cl = cluster_load_of_s[s]
            cp = cluster_pv_of_s[s]
            for t in slots:
                if (cl, t) not in load_map:
                    raise ValueError(f"Missing load prototype for cluster={cl}, slot={t}")
                if (cp, t) not in pv_map:
                    raise ValueError(f"Missing PV prototype for cluster={cp}, slot={t}")
                f_load[(t, s)] = max(0.0, load_map[(cl, t)])
                f_pv[(t, s)] = max(0.0, pv_map[(cp, t)])

        cont = self.param.build_contingencies(slots, dt_h)

        return {
            "slots": slots,
            "scenarios": scenarios,
            "cluster_load_of_s": cluster_load_of_s,
            "cluster_pv_of_s": cluster_pv_of_s,
            "f_load": f_load,
            "f_pv": f_pv,
            "pi_s": pi_s,
            "transitions": transitions,
            "dt_h": dt_h,
            "dt_h_map": dt_h_map,
            "contingencies": cont["contingencies"],
            "windows": cont["windows"],
            "before": cont["before"],
            "pi_c": cont["pi_c"],
        }

    def _build_non_anticipativity(self, model: pyo.ConcreteModel) -> None:
        model.NonAnt_P_BESS_c = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: (m.P_BESS_c[t, s, c] == m.P_BESS_c[t, s, "c0"])
            if (c != "c0" and t in m.Before[c])
            else pyo.Constraint.Skip,
        )
        model.NonAnt_P_BESS_d = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: (m.P_BESS_d[t, s, c] == m.P_BESS_d[t, s, "c0"])
            if (c != "c0" and t in m.Before[c])
            else pyo.Constraint.Skip,
        )
        model.NonAnt_E_BESS = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: (m.E_BESS[t, s, c] == m.E_BESS[t, s, "c0"])
            if (c != "c0" and t in m.Before[c])
            else pyo.Constraint.Skip,
        )
        model.NonAnt_gamma_c = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: (m.gamma_BESS_c[t, s, c] == m.gamma_BESS_c[t, s, "c0"])
            if (c != "c0" and t in m.Before[c])
            else pyo.Constraint.Skip,
        )
        model.NonAnt_gamma_d = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: (m.gamma_BESS_d[t, s, c] == m.gamma_BESS_d[t, s, "c0"])
            if (c != "c0" and t in m.Before[c])
            else pyo.Constraint.Skip,
        )
        model.NonAnt_PV_curt = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: (m.P_PV_curt[t, s, c] == m.P_PV_curt[t, s, "c0"])
            if (c != "c0" and t in m.Before[c])
            else pyo.Constraint.Skip,
        )
        model.NonAnt_L_shed = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: (m.P_L_shed[t, s, c] == m.P_L_shed[t, s, "c0"])
            if (c != "c0" and t in m.Before[c])
            else pyo.Constraint.Skip,
        )
        model.NonAnt_EDS_in = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: (m.P_EDS_in[t, s, c] == m.P_EDS_in[t, s, "c0"])
            if (c != "c0" and t in m.Before[c])
            else pyo.Constraint.Skip,
        )
        model.NonAnt_EDS_out = pyo.Constraint(
            model.T,
            model.S,
            model.C,
            rule=lambda m, t, s, c: (m.P_EDS_out[t, s, c] == m.P_EDS_out[t, s, "c0"])
            if (c != "c0" and t in m.Before[c])
            else pyo.Constraint.Skip,
        )

    def build(self):
        bundle = self._prepare_bundle()

        self.model = pyo.ConcreteModel(name="Sizing_Stochastic")
        self.param.build(self.model, bundle)
        self.bess.build(self.model)
        self.load.build(self.model, bundle)
        self.grid.build(self.model)
        self.pv.build(self.model, bundle)
        self.ev.build(self.model)

        m = self.model
        m.PowerBalance = pyo.Constraint(
            m.T,
            m.S,
            m.C,
            rule=lambda _m, t, s, c: (
                (_m.P_PV_avail[t, s] - _m.P_PV_curt[t, s, c])
                + _m.P_BESS_d[t, s, c]
                - _m.P_BESS_c[t, s, c]
                + _m.P_EDS_in[t, s, c]
                - _m.P_EDS_out[t, s, c]
                == _m.P_L[t, s] - _m.P_L_shed[t, s, c]
            ),
        )

        self._build_non_anticipativity(m)

        m.CAPEX = pyo.Expression(expr=m.c_CAPEX_PV * m.P_hat_PV + m.c_CAPEX_BESS * m.E_hat_BESS)
        m.OPEX_day = pyo.Expression(
            expr=sum(
                m.pi_s[s]
                * m.pi_c[c]
                * m.dt_h[t]
                * (
                    m.c_EDS[t] * m.P_EDS_in[t, s, c]
                    + m.c_L_shed * m.P_L_shed[t, s, c]
                    + m.c_PV_curt * m.P_PV_curt[t, s, c]
                    + m.c_BESS_deg * (m.P_BESS_c[t, s, c] + m.P_BESS_d[t, s, c])
                )
                for t in m.T
                for s in m.S
                for c in m.C
            )
        )
        m.OPEX_annual = pyo.Expression(expr=m.days_per_year * m.OPEX_day)
        m.NPV_OPEX = pyo.Expression(
            expr=sum(m.OPEX_annual / ((1.0 + m.discount_rate) ** y) for y in m.Y)
        )
        m.Objective = pyo.Objective(expr=m.CAPEX + m.NPV_OPEX, sense=pyo.minimize)

        self._meta = {
            "slots": bundle["slots"],
            "scenarios": bundle["scenarios"],
            "contingencies": bundle["contingencies"],
            "dt_h": bundle["dt_h"],
            "cluster_load_of_s": bundle["cluster_load_of_s"],
            "cluster_pv_of_s": bundle["cluster_pv_of_s"],
        }
        return m

    def optimize(self, solver: str = "gurobi", tee: bool = True, **solver_options):
        if self.model is None:
            raise RuntimeError("Build the model before solving.")
        opt = pyo.SolverFactory(solver)
        for k, v in solver_options.items():
            opt.options[k] = v
        self.results = opt.solve(self.model, tee=tee, load_solutions=True)
        return self.results

    def get_results(self):
        if self.model is None:
            return {}

        def _safe(v):
            val = pyo.value(v, exception=False)
            return None if val is None else float(val)

        m = self.model
        return {
            "P_hat_PV_kw": _safe(m.P_hat_PV),
            "E_hat_BESS_kwh": _safe(m.E_hat_BESS),
            "CAPEX": _safe(m.CAPEX),
            "OPEX_day": _safe(m.OPEX_day),
            "OPEX_annual": _safe(m.OPEX_annual),
            "NPV_OPEX": _safe(m.NPV_OPEX),
            "Objective": _safe(m.Objective),
            "metadata": self._meta,
        }
