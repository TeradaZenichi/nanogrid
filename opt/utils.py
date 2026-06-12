# -*- coding: utf-8 -*-
"""
opt/utils.py
Reusable utilities for MPCs (Off-Grid and On-Grid).

Includes:
- time grid construction and predecessor pairs
- creation of the contingency set (subset of T)
- column normalization/checking
- reading and scaling of series (PV and Load)
- assembly of forecasts aligned with the time grid
- helpers for capturing/saving results
"""
import json
import os
import re
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd


# ---------------------------
# Solver selection and fast defaults
# ---------------------------

_DETECTED_SOLVER: Optional[str] = None


def detect_solver() -> str:
    """Return 'gurobi' when usable, otherwise 'appsi_highs' (open source).

    Solving a trivial LP is the only check that covers every Gurobi interface
    (gurobipy, command line, WLS) and validates the license in one shot.
    The result is cached for the process lifetime.
    """
    global _DETECTED_SOLVER
    if _DETECTED_SOLVER is None:
        from pyomo.environ import ConcreteModel, Objective, RangeSet, SolverFactory, Var, quicksum

        try:
            # 2100 variables: above the 2000-var cap of the pip-installed
            # size-limited Gurobi license, so restricted installs are caught
            # here instead of failing later on the real (large) models.
            probe = ConcreteModel()
            probe.I = RangeSet(2100)
            probe.x = Var(probe.I, bounds=(0, 1))
            probe.obj = Objective(expr=quicksum(probe.x[i] for i in probe.I))
            SolverFactory("gurobi").solve(probe, tee=False)
            _DETECTED_SOLVER = "gurobi"
        except Exception as e:
            print(f"[solver] Gurobi unavailable or size-limited ({type(e).__name__}); falling back to HiGHS.")
            _DETECTED_SOLVER = "appsi_highs"
    return _DETECTED_SOLVER


# Above this size, barrier/IPM without crossover beats simplex decisively
# (e.g. the multi-year sizing LP); below it, each solver's default is faster.
LARGE_LP_VARS = 100_000


def _lp_size(model) -> tuple[int, bool]:
    """(n_active_vars, is_pure_lp); stops counting at the first discrete var."""
    from pyomo.environ import Var

    n = 0
    for v in model.component_data_objects(Var, active=True):
        n += 1
        if v.is_binary() or v.is_integer():
            return n, False
    return n, True


def solve_model(model, tee: bool = False,
                time_limit: Optional[float] = None,
                threads: Optional[int] = None,
                mip_gap: Optional[float] = None,
                solver_name: Optional[str] = None,
                load_solutions: bool = True):
    """Solve with Gurobi if licensed, else HiGHS — fastest method by default.

    Large pure LPs (> LARGE_LP_VARS variables, e.g. the sizing model) are
    pointed at barrier/IPM without crossover; small LPs and MIPs keep each
    solver's own defaults, which benchmarked faster for the MPC models.
    """
    from pyomo.environ import SolverFactory

    name = solver_name or detect_solver()
    solver = SolverFactory(name)
    n_vars, lp = _lp_size(model)
    large_lp = lp and n_vars > LARGE_LP_VARS

    if name == "gurobi":
        if large_lp:
            solver.options["Method"] = 2     # barrier
            solver.options["Crossover"] = 0  # skip basis crossover
        if time_limit is not None:
            solver.options["TimeLimit"] = float(time_limit)
        if threads is not None:
            solver.options["Threads"] = int(threads)
        if mip_gap is not None:
            solver.options["MIPGap"] = float(mip_gap)
    else:  # appsi_highs (legacy SolverFactory interface)
        if large_lp:
            solver.options["solver"] = "ipm"
            solver.options["run_crossover"] = "off"
        if time_limit is not None:
            solver.options["time_limit"] = float(time_limit)
        if threads is not None:
            solver.options["threads"] = int(threads)
        if mip_gap is not None:
            solver.options["mip_rel_gap"] = float(mip_gap)

    try:
        return solver.solve(model, tee=tee, load_solutions=load_solutions)
    except Exception as e:
        # Last-resort safety net: a Gurobi that passed detection can still
        # fail at solve time (license size cap, expired WLS, quota). Retry
        # once with HiGHS — unless the caller forced a specific solver.
        if name == "gurobi" and solver_name is None:
            global _DETECTED_SOLVER
            print(f"[solver] Gurobi failed at solve time ({e}); retrying with HiGHS.")
            _DETECTED_SOLVER = "appsi_highs"
            return solve_model(
                model, tee=tee, time_limit=time_limit, threads=threads,
                mip_gap=mip_gap, solver_name="appsi_highs",
                load_solutions=load_solutions,
            )
        raise


# ---------------------------
# Sizing -> operation bridge
# ---------------------------

def apply_sizing_case(params: Dict[str, Any],
                      case: str,
                      results_root: str = "Results/sizing") -> Dict[str, Any]:
    """Override PV/BESS capacities with the sized values of a sizing case.

    Reads Results/sizing/<case>/sizing_decision_variables.json and returns a
    copy of `params` operating the sized system instead of the catalog one:
    - PV.Pmax_kw   <- P_hat_PV_kw
    - BESS.Emax_kwh <- E_hat_BESS_kwh
    - BESS.Pmax_kw, ramp_kw_per_step and E_init_kwh are rescaled preserving
      their original ratios to Emax_kwh (C-rate, ramp/Pmax and initial SoC).
    """
    path = Path(results_root) / case / "sizing_decision_variables.json"
    if not path.exists():
        raise FileNotFoundError(f"Sizing case '{case}' not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    dv = data.get("decision_variables", {}) or {}
    p_pv = dv.get("P_hat_PV_kw")
    e_bess = dv.get("E_hat_BESS_kwh")
    if p_pv is None or e_bess is None:
        raise ValueError(f"Sizing case '{case}' has no P_hat_PV_kw/E_hat_BESS_kwh in {path}")

    out = deepcopy(params)
    bess = out["BESS"]
    e_old = float(bess["Emax_kwh"])
    p_old = float(bess["Pmax_kw"])
    e_new = float(e_bess)

    out["PV"]["Pmax_kw"] = float(p_pv)
    bess["Emax_kwh"] = e_new
    bess["Pmax_kw"] = (p_old / e_old) * e_new  # preserve C-rate
    if bess.get("ramp_kw_per_step") is not None:
        bess["ramp_kw_per_step"] = (float(bess["ramp_kw_per_step"]) / p_old) * bess["Pmax_kw"]
    if bess.get("E_init_kwh") is not None:
        bess["E_init_kwh"] = (float(bess["E_init_kwh"]) / e_old) * e_new  # preserve initial SoC

    out["sizing_case_applied"] = {
        "case": case,
        "source": path.as_posix(),
        "P_hat_PV_kw": float(p_pv),
        "E_hat_BESS_kwh": e_new,
        "BESS_Pmax_kw": float(bess["Pmax_kw"]),
    }
    return out


# ---------------------------
# Time, time grid and contingencies
# ---------------------------

def build_dt_vector(horizon_hours: int,
                    outage_duration_hours: int,
                    dt1_min: int,
                    dt2_min: int) -> List[int]:
    """
    Creates a vector of steps (in minutes) for the horizon:
      - First 'outage_duration_hours' with fine resolution (dt1_min).
      - Remainder of the horizon with coarse resolution (dt2_min).

    Completely replaces the old use of 'fine_hours'.
    """
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be > 0")
    if not (0 <= outage_duration_hours <= horizon_hours):
        raise ValueError("outage_duration_hours must be in [0, horizon_hours]")
    if dt1_min <= 0 or dt2_min <= 0:
        raise ValueError("timestep_1_min and timestep_2_min must be > 0 (min)")

    # fine steps during the outage window
    steps_fine = (outage_duration_hours * 60) // dt1_min
    if steps_fine * dt1_min != outage_duration_hours * 60:
        raise ValueError("outage_duration_hours * 60 must be an exact multiple of timestep_1_min")

    # coarse steps in the remainder
    steps_coarse = ((horizon_hours - outage_duration_hours) * 60) // dt2_min
    if steps_coarse * dt2_min != (horizon_hours - outage_duration_hours) * 60:
        raise ValueError("(horizon_hours - outage_duration_hours) * 60 must be an exact multiple of timestep_2_min")

    dt_min = [dt1_min] * steps_fine + [dt2_min] * steps_coarse
    if len(dt_min) < 2:
        raise ValueError("The horizon must have at least 2 steps.")
    return dt_min


def build_time_grid(start_dt: datetime, dt_min: List[int]) -> List[datetime]:
    """
    Builds the list of timestamps T from a start time and the vector of steps in minutes.
    Returns a list of the same size as len(dt_min), with the last index representing the end of the horizon.
    """
    times = [start_dt]
    for dm in dt_min[:-1]:
        times.append(times[-1] + timedelta(minutes=dm))
    return times


def predecessor_pairs(times: List[datetime]) -> List[Tuple[datetime, datetime]]:
    """Creates consecutive predecessor pairs (t0, t1) from T."""
    return list(zip(times[:-1], times[1:]))


def build_contingency_times(start_dt: datetime,
                            horizon_hours: int,
                            outage_duration_hours: int,
                            dt1_min: int,
                            dt2_min: int) -> List[datetime]:
    """
    Generates the start times 'c' for contingency (outage) scenarios with the rule:
      - include half of the fine steps (dt1) at the beginning,
      - skip the other half of the fine steps,
      - include all coarse steps (dt2) except the last one.

    Notes:
      - If outage_duration_hours = 0, there will be no fine part; all coarse steps except the last are included.
      - Returns a sorted list with no duplicates.
    """
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be > 0")
    if not (0 <= outage_duration_hours <= horizon_hours):
        raise ValueError("outage_duration_hours must be in [0, horizon_hours]")
    if dt1_min <= 0 or dt2_min <= 0:
        raise ValueError("timestep_1_min and timestep_2_min must be > 0 (min)")

    # number of fine and coarse steps (same divisibility checks as in build_dt_vector)
    steps_fine = (outage_duration_hours * 60) // dt1_min
    if steps_fine * dt1_min != outage_duration_hours * 60:
        raise ValueError("outage_duration_hours * 60 must be an exact multiple of timestep_1_min")

    steps_coarse = ((horizon_hours - outage_duration_hours) * 60) // dt2_min
    if steps_coarse * dt2_min != (horizon_hours - outage_duration_hours) * 60:
        raise ValueError("(horizon_hours - outage_duration_hours) * 60 must be an exact multiple of timestep_2_min")

    # complete time grid (same logic as build_time_grid)
    dt_min = [dt1_min] * steps_fine + [dt2_min] * steps_coarse
    times: List[datetime] = [start_dt]
    for dm in dt_min[:-1]:
        times.append(times[-1] + timedelta(minutes=dm))

    # indices that are included in the contingency set
    # half of the fine steps (floor)
    half_fine = steps_fine // 2
    idx_fine_in = list(range(1, half_fine))  # includes the 1st half, starting from index 1
    # coarse steps: all except the last one
    idx_coarse_in = list(range(steps_fine, steps_fine + max(steps_coarse - 1, 0)))

    # final result (sorted and unique)
    cont_idx = sorted(set(idx_fine_in + idx_coarse_in))
    contingencies = [times[i] for i in cont_idx]
    return contingencies


def build_time_and_contingencies_from_params(params: Dict[str, Any],
                                              start_dt: datetime) -> Tuple[List[datetime], List[datetime]]:
    horizon_hours = int(params["horizon_hours"])
    dt1_min = int(params["timestep_1_min"])
    dt2_min = int(params["timestep_2_min"])
    outage_duration_hours = int(params.get("outage_duration_hours", 0))

    dt_min = build_dt_vector(horizon_hours, outage_duration_hours, dt1_min, dt2_min)
    times = build_time_grid(start_dt, dt_min)
    contingencies = build_contingency_times(start_dt, horizon_hours, outage_duration_hours, dt1_min, dt2_min)
    return times, contingencies


# ---------------------------
# Normalization / checking
# ---------------------------

def pnorm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column names (lowercase, separators to '_', no strange symbols)."""
    def _norm(name: str) -> str:
        s = name.strip().lower()
        s = re.sub(r'[^a-z0-9]+', '_', s)
        s = re.sub(r'_{2,}', '_', s).strip('_')
        return s
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    return df


def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Returns the first column found among the candidates or raises ValueError."""
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"CSV must contain one of the following columns: {candidates}")


# ---------------------------
# Series and forecasts
# ---------------------------

def load_series_scaled(params: Dict[str, Any],
                       load_csv: str,
                       pv_csv: str,
                       col_time: str = "timestamp",
                       col_pu: str = "p_norm") -> Tuple[pd.Series, pd.Series]:
    load_df = pd.read_csv(load_csv)
    pv_df = pd.read_csv(pv_csv)
    for name, df in [("load", load_df), ("pv", pv_df)]:
        if col_time not in df.columns or col_pu not in df.columns:
            raise ValueError(f"{name} CSV must contain '{col_time}' and '{col_pu}' columns.")
    load_df[col_time] = pd.to_datetime(load_df[col_time])
    pv_df[col_time] = pd.to_datetime(pv_df[col_time])
    load_df.set_index(col_time, inplace=True)
    pv_df.set_index(col_time, inplace=True)
    load_df.sort_index(inplace=True)
    pv_df.sort_index(inplace=True)

    load_pu = load_df[col_pu].astype(float).clip(0.0, 1.0)
    pv_pu = pv_df[col_pu].astype(float).clip(0.0, 1.0)

    P_L_max = float(params["P_L_nom_kw"])
    P_PV_max = float(params["P_PV_nom_kw"])

    load_kw = load_pu * P_L_max
    pv_kw = pv_pu * P_PV_max
    return load_kw, pv_kw


def slice_forecasts(times: List[datetime],
                    load_series_kw: pd.Series,
                    pv_series_kw: pd.Series) -> Dict[str, Dict[datetime, float]]:
    """
    Extracts forecasts for the timestamps of the T grid.
    Raises KeyError if there are timestamps without a corresponding value in any series.
    """
    missing_load = [t for t in times if t not in load_series_kw.index]
    missing_pv = [t for t in times if t not in pv_series_kw.index]
    if missing_load:
        raise KeyError(f"Missing load values for timestamps: {missing_load[:3]}...")
    if missing_pv:
        raise KeyError(f"Missing PV values for timestamps: {missing_pv[:3]}...")
    fc_load = {t: float(load_series_kw.loc[t]) for t in times}
    fc_pv = {t: float(pv_series_kw.loc[t]) for t in times}
    return {"load_kw": fc_load, "pv_kw": fc_pv}


# ---------------------------
# Result capture / saving
# ---------------------------

def _val(v, default=0.0):
    """Tries to extract a numerical value (compatible with pyomo.environ.value)."""
    try:
        from pyomo.environ import value
        x = value(v)
        return float(x if x is not None else default)
    except Exception:
        try:
            return float(v)
        except Exception:
            return float(default)


def horizon_snapshot(model, times: List[datetime]) -> Dict[str, List[float]]:
    """
    1D Snapshot (for models with variables indexed only on T).
    Kept for compatibility with previous Off-Grid scripts.
    """
    ts = list(times)
    out = {
        "timestamps": [t.isoformat() for t in ts],
        "dt_h":       [_val(model.dt_h[t]) for t in ts],
        "Load_kw":    [_val(model.Load_kw[t]) for t in ts],
        "PV_kw":      [_val(model.PV_kw[t]) for t in ts],
        "X_L":        [_val(model.X_L[t]) for t in ts],
        "X_PV":       [_val(model.X_PV[t]) for t in ts],
        "P_bess_kw":  [_val(model.P_bess[t]) for t in ts],
        "P_ch_kw":    [_val(model.P_ch[t]) for t in ts],
        "P_dis_kw":   [_val(model.P_dis[t]) for t in ts],
        "gamma":      [int(round(_val(model.gamma[t], 0))) for t in ts],
        "E_kwh":      [_val(model.E[t]) for t in ts],
    }
    if getattr(model, "P_EDS", None) is not None:
        out["EDS_kw"] = [_val(model.P_EDS[t]) for t in ts]
    return out


def horizon_snapshot_2d(model,
                        times: List[datetime],
                        scenario: Any) -> Dict[str, List[float]]:
    """
    2D Snapshot (for models with variables indexed on (T, C)), extracting a specific scenario.
    Useful for the stochastic On-Grid model.
    """
    ts = list(times)
    c = scenario
    out = {
        "scenario":        str(c),
        "timestamps":      [t.isoformat() for t in ts],
        "dt_h":            [_val(model.dt_h[t]) for t in ts],
        "Load_kw":         [_val(model.Load_kw[t]) for t in ts],
        "PV_kw":           [_val(model.PV_kw[t]) for t in ts],
        "X_L":             [_val(model.X_L[t, c]) for t in ts],
        "X_PV":            [_val(model.X_PV[t, c]) for t in ts],
        "P_bess_kw":       [_val(model.P_bess[t, c]) for t in ts],
        "P_ch_kw":         [_val(model.P_ch[t, c]) for t in ts],
        "P_dis_kw":        [_val(model.P_dis[t, c]) for t in ts],
        "gamma":           [int(round(_val(model.gamma[t, c], 0))) for t in ts],
        "E_kwh":           [_val(model.E[t, c]) for t in ts],
        "P_grid_in_kw":    [_val(model.P_gin[t, c]) for t in ts] if hasattr(model, "P_gin") else [],
        "P_grid_out_kw":   [_val(model.P_gout[t, c]) for t in ts] if hasattr(model, "P_gout") else [],
        "c_grid":          [_val(model.c_grid[t]) for t in ts] if hasattr(model, "c_grid") else [],
    }
    return out


def save_log(results_log, path: str = "outputs/dispatch_log.json"):
    """Saves a dictionary to a JSON file with indentation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import json as _json
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(results_log, f, ensure_ascii=False, indent=2)
    print(f"[log] saved to {path}")