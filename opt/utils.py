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
import hashlib
import math
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
DEFAULT_SIZING_CASE = 'alpha_gt_0'
DEFAULT_OPERATION_YEAR = 1


def detect_solver() -> str:
    """'gurobi' if a solve works, else 'appsi_highs'. Cached per process.

    The probe has 2100 variables so the size-limited pip license (2000-var
    cap) fails here instead of on the real models.
    """
    global _DETECTED_SOLVER
    if _DETECTED_SOLVER is None:
        from pyomo.environ import ConcreteModel, Objective, RangeSet, SolverFactory, Var, quicksum

        try:
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


# Above this size barrier/IPM without crossover beats the solver defaults.
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
    """Solve with Gurobi if available, else HiGHS. Large pure LPs use
    barrier/IPM without crossover; everything else keeps solver defaults."""
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
        # Gurobi can pass detection and still fail here (license cap/quota);
        # retry once with HiGHS unless the caller forced a solver.
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
    """Copy of `params` with PV/BESS capacities taken from a sizing case.

    Pmax and ramp are rescaled from the catalog ratios. The initial stored
    energy is transferred directly from the optimized cyclic sizing state.
    """
    path = Path(results_root) / case / "sizing_decision_variables.json"
    if not path.exists():
        raise FileNotFoundError(f"Sizing case '{case}' not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    dv = data.get("decision_variables", {}) or {}
    p_pv = dv.get("P_hat_PV_kw")
    e_bess = dv.get("E_hat_BESS_kwh")
    e_init = dv.get("E_BESS_init_kwh")
    if p_pv is None or e_bess is None or e_init is None:
        raise ValueError(
            f"Sizing case '{case}' has no P_hat_PV_kw/E_hat_BESS_kwh/E_BESS_init_kwh in {path}"
        )

    out = deepcopy(params)
    bess = out["BESS"]
    e_old = float(bess["Emax_kwh"])
    p_old = float(bess["Pmax_kw"])
    e_new = float(e_bess)
    e_init_new = float(e_init)
    if not 0.0 <= e_init_new <= e_new:
        raise ValueError(
            f"Sizing case '{case}' has invalid cyclic initial energy: "
            f"E_init={e_init_new}, Emax={e_new}"
        )

    out["PV"]["Pmax_kw"] = float(p_pv)
    bess["Emax_kwh"] = e_new
    bess["Pmax_kw"] = (p_old / e_old) * e_new  # preserve C-rate
    if bess.get("ramp_kw_per_step") is not None:
        bess["ramp_kw_per_step"] = (float(bess["ramp_kw_per_step"]) / p_old) * bess["Pmax_kw"]
    bess["E_init_kwh"] = e_init_new

    out["sizing_case_applied"] = {
        "case": case,
        "source": path.as_posix(),
        "P_hat_PV_kw": float(p_pv),
        "E_hat_BESS_kwh": e_new,
        "E_BESS_init_kwh": e_init_new,
        "BESS_initial_soc_fraction": (e_init_new / e_new) if e_new > 0.0 else 0.0,
        "BESS_Pmax_kw": float(bess["Pmax_kw"]),
    }
    return out


def load_sized_parameters(params_json: str = 'data/parameters.json',
                          case: str = DEFAULT_SIZING_CASE,
                          results_root: str = 'Results/sizing',
                          operation_year: int = DEFAULT_OPERATION_YEAR) -> Dict[str, Any]:
    '''Load catalog parameters and obligatorily apply a solved sizing case.

    The current operational campaign is anchored at year 1. A different year
    must be implemented explicitly instead of silently reusing initial
    capacities.
    '''
    if int(operation_year) != 1:
        raise NotImplementedError('Only operation_year=1 is currently supported')

    params_path = Path(params_json)
    if not params_path.exists():
        raise FileNotFoundError(f'Parameter file not found: {params_path}')
    sizing_path = Path(results_root) / case / 'sizing_decision_variables.json'
    if not sizing_path.exists():
        raise FileNotFoundError(
            f'Sizing case {case!r} not found at {sizing_path}. Run experiments/01_sizing.py first.'
        )
    sizing_data = json.loads(sizing_path.read_text(encoding='utf-8'))
    if not bool(sizing_data.get('has_loaded_solution', False)):
        raise ValueError(f'Sizing case {case!r} has no loaded feasible solution: {sizing_path}')
    if not bool((sizing_data.get('metadata', {}) or {}).get('cyclic_daily_soc', False)):
        raise ValueError(
            f'Sizing case {case!r} predates cyclic daily SoC closure: {sizing_path}. '
            'Regenerate it with experiments/01_sizing.py.'
        )
    if not bool(
        (sizing_data.get('metadata', {}) or {}).get('optimized_cyclic_initial_soc', False)
    ):
        raise ValueError(
            f'Sizing case {case!r} predates optimized cyclic initial SoC: {sizing_path}. '
            'Regenerate it with experiments/01_sizing.py.'
        )
    cycle_residual = (sizing_data.get('model_audit', {}) or {}).get(
        'cycle_closure_max_abs_kwh'
    )
    if cycle_residual is None or abs(float(cycle_residual)) > 1e-6:
        raise ValueError(
            f'Sizing case {case!r} failed its cyclic-closure audit: {cycle_residual}'
        )

    params = json.loads(params_path.read_text(encoding='utf-8'))
    out = apply_sizing_case(params, case=case, results_root=results_root)
    out['sizing_case_applied'].update(
        operation_year=1,
        source_sha256=hashlib.sha256(sizing_path.read_bytes()).hexdigest(),
        parameters_source=params_path.as_posix(),
        parameters_sha256=hashlib.sha256(params_path.read_bytes()).hexdigest(),
    )
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


def outage_hazard_rate_per_hour(
    outage_probability_pct: float,
    reference_hours: float = 24.0,
) -> float:
    """Convert a reference-window outage probability into an hourly hazard."""
    p_ref = float(outage_probability_pct) / 100.0
    if not 0.0 <= p_ref < 1.0:
        raise ValueError("outage_probability_pct must be in [0, 100)")
    if reference_hours <= 0.0:
        raise ValueError("outage_probability_reference_hours must be > 0")
    if p_ref == 0.0:
        return 0.0
    return -math.log1p(-p_ref) / float(reference_hours)


def build_fixed_contingency_data(
    times: List[datetime],
    dt_h_map: Dict[datetime, float],
    horizon_hours: float,
    outage_duration_hours: float,
    outage_probability_pct: float,
    outage_probability_reference_hours: float = 24.0,
    contingency_spacing_hours: float = 2.0,
) -> Dict[str, Any]:
    """Build mesh-independent outage scenarios and first-arrival weights.

    Candidate starts lie on a fixed physical support. Every start and outage
    end must align with the model grid, which prevents coarse intervals from
    silently lengthening an outage. Only complete outages are represented.
    """
    if not times:
        raise ValueError("times must not be empty")
    if any(t not in dt_h_map for t in times):
        raise ValueError("dt_h_map must contain every timestamp in times")
    if any(float(dt_h_map[t]) <= 0.0 for t in times):
        raise ValueError("all model intervals must have positive duration")

    horizon_h = float(horizon_hours)
    outage_h = float(outage_duration_hours)
    spacing_h = float(contingency_spacing_hours)
    if horizon_h <= 0.0:
        raise ValueError("horizon_hours must be > 0")
    if outage_h < 0.0 or outage_h > horizon_h:
        raise ValueError("outage_duration_hours must be in [0, horizon_hours]")
    if spacing_h <= 0.0:
        raise ValueError("contingency_spacing_hours must be > 0")

    modeled_h = sum(float(dt_h_map[t]) for t in times)
    if not math.isclose(modeled_h, horizon_h, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"time grid spans {modeled_h:g} h, but horizon_hours={horizon_h:g}"
        )

    hazard = outage_hazard_rate_per_hour(
        outage_probability_pct,
        reference_hours=float(outage_probability_reference_hours),
    )
    eligible_h = max(0.0, horizon_h - outage_h)
    base = {
        "contingencies": ["c0"],
        "starts": [],
        "windows": {"c0": []},
        "before": {"c0": []},
        "pi_c": {"c0": 1.0},
        "hazard_rate_per_hour": hazard,
        "eligible_start_hours": eligible_h,
        "bins": {},
    }
    if hazard == 0.0 or outage_h == 0.0 or eligible_h <= 0.0:
        return base

    start_dt = times[0]
    time_index = {t: i for i, t in enumerate(times)}
    starts: List[datetime] = []
    windows: Dict[Any, List[datetime]] = {"c0": []}
    before: Dict[Any, List[datetime]] = {"c0": []}
    pi_c: Dict[Any, float] = {}
    bins: Dict[Any, Dict[str, float]] = {}

    offset_h = 0.0
    while offset_h < eligible_h - 1e-9:
        start = start_dt + timedelta(hours=offset_h)
        if start not in time_index:
            raise ValueError(
                "Fixed contingency support is not aligned with the time grid: "
                f"start offset {offset_h:g} h is missing. Choose mesh steps that "
                "divide EDS.contingency_spacing_hours."
            )

        end = start + timedelta(hours=outage_h)
        window = [t for t in times if start <= t < end]
        represented_h = sum(float(dt_h_map[t]) for t in window)
        if not math.isclose(represented_h, outage_h, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "Outage window is not represented exactly by the time grid: "
                f"start={start}, expected={outage_h:g} h, represented={represented_h:g} h. "
                "Choose mesh steps and contingency spacing that align outage starts and ends."
            )

        bin_end_h = min(offset_h + spacing_h, eligible_h)
        starts.append(start)
        windows[start] = window
        before[start] = times[: time_index[start]]
        pi_c[start] = math.exp(-hazard * offset_h) - math.exp(-hazard * bin_end_h)
        bins[start] = {"start_offset_h": offset_h, "end_offset_h": bin_end_h}
        offset_h += spacing_h

    pi_c["c0"] = math.exp(-hazard * eligible_h)
    total = sum(pi_c.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        pi_c["c0"] += 1.0 - total

    return {
        "contingencies": ["c0", *starts],
        "starts": starts,
        "windows": windows,
        "before": before,
        "pi_c": pi_c,
        "hazard_rate_per_hour": hazard,
        "eligible_start_hours": eligible_h,
        "bins": bins,
    }


def build_contingency_times(start_dt: datetime,
                            horizon_hours: int,
                            outage_duration_hours: int,
                            dt1_min: int,
                            dt2_min: int,
                            contingency_spacing_hours: Optional[float] = None) -> List[datetime]:
    """Compatibility wrapper returning the corrected fixed physical support."""
    dt_min = build_dt_vector(
        horizon_hours, outage_duration_hours, dt1_min, dt2_min
    )
    times = build_time_grid(start_dt, dt_min)
    dt_h_map = {t: float(dm) / 60.0 for t, dm in zip(times, dt_min)}
    data = build_fixed_contingency_data(
        times=times,
        dt_h_map=dt_h_map,
        horizon_hours=horizon_hours,
        outage_duration_hours=outage_duration_hours,
        outage_probability_pct=1.0,
        contingency_spacing_hours=float(
            contingency_spacing_hours
            if contingency_spacing_hours is not None
            else (outage_duration_hours or 2.0)
        ),
    )
    return list(data["starts"])


def build_time_and_contingencies_from_params(params: Dict[str, Any],
                                              start_dt: datetime) -> Tuple[List[datetime], List[datetime]]:
    horizon_hours = int(params["horizon_hours"])
    dt1_min = int(params["timestep_1_min"])
    dt2_min = int(params["timestep_2_min"])
    outage_duration_hours = int(params.get("outage_duration_hours", 0))

    dt_min = build_dt_vector(horizon_hours, outage_duration_hours, dt1_min, dt2_min)
    times = build_time_grid(start_dt, dt_min)
    contingencies = build_contingency_times(
        start_dt,
        horizon_hours,
        outage_duration_hours,
        dt1_min,
        dt2_min,
        contingency_spacing_hours=float(
            params.get("contingency_spacing_hours", outage_duration_hours or 2.0)
        ),
    )
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
