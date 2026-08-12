# -*- coding: utf-8 -*-
"""Closed-loop simulation drivers: rolling-horizon MPC and receding stochastic plan.

Both run a GridEnv plant, write the case artifacts (parameters_used.json,
outage_calendar.json, operation_final.csv, metrics.json) and return the
metrics dict. Cases with an existing metrics.json are skipped (resume).
"""

from __future__ import annotations

import json
import hashlib
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from env.grid_env import GridEnv
from .ongrid import OnGridMPC
from .stochastic import OnGridStochasticOperation
from .utils import build_dt_vector, outage_hazard_rate_per_hour

DEFAULT_LOAD_CSV = "data/load_5min_test.csv"
DEFAULT_PV_CSV = "data/pv_5min_test.csv"

# Solver selection (Gurobi -> HiGHS) and fast method live in opt.utils.solve_model.
DEFAULT_SOLVER_OPTS = {"time_limit": 120, "threads": 1, "mip_gap": 0.01}
DEFAULT_STOCH_SOLVER_OPTS = {"time_limit": 1200, "threads": 8, "mip_gap": 0.01}
OPERATION_MODEL_VERSION = "2026.08-continuous-shed-strict-balance-v5"
COST_ACCOUNTING_VERSION = "realized-grid-reliability-plus-throughput-v1"


def parameter_fingerprint(params: dict) -> str:
    """Stable hash used to invalidate cached operational artifacts."""
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_fingerprint(run_config: dict) -> str:
    """Stable identity for simulation settings that are not model parameters."""
    return parameter_fingerprint(run_config)


def validate_time_mesh(params: dict) -> None:
    """Validate both the temporal grid and fixed physical outage support."""
    build_dt_vector(
        horizon_hours=int(params["time"]["horizon_hours"]),
        outage_duration_hours=int(params.get("EDS", {}).get("outage_duration_hours", 0)),
        dt1_min=int(params["time"]["timestep_1_min"]),
        dt2_min=int(params["time"]["timestep_2_min"]),
    )
    # Parameter preparation constructs the fixed outage support without
    # instantiating or calling a solver.
    OnGridMPC(params, relaxation=True).param.build_time_data(datetime(2000, 1, 1))


def validate_sized_system(params: dict) -> None:
    '''Require traceable sizing output before any operational simulation.'''
    info = params.get('sizing_case_applied')
    if not isinstance(info, dict):
        raise ValueError(
            'Operational simulations require a sizing result. '
            'Load parameters with opt.utils.load_sized_parameters().'
        )
    required = {
        'case', 'source', 'source_sha256', 'operation_year',
        'P_hat_PV_kw', 'E_hat_BESS_kwh', 'BESS_Pmax_kw',
    }
    missing = sorted(required.difference(info))
    if missing:
        raise ValueError(f'Incomplete sizing provenance; missing fields: {missing}')
    if int(info['operation_year']) != 1:
        raise ValueError('The current operational campaign must use sizing year 1')

    checks = {
        'PV.Pmax_kw': (float(params['PV']['Pmax_kw']), float(info['P_hat_PV_kw'])),
        'BESS.Emax_kwh': (float(params['BESS']['Emax_kwh']), float(info['E_hat_BESS_kwh'])),
        'BESS.Pmax_kw': (float(params['BESS']['Pmax_kw']), float(info['BESS_Pmax_kw'])),
    }
    inconsistent = [name for name, (actual, expected) in checks.items()
                    if abs(actual - expected) > 1e-9]
    if inconsistent:
        raise ValueError(f'Parameters do not match the applied sizing case: {inconsistent}')


def _prepare_case_dir(params: dict, out_dir: Path, run_config: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(params)
    payload["_operation_model"] = {
        "version": OPERATION_MODEL_VERSION,
        "parameter_fingerprint_sha256": parameter_fingerprint(params),
    }
    payload["_operation_run"] = {
        **run_config,
        "run_fingerprint_sha256": _run_fingerprint(run_config),
    }
    (out_dir / "parameters_used.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_dir


def _make_env(params: dict, load_csv: str, pv_csv: str,
              start_ts: pd.Timestamp, n_iters: int, out_dir: Path) -> GridEnv:
    env = GridEnv(
        params=params,
        load_csv=load_csv,
        pv_csv=pv_csv,
        start_dt0=pd.Timestamp(start_ts),
        n_iters=int(n_iters),
        debug=False,
    )
    env.save_outage_calendar(out_dir / "outage_calendar.json")
    return env


def _operation_cost_metrics(df: pd.DataFrame, params: dict) -> Dict[str, Any]:
    """Summarize realized plant costs, including marginal BESS wear."""
    if df.empty or "cost_total" not in df.columns:
        return {}
    grid_reliability = float(pd.to_numeric(df["cost_total"], errors="coerce").sum())
    required = ("P_bess_charge_mag_kw", "P_bess_discharge_kw")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Cannot reconstruct realized BESS wear; missing operation columns: "
            + ", ".join(missing)
        )
    charge = pd.to_numeric(df["P_bess_charge_mag_kw"], errors="coerce").fillna(0.0).clip(lower=0.0)
    discharge = pd.to_numeric(df["P_bess_discharge_kw"], errors="coerce").fillna(0.0).clip(lower=0.0)
    dt_h = float(params.get("time", {}).get("timestep", 5.0)) / 60.0
    throughput_kwh = float((charge + discharge).sum() * dt_h)
    c_deg = float(params.get("costs", {}).get("bess_degradation_per_kwh", 0.0))
    wear = float(c_deg * throughput_kwh)
    total = float(grid_reliability + wear)
    return {
        "cost_accounting_version": COST_ACCOUNTING_VERSION,
        "operation_grid_reliability_cost": grid_reliability,
        "operation_bess_throughput_kwh": throughput_kwh,
        "operation_wear_cost": wear,
        "operation_total_cost": total,
        "operation_mean_cost": total / len(df),
    }


def _operation_physical_metrics(df: pd.DataFrame, params: dict) -> Dict[str, Any]:
    """Audit realized balance, outage isolation, and BESS state transitions."""
    if df.empty:
        return {}
    required = {
        "Residual_kw",
        "P_grid_in_kw",
        "P_grid_out_kw",
        "outage_active",
        "E_kwh",
        "P_bess_charge_mag_kw",
        "P_bess_discharge_kw",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "Cannot audit realized plant physics; missing operation columns: "
            + ", ".join(missing)
        )

    def numeric(column: str) -> pd.Series:
        return pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    dt_h = float(params.get("time", {}).get("timestep", 5.0)) / 60.0
    residual = numeric("Residual_kw").abs()
    grid_exchange = numeric("P_grid_in_kw").abs() + numeric("P_grid_out_kw").abs()
    outage_raw = df["outage_active"]
    if pd.api.types.is_bool_dtype(outage_raw):
        outage = outage_raw.fillna(False)
    else:
        outage = outage_raw.astype(str).str.strip().str.lower().isin(
            ("true", "1", "yes")
        )

    bess = params.get("BESS", {}) or {}
    e_nom = float(bess.get("Emax_kwh", 0.0))
    e_min = (1.0 - float(bess.get("DoD_frac", 1.0))) * e_nom
    e_max = e_nom
    energy = numeric("E_kwh")
    energy_bound_violation = pd.concat(
        [(e_min - energy).clip(lower=0.0), (energy - e_max).clip(lower=0.0)],
        axis=1,
    ).max(axis=1)

    charge = numeric("P_bess_charge_mag_kw").clip(lower=0.0)
    discharge = numeric("P_bess_discharge_kw").clip(lower=0.0)
    previous_energy = energy.shift(1)
    previous_energy.iloc[0] = float(bess.get("E_init_kwh", energy.iloc[0]))
    expected_energy = previous_energy + dt_h * (
        float(bess.get("eta_c", 1.0)) * charge
        - discharge / max(float(bess.get("eta_d", 1.0)), 1e-12)
    )
    expected_energy = expected_energy.clip(lower=e_min, upper=e_max)
    transition_residual = (energy - expected_energy).abs()

    return {
        "max_abs_power_balance_residual_kw": float(residual.max()),
        "total_abs_power_balance_residual_kwh": float(residual.sum() * dt_h),
        "max_grid_exchange_during_outage_kw": float(
            grid_exchange.loc[outage].max() if outage.any() else 0.0
        ),
        "max_bess_energy_bound_violation_kwh": float(
            energy_bound_violation.max()
        ),
        "max_bess_energy_transition_residual_kwh": float(
            transition_residual.max()
        ),
        "max_plant_simultaneous_ch_dis_kw": float(
            pd.concat([charge, discharge], axis=1).min(axis=1).max()
        ),
    }


def _finish_case(env: GridEnv, out_dir: Path, metrics: Dict[str, Any]) -> Dict[str, Any]:
    sizing = env.p['sizing_case_applied']
    metrics.update(
        operation_model_version=OPERATION_MODEL_VERSION,
        parameter_fingerprint_sha256=parameter_fingerprint(env.p),
        sizing_case=sizing['case'],
        sizing_operation_year=int(sizing['operation_year']),
        sizing_source_sha256=sizing['source_sha256'],
        sized_pv_kw=float(env.p['PV']['Pmax_kw']),
        sized_bess_kwh=float(env.p['BESS']['Emax_kwh']),
        sized_bess_pmax_kw=float(env.p['BESS']['Pmax_kw']),
    )
    eds = env.p.get("EDS", {}) or {}
    bess = env.p.get("BESS", {}) or {}
    horizon_h = float(env.p.get("time", {}).get("horizon_hours", 0.0))
    outage_h = float(eds.get("outage_duration_hours", 0.0))
    hazard = outage_hazard_rate_per_hour(
        float(eds.get("outage_probability_pct", 0.0)),
        float(eds.get("outage_probability_reference_hours", 24.0)),
    )
    metrics.update(
        outage_hazard_rate_per_hour=float(hazard),
        modeled_outage_probability=float(
            1.0 - math.exp(-hazard * max(0.0, horizon_h - outage_h))
        ),
        contingency_spacing_hours=float(
            eds.get("contingency_spacing_hours", outage_h or 2.0)
        ),
        ramp_reference_min=float(
            bess.get("ramp_reference_min", env.p.get("time", {}).get("timestep", 5.0))
        ),
        terminal_energy_policy=str(bess.get("terminal_energy_policy", "initial")),
    )
    df = env.to_dataframe()
    if not df.empty:
        df.to_csv(out_dir / "operation_final.csv", index=True)
        metrics["operation_rows"] = int(len(df))
        metrics.update(_operation_cost_metrics(df, env.p))
        metrics.update(_operation_physical_metrics(df, env.p))
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


def _cached_metrics(
    out_dir: Path,
    params: dict,
    run_config: dict,
) -> Optional[Dict[str, Any]]:
    """Metrics of a finished case, or None (missing or corrupt -> re-run)."""
    path = Path(out_dir) / "metrics.json"
    if not path.exists():
        return None
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics.get("operation_model_version") != OPERATION_MODEL_VERSION:
            print(f'[stale] {Path(out_dir).name}: operation model changed; re-running case')
            return None
        expected_params = parameter_fingerprint(params)
        if metrics.get("parameter_fingerprint_sha256") != expected_params:
            print(f'[stale] {Path(out_dir).name}: parameters changed; re-running case')
            return None
        expected_run = _run_fingerprint(run_config)
        if metrics.get("run_fingerprint_sha256") != expected_run:
            print(f'[stale] {Path(out_dir).name}: run configuration changed; re-running case')
            return None
        expected = params['sizing_case_applied']['source_sha256']
        if metrics.get('sizing_source_sha256') != expected:
            print(f'[stale] {Path(out_dir).name}: sizing provenance changed; re-running case')
            return None
        operation_path = Path(out_dir) / "operation_final.csv"
        if operation_path.exists():
            operation_df = pd.read_csv(operation_path)
            reconstructed = {
                **_operation_cost_metrics(operation_df, params),
                **_operation_physical_metrics(operation_df, params),
            }
            if any(metrics.get(key) != value for key, value in reconstructed.items()):
                metrics.update(reconstructed)
                path.write_text(
                    json.dumps(metrics, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        return metrics
    except (OSError, json.JSONDecodeError):
        return None


def simulate_stochastic(
    params: dict,
    start_ts,
    n_iters: int,
    out_dir,
    load_csv: str = DEFAULT_LOAD_CSV,
    pv_csv: str = DEFAULT_PV_CSV,
    solver_opts: Optional[dict] = None,
    resolve_every_h: float = 24.0,
    resume: bool = True,
) -> Dict[str, Any]:
    """Receding stochastic plan: solve from the measured state, follow the
    plan open-loop (actions held over their model timestep), re-solve every
    `resolve_every_h` hours or when the plan window is exhausted."""
    validate_sized_system(params)
    opts = dict(solver_opts or DEFAULT_STOCH_SOLVER_OPTS)
    run_config = {
        "controller": "stochastic",
        "start_ts": pd.Timestamp(start_ts).isoformat(),
        "n_iters": int(n_iters),
        "load_csv": str(load_csv),
        "pv_csv": str(pv_csv),
        "resolve_every_h": float(resolve_every_h),
        "solver_opts": opts,
    }
    if resume:
        cached = _cached_metrics(out_dir, params, run_config)
        if cached is not None:
            print(f"[skip] {Path(out_dir).name}: already complete (metrics.json found)")
            return cached
    validate_time_mesh(params)
    out_dir = _prepare_case_dir(params, Path(out_dir), run_config)
    env = _make_env(params, load_csv, pv_csv, start_ts, n_iters, out_dir)
    plans_dir = out_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)

    operation = OnGridStochasticOperation(params, relaxation=True)
    n_solves = 0
    n_solve_fail = 0
    n_fallback_steps = 0
    solve_time_total = 0.0
    max_simultaneity_kw = 0.0
    last_status, last_term = "none", "none"
    pbess_prev_kw = 0.0
    next_resolve = env.timestamp  # forces the initial solve

    def _resolve(now: pd.Timestamp) -> None:
        nonlocal n_solves, n_solve_fail, solve_time_total, next_resolve
        nonlocal max_simultaneity_kw, last_status, last_term
        try:
            t0 = time.perf_counter()
            operation.build(
                start_dt=now.to_pydatetime(),
                forecasts=None,  # the stochastic model uses train clusters, not forecasts
                E_hat_kwh=float(env.E_meas),
                P_bess_hat_kw=pbess_prev_kw,
            )
            results = operation.solve(tee=False, **opts)
            dt = time.perf_counter() - t0
            solve_time_total += dt
            n_solves += 1
            last_status = str(results.solver.status)
            last_term = str(results.solver.termination_condition)
            print(
                f"[stochastic] solve {n_solves} at {now}: status={last_status} "
                f"term={last_term} time={dt:.1f}s"
            )
            (plans_dir / f"plan_{now.strftime('%Y%m%dT%H%M')}.json").write_text(
                json.dumps(operation.extract_full_solution(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # relaxation exactness check: should stay at zero
            max_simultaneity_kw = max(
                max_simultaneity_kw,
                max(
                    (min(a.get("P_ch_kw", 0.0), a.get("P_dis_kw", 0.0)) for a in operation._actions),
                    default=0.0,
                ),
            )
            next_resolve = now + pd.Timedelta(hours=float(resolve_every_h))
        except Exception as e:
            n_solve_fail += 1
            print(f"[stochastic] WARN: solve failed at {now}: {e}. Retrying in 1h.")
            next_resolve = now + pd.Timedelta(hours=1.0)

    while not env.done():
        loop_t0 = time.perf_counter()
        now = env.timestamp

        # Re-solve on schedule or when the plan window ran out; never while islanded.
        if env.mode != "offgrid" and (now >= next_resolve or not operation.get_control_at(now)):
            _resolve(now)

        action = operation.get_control_at(now)
        if action:
            P_bess = float(action.get("P_bess_kw", 0.0))
            X_L = float(action.get("X_L", 0.0))
            X_PV = float(action.get("X_PV", 0.0))
            obj = float(action.get("obj", 0.0))
        else:
            P_bess, X_L, X_PV, obj = 0.0, None, None, None  # safe fallback
            if env.mode != "offgrid":
                n_fallback_steps += 1
        row, done = env.step(
            P_bess_kw=P_bess, X_L=X_L, X_PV=X_PV, obj=obj,
            exec_time_sec=time.perf_counter() - loop_t0,
        )
        pbess_prev_kw = float(row.get("P_bess_kw", P_bess)) if isinstance(row, dict) else P_bess
        if done:
            break

    metrics = {
        "controller": "stochastic",
        "status": last_status,
        "termination": last_term,
        "horizon_hours": int(params["time"]["horizon_hours"]),
        "timestep_1_min": int(params["time"]["timestep_1_min"]),
        "timestep_2_min": int(params["time"]["timestep_2_min"]),
        "n_iters": int(n_iters),
        "resolve_every_h": float(resolve_every_h),
        "n_solves": int(n_solves),
        "n_solve_fail": int(n_solve_fail),
        "n_fallback_steps": int(n_fallback_steps),
        "total_solve_time_s": float(solve_time_total),
        "avg_solve_time_s": float(solve_time_total / n_solves) if n_solves else None,
        "max_simultaneous_ch_dis_kw": float(max_simultaneity_kw),
        "run_start_ts": run_config["start_ts"],
        "run_fingerprint_sha256": _run_fingerprint(run_config),
    }
    return _finish_case(env, out_dir, metrics)


def simulate_mpc(
    params: dict,
    forecaster,
    start_ts,
    n_iters: int,
    out_dir,
    load_csv: str = DEFAULT_LOAD_CSV,
    pv_csv: str = DEFAULT_PV_CSV,
    solver_opts: Optional[dict] = None,
    forecaster_name: str = "",
    progress_every: int = 288,
    resume: bool = True,
) -> Dict[str, Any]:
    """Rolling-horizon MPC in closed loop, re-solving at every on-grid step.

    `forecaster` is any object exposing
    get_forecasts(start_dt0, intervals=None, dt_min=..., include_actuals=False)
    -> {"load_kw": {ts: kW}, "pv_kw": {ts: kW}} aligned with env.dt_min.
    """
    validate_sized_system(params)
    opts = dict(solver_opts or DEFAULT_SOLVER_OPTS)
    forecast_label = forecaster_name or type(forecaster).__name__
    forecast_version = getattr(forecaster, "forecast_version", None)
    run_config = {
        "controller": "mpc",
        "forecaster": forecast_label,
        "forecaster_class": f"{type(forecaster).__module__}.{type(forecaster).__qualname__}",
        "start_ts": pd.Timestamp(start_ts).isoformat(),
        "n_iters": int(n_iters),
        "load_csv": str(load_csv),
        "pv_csv": str(pv_csv),
        "solver_opts": opts,
    }
    if forecast_version is not None:
        run_config["forecaster_version"] = str(forecast_version)
    if resume:
        cached = _cached_metrics(out_dir, params, run_config)
        if cached is not None:
            print(f"[skip] {Path(out_dir).name}: already complete (metrics.json found)")
            return cached
    validate_time_mesh(params)
    out_dir = _prepare_case_dir(params, Path(out_dir), run_config)
    env = _make_env(params, load_csv, pv_csv, start_ts, n_iters, out_dir)

    mpc = OnGridMPC(params, relaxation=True)
    pbess_prev_kw = 0.0
    n_ongrid = n_offgrid = n_solve_ok = n_solve_fail = 0
    solve_time_total = 0.0
    max_simultaneity_kw = 0.0
    run_t0 = time.perf_counter()

    while not env.done():
        loop_t0 = time.perf_counter()
        now = env.timestamp

        if env.mode == "offgrid":
            n_offgrid += 1
            step0 = None
        else:
            n_ongrid += 1
            try:
                forecasts = forecaster.get_forecasts(
                    start_dt0=now, intervals=None, dt_min=env.dt_min, include_actuals=False
                )
                if forecasts is None:
                    raise RuntimeError("forecast window out of data range")

                solve_t0 = time.perf_counter()
                mpc.build(
                    start_dt=now.to_pydatetime(),
                    forecasts=forecasts,
                    E_hat_kwh=float(env.E_meas),
                    P_bess_hat_kw=pbess_prev_kw,
                )
                mpc.solve(tee=False, **opts)
                solve_time_total += time.perf_counter() - solve_t0
                n_solve_ok += 1
                max_simultaneity_kw = max(
                    max_simultaneity_kw,
                    mpc.max_simultaneous_charge_discharge_kw(),
                )
                step0 = mpc.extract_first_step(scenario="c0")
            except Exception as e:
                n_solve_fail += 1
                print(f"[mpc] WARN: solve failed at {now}: {e}. Using safe fallback.")
                step0 = None

        if step0 is None:
            P_bess, X_L, X_PV, obj = 0.0, None, None, None
        else:
            P_bess = float(step0.get("P_bess_kw", 0.0))
            X_L = float(step0.get("X_L", 0.0))
            X_PV = float(step0.get("X_PV", 0.0))
            obj = float(step0.get("obj", 0.0))

        row, done = env.step(
            P_bess_kw=P_bess, X_L=X_L, X_PV=X_PV, obj=obj,
            exec_time_sec=time.perf_counter() - loop_t0,
        )
        pbess_prev_kw = float(row.get("P_bess_kw", P_bess)) if isinstance(row, dict) else P_bess

        if env.iter_k % max(1, progress_every) == 0 or done:
            elapsed = time.perf_counter() - run_t0
            avg = solve_time_total / n_solve_ok if n_solve_ok else 0.0
            print(
                f"[mpc] {forecaster_name or type(forecaster).__name__} "
                f"{env.iter_k}/{n_iters} | elapsed={elapsed:.0f}s avg_solve={avg:.2f}s "
                f"ok={n_solve_ok} fail={n_solve_fail} offgrid={n_offgrid}"
            )
        if done:
            break

    metrics = {
        "controller": "mpc",
        "forecaster": forecast_label,
        "forecaster_version": run_config.get("forecaster_version"),
        "status": "ok" if n_solve_fail == 0 else "warning",
        "termination": "completed",
        "horizon_hours": int(params["time"]["horizon_hours"]),
        "timestep_1_min": int(params["time"]["timestep_1_min"]),
        "timestep_2_min": int(params["time"]["timestep_2_min"]),
        "n_iters": int(n_iters),
        "n_ongrid_steps": int(n_ongrid),
        "n_offgrid_steps": int(n_offgrid),
        "n_solve_ok": int(n_solve_ok),
        "n_solve_fail": int(n_solve_fail),
        "total_solve_time_s": float(solve_time_total),
        "avg_solve_time_s": float(solve_time_total / n_solve_ok) if n_solve_ok else None,
        "total_time_s": float(time.perf_counter() - run_t0),
        "max_simultaneous_ch_dis_kw": float(max_simultaneity_kw),
        "run_start_ts": run_config["start_ts"],
        "run_fingerprint_sha256": _run_fingerprint(run_config),
    }
    return _finish_case(env, out_dir, metrics)
