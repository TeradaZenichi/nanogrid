# -*- coding: utf-8 -*-
"""
Closed-loop operation simulators (the logic of the former root pipelines).

Two entry points, both driving a GridEnv plant and saving the same artifacts
(parameters_used.json, outage_calendar.json, operation_final.csv, metrics.json):

- simulate_stochastic: solve the scenario-based model ONCE at the start,
  cache the shared control trajectory, then consume it step by step
  (open-loop execution of a stochastic plan).

- simulate_mpc: rolling-horizon MPC; at every on-grid step asks the given
  `forecaster` for a window (PerfectForecast = oracle, ForecastMPC = LSTM,
  PrototypeForecast = analog day, HybridForecast = mix) and re-solves.

Both return the metrics dict that is also written to <out_dir>/metrics.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from env.grid_env import GridEnv
from .ongrid import OnGridMPC
from .stochastic import OnGridStochasticOperation
from .utils import build_dt_vector

DEFAULT_LOAD_CSV = "data/load_5min_test.csv"
DEFAULT_PV_CSV = "data/pv_5min_test.csv"

# Solver selection (Gurobi -> HiGHS) and fast method live in opt.utils.solve_model.
DEFAULT_SOLVER_OPTS = {"time_limit": 120, "threads": 1, "mip_gap": 0.01}
DEFAULT_STOCH_SOLVER_OPTS = {"time_limit": 1200, "threads": 8, "mip_gap": 0.01}


def validate_time_mesh(params: dict) -> None:
    """Raise early (with the solver untouched) if the (h, dt1, dt2) combo is invalid."""
    build_dt_vector(
        horizon_hours=int(params["time"]["horizon_hours"]),
        outage_duration_hours=int(params.get("EDS", {}).get("outage_duration_hours", 0)),
        dt1_min=int(params["time"]["timestep_1_min"]),
        dt2_min=int(params["time"]["timestep_2_min"]),
    )


def _prepare_case_dir(params: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parameters_used.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
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


def _finish_case(env: GridEnv, out_dir: Path, metrics: Dict[str, Any]) -> Dict[str, Any]:
    df = env.to_dataframe()
    if not df.empty:
        df.to_csv(out_dir / "operation_final.csv", index=True)
        metrics["operation_rows"] = int(len(df))
        if "cost_total" in df.columns:
            metrics["operation_total_cost"] = float(df["cost_total"].sum())
            metrics["operation_mean_cost"] = float(df["cost_total"].mean())
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


def simulate_stochastic(
    params: dict,
    start_ts,
    n_iters: int,
    out_dir,
    load_csv: str = DEFAULT_LOAD_CSV,
    pv_csv: str = DEFAULT_PV_CSV,
    solver_opts: Optional[dict] = None,
) -> Dict[str, Any]:
    """Solve the scenario model once and execute the cached plan in closed loop."""
    validate_time_mesh(params)
    out_dir = _prepare_case_dir(params, Path(out_dir))
    env = _make_env(params, load_csv, pv_csv, start_ts, n_iters, out_dir)

    operation = OnGridStochasticOperation(params, relaxation=True)

    solve_t0 = time.perf_counter()
    operation.build(
        start_dt=env.timestamp.to_pydatetime(),
        forecasts=None,  # the stochastic model uses train clusters, not forecasts
        E_hat_kwh=float(env.E_meas),
        P_bess_hat_kw=0.0,
    )
    results = operation.solve(tee=False, **(solver_opts or DEFAULT_STOCH_SOLVER_OPTS))
    solve_time_s = time.perf_counter() - solve_t0
    print(
        f"[stochastic] solve: status={results.solver.status} "
        f"term={results.solver.termination_condition} time={solve_time_s:.1f}s"
    )

    plan = operation.extract_full_solution()
    (out_dir / "stochastic_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Exactness audit of the Extn-LP relaxation (Pozo et al.): simultaneous
    # charge/discharge should never appear at the optimum.
    max_simultaneity_kw = max(
        (min(a.get("P_ch_kw", 0.0), a.get("P_dis_kw", 0.0)) for a in operation._actions),
        default=0.0,
    )

    # Consume the cached trajectory step by step (open loop: no re-solve).
    while not env.done():
        loop_t0 = time.perf_counter()
        action = operation.get_control(advance=True)
        if action:
            P_bess = float(action.get("P_bess_kw", 0.0))
            X_L = float(action.get("X_L", 0.0))
            X_PV = float(action.get("X_PV", 0.0))
            obj = float(action.get("obj", 0.0))
        else:
            P_bess, X_L, X_PV, obj = 0.0, None, None, None  # safe fallback
        _, done = env.step(
            P_bess_kw=P_bess, X_L=X_L, X_PV=X_PV, obj=obj,
            exec_time_sec=time.perf_counter() - loop_t0,
        )
        if done:
            break

    metrics = {
        "controller": "stochastic",
        "status": str(results.solver.status),
        "termination": str(results.solver.termination_condition),
        "horizon_hours": int(params["time"]["horizon_hours"]),
        "timestep_1_min": int(params["time"]["timestep_1_min"]),
        "timestep_2_min": int(params["time"]["timestep_2_min"]),
        "n_iters": int(n_iters),
        "solve_time_s": float(solve_time_s),
        "max_simultaneous_ch_dis_kw": float(max_simultaneity_kw),
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
) -> Dict[str, Any]:
    """Rolling-horizon MPC in closed loop, re-solving at every on-grid step.

    `forecaster` is any object exposing
    get_forecasts(start_dt0, intervals=None, dt_min=..., include_actuals=False)
    -> {"load_kw": {ts: kW}, "pv_kw": {ts: kW}} aligned with env.dt_min.
    """
    validate_time_mesh(params)
    out_dir = _prepare_case_dir(params, Path(out_dir))
    env = _make_env(params, load_csv, pv_csv, start_ts, n_iters, out_dir)

    mpc = OnGridMPC(params, relaxation=True)
    opts = solver_opts or DEFAULT_SOLVER_OPTS

    pbess_prev_kw = 0.0
    n_ongrid = n_offgrid = n_solve_ok = n_solve_fail = 0
    solve_time_total = 0.0
    max_simultaneity_kw = 0.0  # exactness audit of the Extn-LP relaxation
    run_t0 = time.perf_counter()

    while not env.done():
        loop_t0 = time.perf_counter()
        now = env.timestamp

        if env.mode == "offgrid":
            # The plant follows the islanding balance by itself; no solve.
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
            max_simultaneity_kw = max(
                max_simultaneity_kw,
                min(float(step0.get("P_ch_kw", 0.0)), float(step0.get("P_dis_kw", 0.0))),
            )

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
        "forecaster": forecaster_name or type(forecaster).__name__,
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
    }
    return _finish_case(env, out_dir, metrics)
