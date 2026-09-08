"""Closed-loop drivers for MPC and receding stochastic operation."""

from __future__ import annotations

import copy
import json
import hashlib
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from env.grid_env import GridEnv
from .artifacts import read_operation, write_operation, write_stochastic_plan
from .baseline import BASELINE_MODEL_VERSION, BaselineController, run_baseline_closed_loop
from .closed_loop import run_mpc_closed_loop
from .ongrid import OnGridMPC
from .stochastic import OnGridStochasticOperation
from .utils import build_dt_vector, outage_hazard_rate_per_hour

DEFAULT_LOAD_CSV = "data/load_5min_test.csv"
DEFAULT_PV_CSV = "data/pv_5min_test.csv"

DEFAULT_SOLVER_OPTS = {"time_limit": 120, "threads": 1, "mip_gap": 0.01}
DEFAULT_STOCH_SOLVER_OPTS = {"time_limit": 1200, "threads": 8, "mip_gap": 0.01}
OPERATION_MODEL_VERSION = "2026.09-evaluation-terminal-closure-v8"
COST_ACCOUNTING_VERSION = "realized-net-grid-reliability-plus-throughput-v2"
PARAMETER_FINGERPRINT_VERSION = "portable-sizing-provenance-v2"
IN_PROGRESS_MARKER = ".operation_in_progress.json"
CACHE_COMPATIBLE_OPERATION_MODELS = {
    "2026.08-export-revenue-v7": (
        "v8 removes the redundant generic terminal-energy constraint when "
        "an exact finite-evaluation terminal target is active; completed v7 "
        "runs without solver failures are unchanged"
    ),
    "2026.08-continuous-shed-strict-balance-v5": (
        "v6 adds a fail-only on-grid discharge correction after export and "
        "PV curtailment are exhausted"
    ),
}
CACHE_PHYSICAL_LIMITS = {
    "max_abs_power_balance_residual_kw": 1e-6,
    "max_grid_exchange_during_outage_kw": 1e-6,
    "max_bess_energy_bound_violation_kwh": 1e-9,
    "max_bess_energy_transition_residual_kwh": 1e-9,
    "max_plant_simultaneous_ch_dis_kw": 1e-6,
}


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def parameter_fingerprint(params: dict) -> str:
    """Portable hash used to invalidate cached operational artifacts.

    The sizing file hashes remain part of the payload. Only machine-local source
    paths are removed so artifacts copied between workstations retain identity.
    """
    portable = json.loads(json.dumps(params, ensure_ascii=False))
    sizing = portable.get("sizing_case_applied")
    if isinstance(sizing, dict):
        sizing.pop("source", None)
        sizing.pop("parameters_source", None)
    payload = json.dumps(
        portable, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_fingerprint(run_config: dict) -> str:
    """Stable identity for simulation settings that are not model parameters."""
    normalized = dict(run_config)
    if not bool(normalized.get("close_evaluation_energy", False)):
        # Preserve compatibility with all stochastic caches created before the
        # finite-evaluation closure option existed.
        normalized.pop("close_evaluation_energy", None)
        normalized.pop("terminal_closure_lookahead_h", None)
    return parameter_fingerprint(normalized)


def validate_time_mesh(params: dict) -> None:
    """Validate both the temporal grid and fixed physical outage support."""
    build_dt_vector(
        horizon_hours=int(params["time"]["horizon_hours"]),
        outage_duration_hours=int(params.get("EDS", {}).get("outage_duration_hours", 0)),
        dt1_min=int(params["time"]["timestep_1_min"]),
        dt2_min=int(params["time"]["timestep_2_min"]),
    )
    OnGridMPC(params, relaxation=True).param.build_time_data(datetime(2000, 1, 1))


def validate_sized_system(params: dict) -> None:
    """Require traceable sizing output before operational simulation."""
    info = params.get("sizing_case_applied")
    if not isinstance(info, dict):
        raise ValueError(
            "Operational simulations require a sizing result. "
            "Load parameters with opt.utils.load_sized_parameters()."
        )
    required = {
        "case", "source", "source_sha256", "operation_year",
        "P_hat_PV_kw", "E_hat_BESS_kwh", "BESS_Pmax_kw",
    }
    missing = sorted(required.difference(info))
    if missing:
        raise ValueError(f"Incomplete sizing provenance; missing fields: {missing}")
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
    _write_json(
        out_dir / IN_PROGRESS_MARKER,
        {
            "started_at": pd.Timestamp.now().isoformat(),
            "run_fingerprint_sha256": _run_fingerprint(run_config),
        },
    )
    payload = dict(params)
    payload["_operation_model"] = {
        "version": OPERATION_MODEL_VERSION,
        "parameter_fingerprint_sha256": parameter_fingerprint(params),
        "parameter_fingerprint_version": PARAMETER_FINGERPRINT_VERSION,
    }
    payload["_operation_run"] = {
        **run_config,
        "run_fingerprint_sha256": _run_fingerprint(run_config),
    }
    _write_json(out_dir / "parameters_used.json", payload)
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
    net_grid_reliability = float(pd.to_numeric(df["cost_total"], errors="coerce").sum())
    zero_cost = pd.Series(0.0, index=df.index)
    import_cost = float(pd.to_numeric(df.get("cost_grid", zero_cost), errors="coerce").sum())
    export_revenue = float(
        pd.to_numeric(df.get("revenue_export", zero_cost), errors="coerce").sum()
    )
    shedding_cost = float(
        pd.to_numeric(df.get("cost_shed", zero_cost), errors="coerce").sum()
    )
    curtailment_cost = float(
        pd.to_numeric(df.get("cost_curt", zero_cost), errors="coerce").sum()
    )
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
    total = float(net_grid_reliability + wear)
    return {
        "cost_accounting_version": COST_ACCOUNTING_VERSION,
        "operation_grid_reliability_cost": net_grid_reliability,
        "operation_net_grid_reliability_cost": net_grid_reliability,
        "operation_grid_import_cost": import_cost,
        "operation_export_revenue": export_revenue,
        "operation_net_grid_cost": import_cost - export_revenue,
        "operation_load_shedding_cost": shedding_cost,
        "operation_pv_curtailment_cost": curtailment_cost,
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
        parameter_fingerprint_version=PARAMETER_FINGERPRINT_VERSION,
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
        write_operation(df, out_dir)
        metrics["operation_rows"] = int(len(df))
        metrics.update(_operation_cost_metrics(df, env.p))
        metrics.update(_operation_physical_metrics(df, env.p))
    _write_json(out_dir / "metrics.json", metrics)
    (out_dir / IN_PROGRESS_MARKER).unlink(missing_ok=True)
    return metrics


def _cache_physics_passes(metrics: Dict[str, Any]) -> bool:
    for field, limit in CACHE_PHYSICAL_LIMITS.items():
        try:
            value = float(metrics[field])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(value) or value > limit:
            return False
    return True


def _migrate_cached_parameters(out_dir: Path, previous_version: str, reason: str) -> None:
    path = out_dir / "parameters_used.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    model = dict(payload.get("_operation_model", {}) or {})
    model.update(
        version=OPERATION_MODEL_VERSION,
        cache_compatible_from=previous_version,
        cache_compatibility_reason=reason,
    )
    payload["_operation_model"] = model
    _write_json(path, payload)


def _cached_metrics(
    out_dir: Path,
    params: dict,
    run_config: dict,
) -> Optional[Dict[str, Any]]:
    """Metrics of a finished case, or None (missing or corrupt -> re-run)."""
    if (Path(out_dir) / IN_PROGRESS_MARKER).exists():
        print(f'[incomplete] {Path(out_dir).name}: previous run did not finish; re-running case')
        return None
    path = Path(out_dir) / "metrics.json"
    if not path.exists():
        return None
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        cached_version = str(metrics.get("operation_model_version", ""))
        compatibility_reason = None
        if cached_version != OPERATION_MODEL_VERSION:
            compatibility_reason = CACHE_COMPATIBLE_OPERATION_MODELS.get(cached_version)
        if cached_version != OPERATION_MODEL_VERSION and compatibility_reason is None:
            print(f'[stale] {Path(out_dir).name}: operation model changed; re-running case')
            return None
        changed = False
        expected_params = parameter_fingerprint(params)
        if metrics.get("parameter_fingerprint_sha256") != expected_params:
            parameters_path = Path(out_dir) / "parameters_used.json"
            try:
                cached_params = json.loads(parameters_path.read_text(encoding="utf-8"))
                cached_params.pop("_operation_model", None)
                cached_params.pop("_operation_run", None)
                portable_cache_match = (
                    parameter_fingerprint(cached_params) == expected_params
                )
            except (OSError, json.JSONDecodeError):
                portable_cache_match = False
            if not portable_cache_match:
                print(f'[stale] {Path(out_dir).name}: parameters changed; re-running case')
                return None
            metrics.update(
                parameter_fingerprint_sha256=expected_params,
                parameter_fingerprint_version=PARAMETER_FINGERPRINT_VERSION,
            )
            changed = True
        expected_run = _run_fingerprint(run_config)
        if bool(run_config.get("close_evaluation_energy", False)):
            solver_failures = int(metrics.get("n_solve_fail", 0) or 0)
            if solver_failures:
                print(
                    f'[stale] {Path(out_dir).name}: cached terminal-closure run '
                    f'has {solver_failures} solver failure(s); re-running case'
                )
                return None
            closure_fields_valid = (
                metrics.get("close_evaluation_energy") is True
                and float(metrics.get("terminal_closure_lookahead_h", -1.0))
                == float(run_config["terminal_closure_lookahead_h"])
                and metrics.get("terminal_plan_committed_at") is not None
                and metrics.get("evaluation_terminal_energy_residual_kwh")
                is not None
            )
            if not closure_fields_valid:
                print(
                    f'[stale] {Path(out_dir).name}: terminal-closure metrics '
                    'are missing; re-running case'
                )
                return None
        if metrics.get("run_fingerprint_sha256") != expected_run:
            parameters_path = Path(out_dir) / "parameters_used.json"
            try:
                cached_payload = json.loads(
                    parameters_path.read_text(encoding="utf-8")
                )
                cached_run = dict(cached_payload.get("_operation_run", {}) or {})
                cached_run.pop("run_fingerprint_sha256", None)
                compatible_run = (
                    not bool(run_config.get("close_evaluation_energy", False))
                    and parameters_path.stat().st_mtime_ns
                    <= path.stat().st_mtime_ns
                    and _run_fingerprint(cached_run) == expected_run
                )
            except (OSError, json.JSONDecodeError):
                compatible_run = False
            if not compatible_run:
                print(f'[stale] {Path(out_dir).name}: run configuration changed; re-running case')
                return None
            metrics["run_fingerprint_sha256"] = expected_run
            changed = True
        expected = params['sizing_case_applied']['source_sha256']
        if metrics.get('sizing_source_sha256') != expected:
            print(f'[stale] {Path(out_dir).name}: sizing provenance changed; re-running case')
            return None
        try:
            operation_df = read_operation(out_dir)
            reconstructed = {
                **_operation_cost_metrics(operation_df, params),
                **_operation_physical_metrics(operation_df, params),
            }
            if any(metrics.get(key) != value for key, value in reconstructed.items()):
                metrics.update(reconstructed)
                changed = True
        except FileNotFoundError:
            if compatibility_reason is not None:
                print(f'[stale] {Path(out_dir).name}: compatible cache has no operation artifact')
                return None
        if compatibility_reason is not None:
            if not _cache_physics_passes(metrics):
                print(f'[stale] {Path(out_dir).name}: cached physical audit failed')
                return None
            metrics.update(
                operation_model_version=OPERATION_MODEL_VERSION,
                operation_model_compatible_from=cached_version,
                operation_model_compatibility_reason=compatibility_reason,
            )
            _migrate_cached_parameters(Path(out_dir), cached_version, compatibility_reason)
            changed = True
        if changed:
            _write_json(path, metrics)
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
    close_evaluation_energy: bool = False,
    terminal_closure_lookahead_h: float = 48.0,
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
        "close_evaluation_energy": bool(close_evaluation_energy),
        "terminal_closure_lookahead_h": float(terminal_closure_lookahead_h),
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
    evaluation_initial_energy_kwh = float(env.E_meas)
    evaluation_end_ts = pd.Timestamp(start_ts) + pd.Timedelta(
        hours=float(n_iters) * float(env.dt_h)
    )
    plans_dir = out_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    written_plan_names: set[str] = set()

    operation = OnGridStochasticOperation(params, relaxation=True)
    base_horizon_h = int(params["time"]["horizon_hours"])
    n_solves = 0
    n_solve_fail = 0
    n_fallback_steps = 0
    solve_time_total = 0.0
    max_simultaneity_kw = 0.0
    last_status, last_term = "none", "none"
    pbess_prev_kw = 0.0
    next_resolve = env.timestamp  # forces the initial solve
    last_resolve_failed = False
    terminal_plan_committed_at: str | None = None

    def _resolve(now: pd.Timestamp) -> None:
        nonlocal operation
        nonlocal n_solves, n_solve_fail, solve_time_total, next_resolve
        nonlocal max_simultaneity_kw, last_status, last_term
        nonlocal last_resolve_failed
        nonlocal terminal_plan_committed_at
        try:
            remaining_h = float(
                (evaluation_end_ts - now).total_seconds() / 3600.0
            )
            enforce_evaluation_terminal = bool(
                close_evaluation_energy
                and remaining_h <= float(terminal_closure_lookahead_h) + 1e-9
            )
            planning_params = params
            if enforce_evaluation_terminal:
                # Once the evaluation boundary enters the closure look-ahead,
                # keep it explicitly inside every subsequent optimization.
                # This avoids waiting until the last 36 h, when restoring the
                # initial inventory may already be physically impossible.
                required_h = max(base_horizon_h, int(math.ceil(remaining_h)))
                outage_h = int(params.get("EDS", {}).get("outage_duration_hours", 0))
                dt2_min = int(params["time"]["timestep_2_min"])
                while ((required_h - outage_h) * 60) % dt2_min:
                    required_h += 1
                if required_h != base_horizon_h:
                    planning_params = copy.deepcopy(params)
                    planning_params["time"]["horizon_hours"] = required_h
            candidate_operation = OnGridStochasticOperation(
                planning_params, relaxation=True
            )
            t0 = time.perf_counter()
            candidate_operation.build(
                start_dt=now.to_pydatetime(),
                forecasts=None,  # the stochastic model uses train clusters, not forecasts
                E_hat_kwh=float(env.E_meas),
                P_bess_hat_kw=pbess_prev_kw,
                evaluation_end_dt=(
                    evaluation_end_ts.to_pydatetime()
                    if enforce_evaluation_terminal
                    else None
                ),
                evaluation_terminal_energy_kwh=(
                    evaluation_initial_energy_kwh
                    if enforce_evaluation_terminal
                    else None
                ),
            )
            results = candidate_operation.solve(tee=False, **opts)
            dt = time.perf_counter() - t0
            solve_time_total += dt
            n_solves += 1
            last_status = str(results.solver.status)
            last_term = str(results.solver.termination_condition)
            print(
                f"[stochastic] solve {n_solves} at {now}: status={last_status} "
                f"term={last_term} time={dt:.1f}s"
            )
            written_plan = write_stochastic_plan(
                candidate_operation.extract_full_solution(),
                plans_dir / f"plan_{now.strftime('%Y%m%dT%H%M')}.parquet",
            )
            written_plan_names.add(written_plan.name)
            # relaxation exactness check: should stay at zero
            max_simultaneity_kw = max(
                max_simultaneity_kw,
                max(
                    (
                        min(a.get("P_ch_kw", 0.0), a.get("P_dis_kw", 0.0))
                        for a in candidate_operation._actions
                    ),
                    default=0.0,
                ),
            )
            operation = candidate_operation
            last_resolve_failed = False
            if enforce_evaluation_terminal:
                # Commit the first feasible plan that reaches the finite
                # evaluation boundary. Rebuilding the stochastic scenario set
                # closer to the boundary is not recursively feasible in
                # general, whereas the retained plan is feasible by construction.
                terminal_plan_committed_at = now.isoformat()
                next_resolve = evaluation_end_ts
            else:
                next_resolve = now + pd.Timedelta(hours=float(resolve_every_h))
        except Exception as e:
            n_solve_fail += 1
            last_resolve_failed = True
            print(f"[stochastic] WARN: solve failed at {now}: {e}. Retrying in 1h.")
            next_resolve = now + pd.Timedelta(hours=1.0)

    while not env.done():
        loop_t0 = time.perf_counter()
        now = env.timestamp

        # Re-solve on schedule or when the plan window ran out; never while islanded.
        action_available = bool(operation.get_control_at(now))
        if env.mode != "offgrid" and (
            now >= next_resolve or (not action_available and not last_resolve_failed)
        ):
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
        "close_evaluation_energy": bool(close_evaluation_energy),
        "terminal_closure_lookahead_h": float(terminal_closure_lookahead_h),
        "terminal_plan_committed_at": terminal_plan_committed_at,
        "evaluation_initial_energy_kwh": evaluation_initial_energy_kwh,
        "evaluation_final_energy_kwh": float(env.E_meas),
        "evaluation_terminal_energy_residual_kwh": (
            abs(float(env.E_meas) - evaluation_initial_energy_kwh)
            if close_evaluation_energy
            else None
        ),
        "n_solves": int(n_solves),
        "n_solve_fail": int(n_solve_fail),
        "n_fallback_steps": int(n_fallback_steps),
        "total_solve_time_s": float(solve_time_total),
        "avg_solve_time_s": float(solve_time_total / n_solves) if n_solves else None,
        "max_simultaneous_ch_dis_kw": float(max_simultaneity_kw),
        "run_start_ts": run_config["start_ts"],
        "run_fingerprint_sha256": _run_fingerprint(run_config),
    }
    completed_metrics = _finish_case(env, out_dir, metrics)
    # Only after the replacement trajectory and metrics have been committed,
    # remove plan files left by an older configuration. Interrupted reruns
    # therefore never destroy the last complete artifact set.
    for stale_plan in plans_dir.iterdir():
        if (
            stale_plan.is_file()
            and stale_plan.name not in written_plan_names
            and (
                stale_plan.name.startswith("plan_")
                or stale_plan.name.startswith(".plan_")
            )
        ):
            try:
                stale_plan.unlink()
            except OSError as error:
                print(f"[stochastic] WARN: could not remove stale plan {stale_plan}: {error}")
    return completed_metrics


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
    loop = run_mpc_closed_loop(
        env=env, mpc=mpc, forecaster=forecaster, solver_opts=opts, n_iters=int(n_iters),
        controller_label=forecaster_name or type(forecaster).__name__,
        progress_every=progress_every,
    )

    metrics = {
        "controller": "mpc",
        "forecaster": forecast_label,
        "forecaster_version": run_config.get("forecaster_version"),
        "status": "ok" if loop.n_solve_fail == 0 else "warning",
        "termination": "completed",
        "horizon_hours": int(params["time"]["horizon_hours"]),
        "timestep_1_min": int(params["time"]["timestep_1_min"]),
        "timestep_2_min": int(params["time"]["timestep_2_min"]),
        "n_iters": int(n_iters),
        "n_ongrid_steps": int(loop.n_ongrid_steps),
        "n_offgrid_steps": int(loop.n_offgrid_steps),
        "n_solve_ok": int(loop.n_solve_ok),
        "n_solve_fail": int(loop.n_solve_fail),
        "total_solve_time_s": float(loop.total_solve_time_s),
        "avg_solve_time_s": loop.average_solve_time_s,
        "total_time_s": float(loop.total_time_s),
        "max_simultaneous_ch_dis_kw": float(loop.max_simultaneous_ch_dis_kw),
        "run_start_ts": run_config["start_ts"],
        "run_fingerprint_sha256": _run_fingerprint(run_config),
    }
    return _finish_case(env, out_dir, metrics)


def simulate_baseline(
    params: dict,
    controller: BaselineController,
    start_ts,
    n_iters: int,
    out_dir,
    load_csv: str = DEFAULT_LOAD_CSV,
    pv_csv: str = DEFAULT_PV_CSV,
    progress_every: int = 288,
    resume: bool = True,
) -> Dict[str, Any]:
    """Run one causal rule-based controller through the common plant model."""
    validate_sized_system(params)
    run_config = {
        "controller": "baseline",
        "baseline_name": str(controller.name),
        "baseline_class": f"{type(controller).__module__}.{type(controller).__qualname__}",
        "baseline_version": str(controller.version),
        "baseline_model_version": BASELINE_MODEL_VERSION,
        "baseline_configuration": controller.configuration(),
        "start_ts": pd.Timestamp(start_ts).isoformat(),
        "n_iters": int(n_iters),
        "load_csv": str(load_csv),
        "pv_csv": str(pv_csv),
    }
    if resume:
        cached = _cached_metrics(out_dir, params, run_config)
        if cached is not None:
            print(f"[skip] {Path(out_dir).name}: already complete (metrics.json found)")
            return cached

    validate_time_mesh(params)
    out_dir = _prepare_case_dir(params, Path(out_dir), run_config)
    env = _make_env(params, load_csv, pv_csv, start_ts, n_iters, out_dir)
    loop = run_baseline_closed_loop(
        env=env,
        controller=controller,
        n_iters=int(n_iters),
        progress_every=progress_every,
    )
    metrics = {
        "controller": "baseline",
        "baseline_name": str(controller.name),
        "baseline_version": str(controller.version),
        "baseline_model_version": BASELINE_MODEL_VERSION,
        "baseline_configuration": controller.configuration(),
        "status": "ok",
        "termination": "completed",
        "horizon_hours": int(params["time"]["horizon_hours"]),
        "timestep_1_min": int(params["time"]["timestep_1_min"]),
        "timestep_2_min": int(params["time"]["timestep_2_min"]),
        "n_iters": int(n_iters),
        "n_ongrid_steps": int(loop.n_ongrid_steps),
        "n_offgrid_steps": int(loop.n_offgrid_steps),
        "n_solve_ok": 0,
        "n_solve_fail": 0,
        "total_solve_time_s": 0.0,
        "avg_solve_time_s": None,
        "total_action_time_s": float(loop.total_action_time_s),
        "avg_action_time_s": loop.average_action_time_s,
        "total_time_s": float(loop.total_time_s),
        "max_simultaneous_ch_dis_kw": 0.0,
        "run_start_ts": run_config["start_ts"],
        "run_fingerprint_sha256": _run_fingerprint(run_config),
    }
    return _finish_case(env, out_dir, metrics)
