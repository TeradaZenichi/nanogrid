"""Run the paper's corrected and resumable operational campaigns."""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from forecasting import PerfectForecast, PrototypeForecast
from forecasting.causal import FORECAST_ASSEMBLY_VERSION
from opt import simulate_mpc, simulate_stochastic
from opt.artifacts import operation_artifact_path, read_operation
from opt.campaigns import sha256_file
from opt.operation import (
    COST_ACCOUNTING_VERSION,
    OPERATION_MODEL_VERSION,
    _operation_cost_metrics,
    parameter_fingerprint,
    validate_time_mesh,
)
from opt.utils import load_series_scaled, load_sized_parameters, load_sized_parameters_from_artifact


PARAMS_JSON = str(ROOT / "data" / "parameters.json")
LOAD_CSV = str(ROOT / "data" / "load_5min_test.csv")
PV_CSV = str(ROOT / "data" / "pv_5min_test.csv")
SIZING_ROOT = ROOT / "outputs" / "sizing-sweep" / "with-degradation"
OPERATION_ROOT = ROOT / "outputs" / "operation-sweep" / "with-degradation"
DEFAULT_OUT_ROOT = OPERATION_ROOT
DEFAULT_SMOKE_ROOT = ROOT / "outputs" / "_smoke-pipeline"
DEFAULT_N_ITERS = 2880
STAGE = "all"
WORKERS = 2
N_ITERS = DEFAULT_N_ITERS
SMOKE_ITERS = 3
TOP_K = 3
FORECAST_CONTROLLERS = "ideal,prototype,lstm"
FRESH = False
SIZING_CASES = ("economic", "critical_50", "full_100")
PARAMETERS_ENV = "NANOGRID_OPERATION_PARAMETERS"
SIZING_ARTIFACT_ENV = "NANOGRID_OPERATION_SIZING_ARTIFACT"
CAMPAIGN_ID_ENV = "NANOGRID_OPERATION_CAMPAIGN_ID"

MESH_H = (12, 24, 36)
MESH_T1 = (5, 10, 15)
MESH_T2 = (30, 60, 120)
EVALUATION_CONTROLLERS = ("ideal", "prototype", "lstm")
SELECTION_CONTROLLERS = ("prototype", "lstm")
MESH_CONTROLLERS = EVALUATION_CONTROLLERS  # Backward-compatible public alias.
MESH_SELECTION_VERSION = "operational-forecaster-regret-annual-wear-v5"
MESH_SELECTION_WEIGHTS = {"mean_regret": 0.4, "max_regret": 0.4, "solve_time": 0.2}

CAUSAL_PILOT_MESHES = ({"h": 12, "t1": 5, "t2": 30}, {"h": 12, "t1": 5, "t2": 120}, {"h": 36, "t1": 5, "t2": 30})
MONTHS = (
    "2009-05",
    "2009-06",
    "2009-07",
    "2009-08",
    "2009-09",
    "2009-10",
    "2009-11",
    "2009-12",
    "2010-01",
    "2010-02",
    "2010-03",
    "2010-04",
)
MONTHLY_VALIDATION_WINDOWS = {month: f"{month}-01 00:00:00" for month in MONTHS}
BASE_OUTAGE_SEED = 42
MONTHLY_OUTAGE_SEEDS = {month: BASE_OUTAGE_SEED + i for i, month in enumerate(MONTHS)}
RECOURSE_WINDOWS = (
    ("may_seed42", "2009-05-01 00:00:00", 42),
    ("may_seed44", "2009-05-01 00:00:00", 44),
    ("aug_seed42", "2009-08-01 00:00:00", 42),
    ("nov_seed42", "2009-11-01 00:00:00", 42),
)
ROBUSTNESS_WINDOWS = (
    ("may", "2009-05-01 00:00:00", MONTHLY_OUTAGE_SEEDS["2009-05"]),
    ("aug", "2009-08-01 00:00:00", MONTHLY_OUTAGE_SEEDS["2009-08"]),
    ("nov", "2009-11-01 00:00:00", MONTHLY_OUTAGE_SEEDS["2009-11"]),
    ("feb", "2010-02-01 00:00:00", MONTHLY_OUTAGE_SEEDS["2010-02"]),
)
ROBUSTNESS_VARIANTS = (
    ("base", None, {}),
    ("outage_2pct", None, {"EDS": {"outage_probability_pct": 2.0}}),
    ("outage_5pct", None, {"EDS": {"outage_probability_pct": 5.0}}),
    ("seed_44", 44, {}),
    ("seed_47", 47, {}),
    (
        "noise_005",
        None,
        {"BESS": {"noisy": True, "noise": {"type": "gauss", "std_frac": 0.05, "std_kw": 0.0, "seed": 123}}},
    ),
    (
        "noise_010",
        None,
        {"BESS": {"noisy": True, "noise": {"type": "gauss", "std_frac": 0.10, "std_kw": 0.0, "seed": 123}}},
    ),
)


def _mesh_tag(h: int, t1: int, t2: int) -> str:
    return f"h{h}_t1_{t1}_t2_{t2}"


def _base_params(
    mesh: dict[str, int],
    seed: int | None = None,
    parameters_json: str | Path | None = None,
    sizing_artifact: str | Path | None = None,
) -> dict:
    parameters_json = parameters_json or os.environ.get(PARAMETERS_ENV, PARAMS_JSON)
    sizing_artifact = sizing_artifact or os.environ.get(SIZING_ARTIFACT_ENV)
    if sizing_artifact:
        params = load_sized_parameters_from_artifact(parameters_json, sizing_artifact)
    else:
        params = load_sized_parameters(str(parameters_json))
    params["time"].update(horizon_hours=int(mesh["h"]), timestep_1_min=int(mesh["t1"]), timestep_2_min=int(mesh["t2"]))
    if seed is not None:
        params["EDS"]["seed"] = int(seed)
    validate_time_mesh(params)
    return params


def _forecaster(name: str, params: dict):
    scaling = {"P_L_nom_kw": float(params["Load"]["Pmax_kw"]), "P_PV_nom_kw": float(params["PV"]["Pmax_kw"])}
    load_s, pv_s = load_series_scaled(scaling, LOAD_CSV, PV_CSV)
    if name == "ideal":
        return PerfectForecast(load_s, pv_s), "ideal"
    if name == "prototype":
        forecast = PrototypeForecast(
            None, load_s, pv_s, float(params["PV"]["Pmax_kw"]), float(params["Load"]["Pmax_kw"]), strategy="prefix"
        )
        return forecast, "prototype-prefix"
    if name == "lstm":
        from forecasting.get_forecasting import ForecastMPC

        forecast = ForecastMPC({}, load_s, pv_s, float(params["PV"]["Pmax_kw"]), float(params["Load"]["Pmax_kw"]))
        return forecast, "lstm"
    raise ValueError(f"Unknown controller: {name}")


def _audit_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []

    def value(name: str) -> float:
        raw = metrics.get(name, 0.0)
        if raw is None or pd.isna(raw):
            return 0.0
        return float(raw)

    solver_failures = int(metrics.get("n_solve_fail", 0) or 0)
    if solver_failures:
        reasons.append(f"{solver_failures} solver failure(s)")

    overlap = max(value("max_simultaneous_ch_dis_kw"), value("max_plant_simultaneous_ch_dis_kw"))
    if overlap > 1e-6:
        reasons.append(f"charge/discharge overlap {overlap:.12g} kW")

    checks = (
        ("max_abs_power_balance_residual_kw", 1e-6, "power-balance residual", "kW"),
        ("max_grid_exchange_during_outage_kw", 1e-9, "outage grid exchange", "kW"),
        ("max_bess_energy_bound_violation_kwh", 1e-9, "BESS energy-bound violation", "kWh"),
        ("max_bess_energy_transition_residual_kwh", 1e-9, "BESS transition residual", "kWh"),
    )
    for metric, tolerance, label, unit in checks:
        actual = value(metric)
        if actual > tolerance:
            reasons.append(f"{label} {actual:.12g} {unit}")

    terminal_residual = metrics.get("evaluation_terminal_energy_residual_kwh")
    if (
        terminal_residual is not None
        and not pd.isna(terminal_residual)
        and float(terminal_residual) > 1e-6
    ):
        reasons.append(
            f"evaluation terminal-energy residual {float(terminal_residual):.12g} kWh"
        )

    return {
        "audit_pass": not reasons,
        "audit_failures": len(reasons),
        "audit_solver_failures": solver_failures,
        "audit_reasons": " | ".join(reasons),
        "audit_max_simultaneous_ch_dis_kw": overlap,
        "audit_max_abs_power_balance_residual_kw": value("max_abs_power_balance_residual_kw"),
        "audit_max_grid_exchange_during_outage_kw": value("max_grid_exchange_during_outage_kw"),
        "audit_max_bess_energy_bound_violation_kwh": value("max_bess_energy_bound_violation_kwh"),
        "audit_max_bess_energy_transition_residual_kwh": value("max_bess_energy_transition_residual_kwh"),
    }


def execute_task(task: dict[str, Any]) -> dict[str, Any]:
    params = _base_params(
        task["mesh"],
        seed=task.get("seed"),
        parameters_json=task.get("parameters_json"),
        sizing_artifact=task.get("sizing_artifact"),
    )
    for section, values in task.get("overrides", {}).items():
        params[section].update(values)
    validate_time_mesh(params)

    out_dir = Path(task["out_dir"])
    controller = str(task["controller"])
    start = pd.Timestamp(task["start"])
    n_iters = int(task["n_iters"])
    if controller == "stochastic":
        metrics = simulate_stochastic(
            params,
            start,
            n_iters,
            out_dir,
            resolve_every_h=float(task.get("resolve_every_h", 24.0)),
            close_evaluation_energy=bool(task.get("close_evaluation_energy", False)),
            terminal_closure_lookahead_h=float(
                task.get("terminal_closure_lookahead_h", 48.0)
            ),
            resume=bool(task.get("resume", True)),
        )
        controller_name = "stochastic"
    else:
        forecast, forecast_name = _forecaster(controller, params)
        metrics = simulate_mpc(
            params,
            forecast,
            start,
            n_iters,
            out_dir,
            forecaster_name=forecast_name,
            resume=bool(task.get("resume", True)),
        )
        controller_name = controller

    metrics.update(
        case_id=str(task["case_id"]),
        stage=str(task["stage"]),
        controller_name=controller_name,
        start=str(start),
        h=int(task["mesh"]["h"]),
        t1=int(task["mesh"]["t1"]),
        t2=int(task["mesh"]["t2"]),
        combo=_mesh_tag(**task["mesh"]),
        campaign_id=str(task.get("campaign_id", "economic")),
        sizing_case=str(params["sizing_case_applied"]["case"]),
        sizing_source_sha256=str(params["sizing_case_applied"]["source_sha256"]),
    )
    metrics.update(task.get("labels", {}))
    metrics.update(_audit_metrics(metrics))
    return metrics


def _write_progress(
    path: Path, stage: str, completed: int, total: int, started: float, recovered_rows: int = 0
) -> None:
    elapsed = time.perf_counter() - started
    eta = (elapsed / completed) * (total - completed) if completed else None
    payload = {
        "stage": stage,
        "completed": completed,
        "total": total,
        "recovered_summary_rows": int(recovered_rows),
        "elapsed_s": elapsed,
        "eta_s": eta,
        "operation_model_version": OPERATION_MODEL_VERSION,
        "updated_at": pd.Timestamp.now().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _replace_with_retry(temporary, path)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    _replace_with_retry(temporary, path)


def _replace_with_retry(temporary: Path, destination: Path) -> None:
    for attempt in range(10):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


def _existing_summary_rows(path: Path, case_ids: set[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        summary = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return []
    if "case_id" not in summary.columns:
        return []
    summary["case_id"] = summary["case_id"].astype(str)
    summary = summary[summary["case_id"].isin(case_ids)]
    summary = summary.drop_duplicates(subset="case_id", keep="last")
    return summary.to_dict(orient="records")


def run_tasks(tasks: list[dict[str, Any]], workers: int, summary_path: Path) -> pd.DataFrame:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    stage = str(tasks[0]["stage"]) if tasks else summary_path.stem
    state_path = summary_path.parent / "pipeline_state.json"
    started = time.perf_counter()
    context = {
        "parameters_json": os.environ.get(PARAMETERS_ENV, PARAMS_JSON),
        "sizing_artifact": os.environ.get(SIZING_ARTIFACT_ENV),
        "campaign_id": os.environ.get(CAMPAIGN_ID_ENV, "reference"),
    }
    for task in tasks:
        for key, value in context.items():
            if value is not None:
                task.setdefault(key, value)

    case_ids = {str(task["case_id"]) for task in tasks}
    recovered = _existing_summary_rows(summary_path, case_ids)
    rows_by_case = {str(row["case_id"]): row for row in recovered}
    processed = 0
    if recovered:
        print(f"[resume] preserving {len(recovered)} row(s) from {summary_path.name}", flush=True)

    def record(task: dict[str, Any], result=None, error: Exception | None = None) -> None:
        nonlocal processed
        if error is None:
            row = result
            status = "ok"
        else:
            row = {
                "case_id": task["case_id"],
                "stage": task["stage"],
                "controller_name": task["controller"],
                "combo": _mesh_tag(**task["mesh"]),
                "campaign_id": task["campaign_id"],
                "sizing_source_sha256": (
                    sha256_file(Path(task["sizing_artifact"])) if task.get("sizing_artifact") else ""
                ),
                **task.get("labels", {}),
                "status": "error",
                "audit_pass": False,
                "error": str(error),
            }
            status = f"FAILED: {error}"
        rows_by_case[str(task["case_id"])] = row
        processed += 1
        rows = list(rows_by_case.values())
        _write_summary(summary_path, rows)
        _write_progress(state_path, stage, processed, len(tasks), started, recovered_rows=len(recovered))
        print(f"[{processed}/{len(tasks)} | summary={len(rows)}] {task['case_id']}: {status}", flush=True)

    if workers <= 1:
        for task in tasks:
            try:
                record(task, result=execute_task(task))
            except Exception as exc:
                record(task, error=exc)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = {pool.submit(execute_task, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    record(task, result=future.result())
                except Exception as exc:
                    record(task, error=exc)
    return pd.DataFrame(rows_by_case.values())


def _task(
    stage: str,
    controller: str,
    mesh: dict[str, int],
    start: str,
    n_iters: int,
    out_dir: Path,
    labels: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    case_id = "/".join(str(x) for x in (stage, *(labels or {}).values(), controller, _mesh_tag(**mesh)))
    return {
        "case_id": case_id,
        "stage": stage,
        "controller": controller,
        "mesh": dict(mesh),
        "start": start,
        "n_iters": int(n_iters),
        "out_dir": str(out_dir),
        "labels": labels or {},
        **kwargs,
    }


def _normalize(series: pd.Series) -> pd.Series:
    lo = float(series.min())
    hi = float(series.max())
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def _candidate_table(
    summary: pd.DataFrame,
    required_controllers: Iterable[str] = SELECTION_CONTROLLERS,
    expected_runs_per_controller: int = 1,
) -> pd.DataFrame:
    required_controllers = tuple(dict.fromkeys(str(x) for x in required_controllers))
    if not required_controllers:
        raise ValueError("At least one controller is required for mesh selection")
    expected_runs_per_controller = int(expected_runs_per_controller)
    if expected_runs_per_controller < 1:
        raise ValueError("expected_runs_per_controller must be positive")

    df = summary.copy()
    required_columns = {
        "combo",
        "h",
        "t1",
        "t2",
        "controller_name",
        "audit_pass",
        "operation_total_cost",
        "avg_solve_time_s",
        "case_id",
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise RuntimeError("Mesh summary is missing required columns: " + ", ".join(missing_columns))
    for col in ("operation_total_cost", "avg_solve_time_s", "n_solve_fail"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    relevant = df[df["controller_name"].isin(required_controllers)].copy()
    df = relevant[
        (relevant["audit_pass"] == True)  # noqa: E712
        & relevant["operation_total_cost"].notna()
        & relevant["avg_solve_time_s"].notna()
    ].copy()

    available_controllers = set(df["controller_name"].unique())
    missing_controllers = [controller for controller in required_controllers if controller not in available_controllers]
    if missing_controllers:
        details = []
        for controller in missing_controllers:
            controller_rows = relevant[relevant["controller_name"] == controller]
            if controller_rows.empty:
                details.append(f"{controller} (no simulation rows)")
                continue
            rejected = int((controller_rows["audit_pass"] != True).sum())  # noqa: E712
            invalid_metrics = int(
                (
                    (controller_rows["audit_pass"] == True)  # noqa: E712
                    & (controller_rows["operation_total_cost"].isna() | controller_rows["avg_solve_time_s"].isna())
                ).sum()
            )
            overlap = pd.to_numeric(
                controller_rows.get("max_simultaneous_ch_dis_kw", pd.Series(index=controller_rows.index, dtype=float)),
                errors="coerce",
            )
            detail = f"{controller} ({rejected} rejected by audit"
            if invalid_metrics:
                detail += f"; {invalid_metrics} with incomplete metrics"
            if overlap.notna().any():
                detail += f"; maximum charge/discharge overlap {float(overlap.max()):.6g} kW"
            balance = pd.to_numeric(
                controller_rows.get(
                    "max_abs_power_balance_residual_kw", pd.Series(index=controller_rows.index, dtype=float)
                ),
                errors="coerce",
            )
            if balance.notna().any():
                detail += f"; maximum power-balance residual {float(balance.max()):.6g} kW"
            details.append(detail + ")")
        raise RuntimeError(
            "Mesh simulations completed, but selection requires at least one audited "
            "case for every forecaster. No audited cases for: "
            + ", ".join(details)
            + ". Inspect the physical audit metrics before rerunning selection."
        )

    keys = ["combo", "h", "t1", "t2"]
    by_controller = df.groupby([*keys, "controller_name"], as_index=False).agg(
        mean_cost=("operation_total_cost", "mean"),
        mean_solve_time_s=("avg_solve_time_s", "mean"),
        runs=("case_id", "count"),
    )
    required_set = set(required_controllers)
    coverage = by_controller.groupby("combo", as_index=False).agg(
        controller_set=("controller_name", lambda values: set(values)), minimum_runs=("runs", "min")
    )
    complete_combos = coverage.loc[
        coverage["controller_set"].eq(required_set) & coverage["minimum_runs"].ge(expected_runs_per_controller), "combo"
    ]
    by_controller = by_controller[by_controller["combo"].isin(complete_combos)].copy()
    if by_controller.empty:
        raise RuntimeError(
            f"No mesh has {expected_runs_per_controller} audited successful run(s) "
            "for every required forecaster: " + ", ".join(required_controllers)
        )

    best_cost = by_controller.groupby("controller_name")["mean_cost"].transform("min")
    denominator = best_cost.abs().clip(lower=1e-12)
    by_controller["relative_regret"] = ((by_controller["mean_cost"] - best_cost) / denominator).clip(lower=0.0)

    grouped = by_controller.groupby(keys, as_index=False).agg(
        mean_cost=("mean_cost", "mean"),
        mean_solve_time_s=("mean_solve_time_s", "mean"),
        mean_regret=("relative_regret", "mean"),
        max_regret=("relative_regret", "max"),
        controllers=("controller_name", "nunique"),
        runs=("runs", "sum"),
    )
    for value, prefix in (("mean_cost", "cost"), ("mean_solve_time_s", "solve_time_s"), ("relative_regret", "regret")):
        wide = by_controller.pivot(index=keys, columns="controller_name", values=value)
        wide = wide.rename(columns={controller: f"{prefix}_{controller}" for controller in wide.columns}).reset_index()
        wide.columns.name = None
        grouped = grouped.merge(wide, on=keys, how="left", validate="one_to_one")

    grouped["mean_regret_norm"] = _normalize(grouped["mean_regret"])
    grouped["max_regret_norm"] = _normalize(grouped["max_regret"])
    grouped["time_norm"] = _normalize(grouped["mean_solve_time_s"])
    grouped["selection_score"] = (
        MESH_SELECTION_WEIGHTS["mean_regret"] * grouped["mean_regret_norm"]
        + MESH_SELECTION_WEIGHTS["max_regret"] * grouped["max_regret_norm"]
        + MESH_SELECTION_WEIGHTS["solve_time"] * grouped["time_norm"]
    )
    grouped["pareto"] = True
    objectives = ["mean_regret", "max_regret", "mean_solve_time_s"]
    for idx, row in grouped.iterrows():
        dominated = (
            grouped[objectives].le(row[objectives]).all(axis=1) & grouped[objectives].lt(row[objectives]).any(axis=1)
        ).any()
        grouped.loc[idx, "pareto"] = not bool(dominated)
    return grouped.sort_values(["selection_score", "max_regret", "mean_regret", "mean_solve_time_s"])


def _selection_artifact_is_current(data: dict[str, Any]) -> bool:
    return (
        data.get("operation_model_version") == OPERATION_MODEL_VERSION
        and data.get("mesh_selection_version") == MESH_SELECTION_VERSION
        and tuple(data.get("required_controllers", ())) == SELECTION_CONTROLLERS
    )


def select_meshes(
    summary_path: Path, output_path: Path, top_k: int, expected_runs_per_controller: int = 1
) -> list[dict[str, int]]:
    candidates = _candidate_table(
        pd.read_csv(summary_path),
        required_controllers=SELECTION_CONTROLLERS,
        expected_runs_per_controller=expected_runs_per_controller,
    )
    ordered = candidates[candidates["pareto"]].copy()
    if len(ordered) < top_k:
        ordered = pd.concat([ordered, candidates[~candidates["combo"].isin(ordered["combo"])]])
    selected = ordered.head(int(top_k))
    payload = {
        "operation_model_version": OPERATION_MODEL_VERSION,
        "mesh_selection_version": MESH_SELECTION_VERSION,
        "required_controllers": list(SELECTION_CONTROLLERS),
        "expected_runs_per_controller": int(expected_runs_per_controller),
        "selection_rule": (
            "Pareto front over mean forecaster regret, maximum forecaster regret, "
            "and mean solve time; score = 0.4 normalized mean regret + "
            "0.4 normalized maximum regret + 0.2 normalized solve time"
        ),
        "selection_weights": MESH_SELECTION_WEIGHTS,
        "selected": json.loads(selected.to_json(orient="records")),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    candidates.to_csv(output_path.with_suffix(".csv"), index=False)
    return [{"h": int(row.h), "t1": int(row.t1), "t2": int(row.t2)} for row in selected.itertuples(index=False)]


def write_mesh_effects(summary: pd.DataFrame, path: Path) -> None:
    valid = summary[summary.get("audit_pass", False) == True].copy()  # noqa: E712
    rows = []
    for factor in ("h", "t1", "t2"):
        table = valid.groupby(["controller_name", factor], as_index=False).agg(
            cases=("case_id", "count"),
            mean_cost=("operation_total_cost", "mean"),
            std_cost=("operation_total_cost", "std"),
            mean_solve_time_s=("avg_solve_time_s", "mean"),
        )
        table.insert(1, "factor", factor)
        table.rename(columns={factor: "level"}, inplace=True)
        rows.append(table)
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)


def stage_mesh(args) -> pd.DataFrame:
    root = args.out_root / "01-mesh"
    tasks = []
    for h in MESH_H:
        for t1 in MESH_T1:
            for t2 in MESH_T2:
                mesh = {"h": h, "t1": t1, "t2": t2}
                for controller in EVALUATION_CONTROLLERS:
                    tasks.append(
                        _task(
                            "mesh",
                            controller,
                            mesh,
                            "2009-05-01",
                            args.n_iters,
                            root / controller / _mesh_tag(**mesh),
                            resume=not args.fresh,
                        )
                    )
    summary = run_tasks(tasks, args.workers, root / "summary.csv")
    write_mesh_effects(summary, root / "factor_effects.csv")
    select_meshes(root / "summary.csv", root / "selected_meshes.json", args.top_k)
    return summary


def _selected_meshes(args) -> list[dict[str, int]]:
    path = args.out_root / "01-mesh" / "selected_meshes.json"
    data = None
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = None
    if not data or not _selection_artifact_is_current(data):
        return select_meshes(args.out_root / "01-mesh" / "summary.csv", path, args.top_k)
    return [{"h": int(x["h"]), "t1": int(x["t1"]), "t2": int(x["t2"])} for x in data["selected"]]


def _refresh_baseline_costs(baseline: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    baseline = baseline.copy()
    for index, row in baseline.iterrows():
        if not bool(row.get("audit_pass", False)):
            continue
        mesh = {"h": int(row["h"]), "t1": int(row["t1"]), "t2": int(row["t2"])}
        case_dir = out_root / "01-mesh" / str(row["controller_name"]) / str(row["combo"])
        operation_path = operation_artifact_path(case_dir)
        cost_metrics = _operation_cost_metrics(
            read_operation(operation_path), _base_params(mesh, seed=MONTHLY_OUTAGE_SEEDS[MONTHS[0]])
        )
        for key, value in cost_metrics.items():
            baseline.at[index, key] = value
        metrics_path = case_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics.update(cost_metrics)
            metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return baseline


def stage_seasonal(args) -> pd.DataFrame:
    root = args.out_root / "02-seasonal-mesh"
    baseline_month = MONTHS[0]
    tasks = []
    for h in MESH_H:
        for t1 in MESH_T1:
            for t2 in MESH_T2:
                mesh = {"h": h, "t1": t1, "t2": t2}
                for month, start in MONTHLY_VALIDATION_WINDOWS.items():
                    if month == baseline_month:
                        continue
                    outage_seed = MONTHLY_OUTAGE_SEEDS[month]
                    for controller in EVALUATION_CONTROLLERS:
                        tasks.append(
                            _task(
                                "seasonal",
                                controller,
                                mesh,
                                start,
                                args.n_iters,
                                root / month / controller / _mesh_tag(**mesh),
                                labels={"month": month, "outage_seed": outage_seed},
                                seed=outage_seed,
                                resume=not args.fresh,
                            )
                        )

    monthly = run_tasks(tasks, args.workers, root / "incremental_summary.csv")
    baseline_path = args.out_root / "01-mesh" / "summary.csv"
    if not baseline_path.exists():
        raise FileNotFoundError("The annual mesh campaign reuses May from 01-mesh; run --stage mesh first")
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[baseline["controller_name"].isin(EVALUATION_CONTROLLERS) & baseline["combo"].notna()].copy()
    baseline = _refresh_baseline_costs(baseline, args.out_root)
    baseline["month"] = baseline_month
    baseline["outage_seed"] = MONTHLY_OUTAGE_SEEDS[baseline_month]
    summary = pd.concat([baseline, monthly], ignore_index=True, sort=False)
    summary.to_csv(root / "summary.csv", index=False)
    write_mesh_effects(summary, root / "factor_effects.csv")
    champion = select_meshes(
        root / "summary.csv",
        root / "champion_mesh.json",
        1,
        expected_runs_per_controller=len(MONTHLY_VALIDATION_WINDOWS),
    )[0]
    (args.out_root / "champion_mesh.json").write_text(
        json.dumps(
            {
                "operation_model_version": OPERATION_MODEL_VERSION,
                "mesh_selection_version": MESH_SELECTION_VERSION,
                "required_controllers": list(SELECTION_CONTROLLERS),
                "validation_months": list(MONTHLY_VALIDATION_WINDOWS),
                **champion,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def _champion(args) -> dict[str, int]:
    path = args.out_root / "champion_mesh.json"
    data = None
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = None
    if not data or not _selection_artifact_is_current(data):
        if not (args.out_root / "02-seasonal-mesh" / "summary.csv").exists():
            raise FileNotFoundError(
                "Run --stage mesh and --stage annual-mesh with prototype and LSTM before this stage"
            )
        champion = select_meshes(
            args.out_root / "02-seasonal-mesh" / "summary.csv",
            args.out_root / "02-seasonal-mesh" / "champion_mesh.json",
            1,
            expected_runs_per_controller=len(MONTHLY_VALIDATION_WINDOWS),
        )[0]
        data = {
            "operation_model_version": OPERATION_MODEL_VERSION,
            "mesh_selection_version": MESH_SELECTION_VERSION,
            "required_controllers": list(SELECTION_CONTROLLERS),
            "validation_months": list(MONTHLY_VALIDATION_WINDOWS),
            **champion,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"h": int(data["h"]), "t1": int(data["t1"]), "t2": int(data["t2"])}


def stage_forecast(args) -> pd.DataFrame:
    root = args.out_root / "03-forecast-operation"
    mesh = _champion(args)
    controllers = tuple(x.strip() for x in args.forecast_controllers.split(",") if x.strip())
    tasks = []
    for month in MONTHS:
        outage_seed = MONTHLY_OUTAGE_SEEDS[month]
        for controller in controllers:
            tasks.append(
                _task(
                    "forecast",
                    controller,
                    mesh,
                    f"{month}-01",
                    args.n_iters,
                    root / month / controller,
                    labels={"month": month, "outage_seed": outage_seed},
                    seed=outage_seed,
                    resume=not args.fresh,
                )
            )
    return run_tasks(tasks, args.workers, root / "summary.csv")


def stage_recourse(args) -> pd.DataFrame:
    root = args.out_root / "04-recourse"
    selected = _champion(args)
    mesh = {"h": 36, "t1": selected["t1"], "t2": selected["t2"]}
    tasks = []
    for window, start, seed in RECOURSE_WINDOWS:
        for frequency in (24, 6, 1):
            label = f"{window}_r{frequency}h"
            tasks.append(
                _task(
                    "recourse",
                    "stochastic",
                    mesh,
                    start,
                    args.n_iters,
                    root / label,
                    labels={"window": window, "seed": seed, "resolve_every_h": frequency},
                    seed=seed,
                    resolve_every_h=frequency,
                    close_evaluation_energy=True,
                    terminal_closure_lookahead_h=48.0,
                    resume=not args.fresh,
                )
            )
    return run_tasks(tasks, max(1, min(args.workers, 2)), root / "summary.csv")


def stage_robustness(args) -> pd.DataFrame:
    root = args.out_root / "05-robustness"
    mesh = _champion(args)
    tasks = []
    for window, start, window_seed in ROBUSTNESS_WINDOWS:
        for variant, fixed_seed, overrides in ROBUSTNESS_VARIANTS:
            seed = window_seed if fixed_seed is None else fixed_seed
            for controller in ("prototype", "stochastic"):
                tasks.append(
                    _task(
                        "robustness",
                        controller,
                        mesh,
                        start,
                        args.n_iters,
                        root / window / variant / controller,
                        labels={"window": window, "variant": variant, "seed": seed},
                        seed=seed,
                        overrides=overrides,
                        resume=not args.fresh,
                    )
                )
    return run_tasks(tasks, max(1, min(args.workers, 2)), root / "summary.csv")


def _write_causal_pilot_comparison(summary: pd.DataFrame, path: Path) -> None:
    keys = ["month", "combo"]
    metrics = ["operation_total_cost", "avg_solve_time_s", "total_time_s"]
    available = [column for column in metrics if column in summary]
    if not available:
        return

    frames = []
    for controller in SELECTION_CONTROLLERS:
        frame = summary[summary["controller_name"] == controller]
        frame = frame[[*keys, *available]].drop_duplicates(keys, keep="last")
        frame = frame.rename(columns={column: f"{column}_{controller}" for column in available})
        frames.append(frame)
    comparison = frames[0]
    for frame in frames[1:]:
        comparison = comparison.merge(frame, on=keys, how="outer", validate="one_to_one")

    if "operation_total_cost" in available:
        lstm_cost = comparison["operation_total_cost_lstm"]
        prototype_cost = comparison["operation_total_cost_prototype"]
        comparison["delta_cost_lstm_minus_prototype"] = lstm_cost - prototype_cost
        comparison["delta_cost_pct_lstm_vs_prototype"] = (
            100.0 * comparison["delta_cost_lstm_minus_prototype"] / prototype_cost.abs().clip(lower=1e-12)
        )
    comparison.to_csv(path, index=False)


def stage_causal_pilot(args) -> pd.DataFrame:
    root = args.out_root / "00-causal-pilot"
    tasks = []
    for controller in SELECTION_CONTROLLERS:
        for mesh in CAUSAL_PILOT_MESHES:
            for month, start in MONTHLY_VALIDATION_WINDOWS.items():
                outage_seed = MONTHLY_OUTAGE_SEEDS[month]
                tasks.append(
                    _task(
                        "causal-pilot",
                        controller,
                        mesh,
                        start,
                        args.n_iters,
                        root / month / controller / _mesh_tag(**mesh),
                        labels={
                            "month": month,
                            "outage_seed": outage_seed,
                            "forecast_assembly_version": FORECAST_ASSEMBLY_VERSION,
                        },
                        seed=outage_seed,
                        resume=not args.fresh,
                    )
                )
    summary = run_tasks(tasks, args.workers, root / "summary.csv")
    _write_causal_pilot_comparison(summary, root / "comparison_lstm_prototype.csv")
    return summary


def stage_smoke(args) -> pd.DataFrame:
    root = args.out_root / "00-smoke"
    mesh = {"h": 12, "t1": 15, "t2": 120}
    tasks = [
        _task("smoke", controller, mesh, "2009-05-01", args.smoke_iters, root / controller, resume=not args.fresh)
        for controller in EVALUATION_CONTROLLERS
    ]
    return run_tasks(tasks, 1, root / "summary.csv")


def write_manifest(args) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.sizing_artifact:
        params = load_sized_parameters_from_artifact(args.parameters, args.sizing_artifact)
    else:
        params = load_sized_parameters(str(args.parameters))
    sizing_source = Path(params["sizing_case_applied"]["source"])
    sizing_sha256 = sha256_file(sizing_source)
    parameters_sha256 = sha256_file(args.parameters)
    manifest_path = args.out_root / "campaign_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_id = previous.get("campaign_id")
        previous_inputs = previous.get("portable_inputs", {}) or {}
        previous_sizing = previous_inputs.get("sizing_artifact_sha256")
        previous_parameters = previous_inputs.get("parameters_sha256")
        if previous_id and previous_id != args.campaign_id:
            raise RuntimeError(f"Output root belongs to campaign {previous_id!r}, not {args.campaign_id!r}")
        if previous_sizing and previous_sizing != sizing_sha256:
            raise RuntimeError("Output root already contains results from another sizing artifact")
        if previous_parameters and previous_parameters != parameters_sha256:
            raise RuntimeError("Output root already contains results from another parameter file")

    inputs = args.out_root / "campaign-inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    parameters_copy = inputs / "parameters.json"
    parameters_copy.write_bytes(args.parameters.read_bytes())
    sizing_copy = inputs / "sizing_artifact.json"
    sizing_copy.write_bytes(sizing_source.read_bytes())
    (inputs / "operational_parameters.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "schema_version": 2,
        "campaign_id": args.campaign_id,
        "operation_model_version": OPERATION_MODEL_VERSION,
        "cost_accounting_version": COST_ACCOUNTING_VERSION,
        "forecast_assembly_version": FORECAST_ASSEMBLY_VERSION,
        "base_parameter_fingerprint_sha256": parameter_fingerprint(params),
        "created_at": pd.Timestamp.now().isoformat(),
        "legacy_results_preserved": True,
        "output_root": str(args.out_root),
        "portable_inputs": {
            "parameters": str(parameters_copy),
            "parameters_sha256": parameters_sha256,
            "sizing_artifact": str(sizing_copy),
            "sizing_artifact_sha256": sizing_sha256,
            "sizing_case": params["sizing_case_applied"]["case"],
            "pv_size_kw": params["sizing_case_applied"]["P_hat_PV_kw"],
            "bess_size_kwh": params["sizing_case_applied"]["E_hat_BESS_kwh"],
            "bess_power_kw": params["sizing_case_applied"]["BESS_Pmax_kw"],
        },
        "corrections": [
            "fixed physical outage support independent of dt1/dt2",
            "first-arrival hazard weights calibrated to the 24 h outage probability",
            "complete pre-outage non-anticipativity for BESS, grid, shedding, and curtailment",
            "5 min physical ramp reference independent of dt1",
            "terminal energy after the last interval not below measured initial energy",
            "exact common terminal energy at the finite recourse evaluation boundary",
            "BESS wear charged on bidirectional throughput instead of absolute net power",
            "fixed export remuneration included in MPC, stochastic operation, and realized costs",
            "continuous realized load shedding consistent with the LP operational models",
            "last-resort discharge reduction when export and PV curtailment cannot absorb surplus",
            "strict realized power-balance, outage-isolation, and BESS-state audits",
            "cache invalidation by model version, parameter fingerprint, and complete run signature",
            "hourly LSTM inputs restricted to fully completed observation hours",
            "LSTM included in the full temporal-mesh and monthly validation campaigns",
            "ideal retained as a target but excluded from temporal-mesh selection",
            "mesh selected by implementable-controller relative regret and solve time",
            "all temporal meshes validated for 10 days in every month of the test year",
            "month-specific outage seeds shared by all configurations within each month",
        ],
        "mesh_design": {
            "evaluation_controllers": list(EVALUATION_CONTROLLERS),
            "selection_controllers": list(SELECTION_CONTROLLERS),
            "selection_version": MESH_SELECTION_VERSION,
            "selection_weights": MESH_SELECTION_WEIGHTS,
        },
        "planned_runs": {
            "causal_lstm_prototype_pilot": (
                len(CAUSAL_PILOT_MESHES) * len(MONTHLY_VALIDATION_WINDOWS) * len(SELECTION_CONTROLLERS)
            ),
            "mesh": (len(MESH_H) * len(MESH_T1) * len(MESH_T2) * len(EVALUATION_CONTROLLERS)),
            "annual_mesh_configurations": (len(MESH_H) * len(MESH_T1) * len(MESH_T2) * len(EVALUATION_CONTROLLERS)),
            "annual_mesh_monthly_runs": (
                len(MESH_H)
                * len(MESH_T1)
                * len(MESH_T2)
                * len(EVALUATION_CONTROLLERS)
                * len(MONTHLY_VALIDATION_WINDOWS)
            ),
            "annual_mesh_incremental_runs_after_may": (
                len(MESH_H)
                * len(MESH_T1)
                * len(MESH_T2)
                * len(EVALUATION_CONTROLLERS)
                * (len(MONTHLY_VALIDATION_WINDOWS) - 1)
            ),
            "forecast": len(MONTHS) * len([x for x in args.forecast_controllers.split(",") if x]),
            "recourse": len(RECOURSE_WINDOWS) * 3,
            "robustness": (len(ROBUSTNESS_WINDOWS) * len(ROBUSTNESS_VARIANTS) * 2),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_campaign_definition(args) -> None:
    definition = ROOT / "operation-campaigns" / args.campaign_id / "campaign.json"
    if not definition.is_file():
        return
    payload = json.loads(definition.read_text(encoding="utf-8"))
    if payload.get("campaign_id") != args.campaign_id:
        raise RuntimeError(f"Invalid campaign definition: {definition}")
    if not args.sizing_artifact:
        raise RuntimeError(f"Campaign {args.campaign_id!r} requires --sizing-artifact")
    expected_sizing = str(payload.get("sizing_artifact_sha256", ""))
    expected_parameters = str(payload.get("parameters_sha256", ""))
    if sha256_file(args.sizing_artifact) != expected_sizing:
        raise RuntimeError(f"Sizing artifact does not match campaign {args.campaign_id!r}: {args.sizing_artifact}")
    if sha256_file(args.parameters) != expected_parameters:
        raise RuntimeError(f"Parameter file does not match campaign {args.campaign_id!r}: {args.parameters}")


def resolve_output_root(stage: str, out_root: Path) -> Path:
    resolved = out_root.resolve()
    if stage == "smoke" and resolved == DEFAULT_OUT_ROOT.resolve():
        return DEFAULT_SMOKE_ROOT.resolve()
    return resolved


def _run_pipeline(args: SimpleNamespace) -> None:
    if args.workers < 1 or args.n_iters < 1 or args.smoke_iters < 1 or args.top_k < 1:
        raise ValueError("workers, n-iters, smoke-iters, and top-k must be positive")
    args.parameters = args.parameters.resolve()
    args.out_root = resolve_output_root(args.stage, args.out_root)
    if not args.sizing_artifact:
        default_artifact = ROOT / "operation-campaigns" / args.campaign_id / "sizing_artifact.json"
        if default_artifact.is_file():
            args.sizing_artifact = default_artifact
    if args.sizing_artifact:
        args.sizing_artifact = args.sizing_artifact.resolve()
    validate_campaign_definition(args)
    os.environ[PARAMETERS_ENV] = str(args.parameters)
    os.environ[CAMPAIGN_ID_ENV] = str(args.campaign_id)
    if args.sizing_artifact:
        os.environ[SIZING_ARTIFACT_ENV] = str(args.sizing_artifact)
    else:
        os.environ.pop(SIZING_ARTIFACT_ENV, None)
    write_manifest(args)
    stages = {
        "smoke": stage_smoke,
        "causal-pilot": stage_causal_pilot,
        "mesh": stage_mesh,
        "annual-mesh": stage_seasonal,
        "seasonal": stage_seasonal,
        "forecast": stage_forecast,
        "recourse": stage_recourse,
        "robustness": stage_robustness,
    }
    order = (
        ("causal-pilot", "mesh", "annual-mesh", "forecast", "recourse", "robustness")
        if args.stage == "all"
        else (args.stage,)
    )
    for name in order:
        print(f"\n===== corrected pipeline: {name} =====", flush=True)
        stages[name](args)


def main() -> None:
    parameters = SIZING_ROOT / "parameters.json"
    if not parameters.exists():
        raise FileNotFoundError(f"Missing sizing parameters: {parameters}")

    for sizing_case in SIZING_CASES:
        artifact = SIZING_ROOT / sizing_case / "sizing_decision_variables.json"
        if not artifact.exists():
            raise FileNotFoundError(f"Missing sizing artifact: {artifact}")
        print(f"\n=== with-degradation / {sizing_case} ===", flush=True)
        args = SimpleNamespace(
            stage=STAGE,
            out_root=OPERATION_ROOT / sizing_case,
            parameters=parameters,
            sizing_artifact=artifact,
            campaign_id=f"with-degradation-{sizing_case}",
            workers=WORKERS,
            n_iters=N_ITERS,
            smoke_iters=SMOKE_ITERS,
            top_k=TOP_K,
            forecast_controllers=FORECAST_CONTROLLERS,
            fresh=FRESH,
        )
        _run_pipeline(args)


if __name__ == "__main__":
    main()
