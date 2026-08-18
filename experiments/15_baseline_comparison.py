"""Run causal rule-based baselines against the selected MPC reference cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opt import (
    LoadShiftingController,
    PeakShavingController,
    ReserveOnlyController,
    SelfConsumptionController,
    simulate_baseline,
)
from opt.artifacts import operation_artifact_path, read_operation
from opt.baseline import BASELINE_MODEL_VERSION
from opt.campaigns import (
    CampaignProvenance,
    campaign_layout,
    load_campaign_provenance,
    sha256_file,
    validate_parameter_provenance,
    validate_summary_provenance,
)
from opt.operation import OPERATION_MODEL_VERSION
from opt.utils import load_series_scaled


DEFAULT_CAMPAIGN_ROOT = ROOT / "outputs" / "sweeps" / "economic"
DEFAULT_SMOKE_ROOT = ROOT / "outputs" / "_smoke-baselines"
DEFAULT_LOAD_TRAIN = ROOT / "data" / "load_5min_train.csv"
DEFAULT_PV_TRAIN = ROOT / "data" / "pv_5min_train.csv"
BASELINE_NAMES = ("reserve_only", "self_consumption", "load_shifting", "peak_shaving")
BASELINE_CAMPAIGN_VERSION = "selected-mesh-paired-baselines-v1"
AUDIT_TOLERANCE = 1e-8


def project_path(path: Path | str) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def path_label(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def exogenous_hash(operation: pd.DataFrame) -> str:
    columns = ["timestamp", "Load_kw", "PV_kw", "outage_active"]
    missing = sorted(set(columns).difference(operation.columns))
    if missing:
        raise ValueError("Missing exogenous columns: " + ", ".join(missing))
    values = operation[columns].copy()
    values["timestamp"] = values["timestamp"].astype(str)
    for column in ("Load_kw", "PV_kw"):
        values[column] = pd.to_numeric(values[column], errors="raise").map(
            lambda value: format(float(value), ".15g")
        )
    values["outage_active"] = boolean(values["outage_active"]).astype(int)
    payload = values.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def controller_for(name: str, params: dict, peak_limit_kw: float, reserve_fraction: float):
    if name == "reserve_only":
        return ReserveOnlyController()
    if name == "self_consumption":
        return SelfConsumptionController(reserve_fraction=reserve_fraction)
    if name == "peak_shaving":
        return PeakShavingController(peak_limit_kw, reserve_fraction=reserve_fraction)
    if name == "load_shifting":
        tariffs = [float(value) for value in params["costs"]["EDS"].values()]
        return LoadShiftingController(min(tariffs), max(tariffs), reserve_fraction)
    raise ValueError(f"Unknown baseline: {name}")


def calibrate_peak_limit(
    params: dict, quantile: float, load_csv: Path, pv_csv: Path,
) -> tuple[float, dict[str, Any]]:
    if not 0.0 < quantile < 1.0:
        raise ValueError("peak_quantile must be between zero and one")
    scaling = {
        "P_L_nom_kw": float(params["Load"]["Pmax_kw"]),
        "P_PV_nom_kw": float(params["PV"]["Pmax_kw"]),
    }
    load, pv = load_series_scaled(scaling, str(load_csv), str(pv_csv))
    aligned = pd.concat({"load": load, "pv": pv}, axis=1, join="inner").dropna()
    net_import = (aligned["load"] - aligned["pv"]).clip(lower=0.0)
    limit = float(net_import.quantile(quantile))
    metadata = {
        "method": "training net-load quantile",
        "quantile": float(quantile),
        "import_limit_kw": limit,
        "samples": int(len(net_import)),
        "load_train": path_label(load_csv),
        "pv_train": path_label(pv_csv),
        "load_train_sha256": sha256_file(load_csv),
        "pv_train_sha256": sha256_file(pv_csv),
    }
    return limit, metadata


def reference_windows(
    path: Path,
    provenance: CampaignProvenance,
    months: set[str] | None = None,
) -> list[dict[str, Any]]:
    reference = pd.read_csv(path)
    validate_summary_provenance(reference, provenance, "selected mesh reference")
    rows = reference[reference["controller_name"] == "prototype"].copy()
    if months is not None:
        rows = rows[rows["month"].astype(str).isin(months)]
    if rows.empty or rows["month"].duplicated().any():
        raise RuntimeError("The canonical reference must contain one prototype case per month")
    if "case_audit_pass" in rows and not boolean(rows["case_audit_pass"]).all():
        raise RuntimeError("At least one selected reference trajectory failed its audit")

    windows = []
    for _, row in rows.sort_values("month").iterrows():
        case_dir = ROOT / str(row["case_dir"])
        params_path = case_dir / "parameters_used.json"
        metrics_path = case_dir / "metrics.json"
        operation_path = operation_artifact_path(case_dir, required=False)
        for artifact in (params_path, metrics_path, operation_path):
            if not artifact.exists():
                raise FileNotFoundError(f"Missing canonical reference artifact: {artifact}")

        saved = json.loads(params_path.read_text(encoding="utf-8"))
        run = saved.get("_operation_run", {})
        params = {key: value for key, value in saved.items() if not key.startswith("_operation_")}
        validate_parameter_provenance(params, provenance, str(case_dir))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        windows.append({
            "campaign_id": provenance.campaign_id,
            "sizing_source_sha256": provenance.sizing_source_sha256,
            "month": str(row["month"]),
            "params": params,
            "start_ts": str(run.get("start_ts", metrics["run_start_ts"])),
            "n_iters": int(run.get("n_iters", metrics["n_iters"])),
            "load_csv": str((ROOT / run.get("load_csv", "data/load_5min_test.csv")).resolve()),
            "pv_csv": str((ROOT / run.get("pv_csv", "data/pv_5min_test.csv")).resolve()),
            "reference_operation": str(operation_path),
            "reference_exogenous_sha256": str(row["exogenous_sha256"]),
            "outage_seed": int(params["EDS"]["seed"]),
            "h": int(row["h"]),
            "t1": int(row["t1"]),
            "t2": int(row["t2"]),
            "combo": str(row["combo"]),
        })
    return windows


def trajectory_metrics(operation: pd.DataFrame, params: dict) -> dict[str, Any]:
    dt_h = float(params["time"]["timestep"]) / 60.0
    outage = boolean(operation["outage_active"])
    outage_start = outage & ~outage.shift(1, fill_value=False)

    def values(column: str) -> pd.Series:
        return pd.to_numeric(operation[column], errors="coerce").fillna(0.0)

    load = values("Load_kw")
    served = values("Load_served_kw")
    shed = values("Shedding_kw")
    pv = values("PV_kw")
    pv_used = values("PV_used_kw")
    curtailed = values("Curtailment_kw")
    charge = values("P_bess_charge_mag_kw")
    discharge = values("P_bess_discharge_kw")
    soc = values("SoC_pct")
    return {
        "evaluated_hours": float(len(operation) * dt_h),
        "grid_import_kwh": float(values("P_grid_in_kw").sum() * dt_h),
        "grid_export_kwh": float(values("P_grid_out_kw").sum() * dt_h),
        "peak_grid_import_kw": float(values("P_grid_in_kw").max()),
        "load_demand_kwh": float(load.sum() * dt_h),
        "load_served_kwh": float(served.sum() * dt_h),
        "load_shed_kwh": float(shed.sum() * dt_h),
        "outage_load_shed_kwh": float(shed[outage].sum() * dt_h),
        "load_shed_hours": float((shed > 1e-9).sum() * dt_h),
        "outage_load_shed_hours": float(((shed > 1e-9) & outage).sum() * dt_h),
        "max_load_shed_kw": float(shed.max()),
        "pv_available_kwh": float(pv.sum() * dt_h),
        "pv_used_kwh": float(pv_used.sum() * dt_h),
        "pv_curtailed_kwh": float(curtailed.sum() * dt_h),
        "outage_pv_curtailed_kwh": float(curtailed[outage].sum() * dt_h),
        "bess_charge_kwh": float(charge.sum() * dt_h),
        "bess_discharge_kwh": float(discharge.sum() * dt_h),
        "min_soc_pct": float(soc.min()),
        "max_soc_pct": float(soc.max()),
        "final_soc_pct": float(soc.iloc[-1]),
        "outage_events": int(outage_start.sum()),
        "outage_hours": float(outage.sum() * dt_h),
        "mean_soc_at_outage_start_pct": (
            float(soc[outage_start].mean()) if outage_start.any() else float("nan")
        ),
        "cost_grid": float(values("cost_grid").sum()),
        "cost_shed": float(values("cost_shed").sum()),
        "cost_curt": float(values("cost_curt").sum()),
        "noise_steps": int(boolean(operation["noise_applied"]).sum()),
    }


def execute_task(task: dict[str, Any]) -> dict[str, Any]:
    params = task["params"]
    controller = controller_for(
        task["controller"], params, task["peak_limit_kw"], task["reserve_fraction"]
    )
    metrics = simulate_baseline(
        params=params,
        controller=controller,
        start_ts=task["start_ts"],
        n_iters=task["n_iters"],
        out_dir=task["out_dir"],
        load_csv=task["load_csv"],
        pv_csv=task["pv_csv"],
        resume=task["resume"],
    )
    operation = read_operation(task["out_dir"])
    reference = read_operation(task["reference_operation"], nrows=task["n_iters"])
    observed_hash = exogenous_hash(operation)
    expected_hash = exogenous_hash(reference)
    metrics.update(trajectory_metrics(operation, params))
    metrics.update(
        case_id=f"{task['month']}/{controller.name}",
        stage="baselines",
        controller_name=controller.name,
        controller_family="rule_based",
        month=task["month"],
        start=task["start_ts"],
        outage_seed=task["outage_seed"],
        h=task["h"],
        t1=task["t1"],
        t2=task["t2"],
        combo=task["combo"],
        campaign_id=task["campaign_id"],
        sizing_source_sha256=task["sizing_source_sha256"],
        exogenous_sha256=observed_hash,
        expected_exogenous_sha256=expected_hash,
        pairing_pass=observed_hash == expected_hash,
    )
    physical = (
        float(metrics["max_abs_power_balance_residual_kw"]) <= AUDIT_TOLERANCE
        and float(metrics["max_grid_exchange_during_outage_kw"]) <= AUDIT_TOLERANCE
        and float(metrics["max_bess_energy_bound_violation_kwh"]) <= AUDIT_TOLERANCE
        and float(metrics["max_bess_energy_transition_residual_kwh"]) <= AUDIT_TOLERANCE
        and float(metrics["max_plant_simultaneous_ch_dis_kw"]) <= AUDIT_TOLERANCE
    )
    metrics["audit_pass"] = bool(physical and metrics["pairing_pass"])
    return metrics


def write_progress(path: Path, completed: int, total: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    eta = elapsed * (total - completed) / completed if completed else None
    payload = {
        "stage": "baselines",
        "completed": completed,
        "total": total,
        "elapsed_s": elapsed,
        "eta_s": eta,
        "updated_at": pd.Timestamp.now().isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_tasks(tasks: list[dict[str, Any]], workers: int, output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.csv"
    state_path = output / "pipeline_state.json"
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    def record(task: dict[str, Any], result=None, error: Exception | None = None) -> None:
        if error is None:
            row = result
            status = "ok"
        else:
            row = {
                "case_id": f"{task['month']}/{task['controller']}",
                "campaign_id": task["campaign_id"],
                "sizing_source_sha256": task["sizing_source_sha256"],
                "month": task["month"],
                "controller_name": task["controller"],
                "status": "error",
                "audit_pass": False,
                "pairing_pass": False,
                "error": str(error),
            }
            status = f"FAILED: {error}"
        rows.append(row)
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        write_progress(state_path, len(rows), len(tasks), started)
        print(f"[{len(rows)}/{len(tasks)}] {row['case_id']}: {status}", flush=True)

    if workers == 1:
        for task in tasks:
            try:
                record(task, result=execute_task(task))
            except Exception as error:
                record(task, error=error)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(execute_task, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    record(task, result=future.result())
                except Exception as error:
                    record(task, error=error)
    return pd.DataFrame(rows)


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    if "status" not in summary or (summary["status"] == "error").any():
        raise RuntimeError("Baseline campaign contains failed cases")
    sum_fields = (
        "evaluated_hours", "operation_grid_reliability_cost", "operation_wear_cost",
        "operation_total_cost", "grid_import_kwh", "grid_export_kwh",
        "load_demand_kwh", "load_served_kwh", "load_shed_kwh",
        "outage_load_shed_kwh", "load_shed_hours", "outage_load_shed_hours",
        "pv_available_kwh", "pv_used_kwh", "pv_curtailed_kwh",
        "outage_pv_curtailed_kwh", "bess_charge_kwh", "bess_discharge_kwh",
        "operation_bess_throughput_kwh", "outage_events", "outage_hours",
        "cost_grid", "cost_shed", "cost_curt", "noise_steps",
    )
    rows = []
    for controller, group in summary.groupby("controller_name", sort=True):
        row = {
            "campaign_id": str(group["campaign_id"].iloc[0]),
            "sizing_source_sha256": str(group["sizing_source_sha256"].iloc[0]),
            "controller_name": controller,
            "controller_family": "rule_based",
            "windows": int(group["month"].nunique()),
            "trajectory_count": int(len(group)),
            "mean_solve_time_s": float("nan"),
            "mean_action_time_s": float(group["avg_action_time_s"].mean()),
            "total_runtime_h": float(group["total_time_s"].sum() / 3600.0),
            "peak_grid_import_kw": float(group["peak_grid_import_kw"].max()),
            "max_load_shed_kw": float(group["max_load_shed_kw"].max()),
            "min_soc_pct": float(group["min_soc_pct"].min()),
            "max_soc_pct": float(group["max_soc_pct"].max()),
            "mean_soc_at_outage_start_pct": float(group["mean_soc_at_outage_start_pct"].mean()),
            "all_cases_audited": bool(group["audit_pass"].all()),
            "all_cases_paired": bool(group["pairing_pass"].all()),
        }
        row.update({field: float(group[field].sum()) for field in sum_fields})
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(
    args, monthly: pd.DataFrame, baseline_summary: pd.DataFrame,
    calibration: dict[str, Any], controllers: list[str], full_campaign: bool,
) -> None:
    baseline_summary.to_csv(args.out_root / "baseline_summary.csv", index=False)
    comparison_path = None
    if full_campaign:
        reference = pd.read_csv(args.reference_summary)
        validate_summary_provenance(reference, args.provenance, "controller summary")
        reference["controller_family"] = reference["controller_name"].map(
            {"ideal": "perfect_information", "lstm": "mpc", "prototype": "mpc"}
        )
        reference["mean_action_time_s"] = float("nan")
        reference["all_cases_paired"] = True
        comparison = pd.concat([reference, baseline_summary], ignore_index=True, sort=False)
        comparison_path = args.out_root / "controller_comparison_120d.csv"
        comparison.to_csv(comparison_path, index=False)

    manifest = {
        "campaign_id": args.provenance.campaign_id,
        "sizing_source_sha256": args.provenance.sizing_source_sha256,
        "campaign_version": BASELINE_CAMPAIGN_VERSION,
        "baseline_model_version": BASELINE_MODEL_VERSION,
        "operation_model_version": OPERATION_MODEL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_monthly": path_label(args.reference_monthly),
        "reference_monthly_sha256": sha256_file(args.reference_monthly),
        "controllers": controllers,
        "reserve_fraction": args.reserve_fraction,
        "peak_shaving_calibration": calibration,
        "months": sorted(monthly["month"].dropna().astype(str).unique()),
        "cases": int(len(monthly)),
        "audited_cases": int(boolean(monthly["audit_pass"]).sum()),
        "paired_cases": int(boolean(monthly["pairing_pass"]).sum()),
        "full_120_day_campaign": bool(full_campaign),
        "comparison_output": comparison_path.name if comparison_path else None,
    }
    (args.out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    text = f"""# Rule-based operational baselines

The cases in this directory use the same sized system, physical `GridEnv`,
load/PV trajectories, initial energy, and outage realization as the selected
MPC reference. Each policy is causal and reads only the measurements available
at the current 5-min step. The peak-shaving threshold ({calibration['import_limit_kw']:.6f} kW)
was fixed from the {calibration['quantile']:.0%} quantile of training net load.

Controllers: {', '.join(controllers)}.

Reproduce the full paired campaign with:

```powershell
.\\.venv\\Scripts\\python.exe experiments\\15_baseline_comparison.py --campaign-root {path_label(args.campaign_root)} --campaign-id {args.provenance.campaign_id} --workers 4
```

Use `--smoke` for a short integration run. `reserve_only` keeps the BESS idle
on-grid; the common emergency controller can still use it during an outage.
"""
    (args.out_root / "README.md").write_text(text, encoding="utf-8")


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--campaign-id")
    parser.add_argument("--controllers", default=",".join(BASELINE_NAMES))
    parser.add_argument("--months", default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-iters", type=int)
    parser.add_argument("--reserve-fraction", type=float, default=0.0)
    parser.add_argument("--peak-quantile", type=float, default=0.75)
    parser.add_argument("--reference-monthly", type=Path)
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--load-train", type=Path, default=DEFAULT_LOAD_TRAIN)
    parser.add_argument("--pv-train", type=Path, default=DEFAULT_PV_TRAIN)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    args.campaign_root = project_path(args.campaign_root)
    layout = campaign_layout(args.campaign_root)
    out_root_explicit = args.out_root is not None
    args.reference_monthly = args.reference_monthly or (
        layout["reference"] / "selected_mesh_monthly.csv"
    )
    args.reference_summary = args.reference_summary or (
        layout["reference"] / "selected_mesh_120d_summary.csv"
    )
    args.out_root = args.out_root or layout["baselines"]
    for name in ("reference_monthly", "reference_summary", "load_train", "pv_train", "out_root"):
        setattr(args, name, project_path(getattr(args, name)))
    args.provenance = load_campaign_provenance(args.campaign_root, args.campaign_id)
    args.campaign_id = args.provenance.campaign_id
    controllers = parse_csv_list(args.controllers)
    unknown = sorted(set(controllers).difference(BASELINE_NAMES))
    if unknown:
        raise ValueError("Unknown baselines: " + ", ".join(unknown))
    if len(controllers) != len(set(controllers)):
        raise ValueError("controllers must not contain duplicates")
    if args.workers < 1 or (args.n_iters is not None and args.n_iters < 1):
        raise ValueError("workers and n_iters must be positive")
    if not 0.0 <= args.reserve_fraction <= 1.0:
        raise ValueError("reserve_fraction must be between zero and one")

    selected_months = None if args.months == "all" else set(parse_csv_list(args.months))
    windows = reference_windows(args.reference_monthly, args.provenance, selected_months)
    if args.smoke:
        windows = windows[:1]
        args.n_iters = args.n_iters or 12
        if not out_root_explicit:
            args.out_root = (DEFAULT_SMOKE_ROOT / args.campaign_id).resolve()
    if not windows:
        raise RuntimeError("No reference windows matched the requested months")

    peak_limit, calibration = calibrate_peak_limit(
        windows[0]["params"], args.peak_quantile, args.load_train, args.pv_train
    )
    tasks = []
    for window in windows:
        n_iters = min(window["n_iters"], args.n_iters) if args.n_iters else window["n_iters"]
        for controller in controllers:
            tasks.append({
                **window,
                "controller": controller,
                "n_iters": int(n_iters),
                "peak_limit_kw": peak_limit,
                "reserve_fraction": args.reserve_fraction,
                "out_dir": str(args.out_root / window["month"] / controller),
                "resume": not args.fresh,
            })

    args.out_root.mkdir(parents=True, exist_ok=True)
    monthly = run_tasks(tasks, args.workers, args.out_root)
    monthly = monthly.sort_values(["month", "controller_name"]).reset_index(drop=True)
    monthly.to_csv(args.out_root / "summary.csv", index=False)
    baseline_summary = aggregate(monthly)
    full_campaign = (
        set(monthly["month"].astype(str)) == set(pd.read_csv(args.reference_monthly)["month"].astype(str))
        and args.n_iters is None
        and set(controllers) == set(BASELINE_NAMES)
    )
    write_outputs(args, monthly, baseline_summary, calibration, controllers, full_campaign)
    print(f"Baseline artifacts written to {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
