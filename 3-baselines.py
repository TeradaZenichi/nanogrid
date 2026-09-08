"""Run independent rule-based baselines and pair them with available MPC results.

Baseline artifacts live under ``outputs/baseline-sweep`` and therefore never
modify the operational campaign. Baseline generation depends only on a sizing
artifact and the common input series. Comparisons are built incrementally from
the valid, audited monthly artifacts currently available from ``2-operation.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from opt import (
    LoadShiftingController,
    PeakShavingController,
    ReserveOnlyController,
    SelfConsumptionController,
    load_sized_parameters_from_artifact,
    simulate_baseline,
)
from opt.artifacts import read_operation
from opt.baseline import BASELINE_MODEL_VERSION
from opt.campaigns import sha256_file
from opt.operation import CACHE_PHYSICAL_LIMITS, OPERATION_MODEL_VERSION
from opt.utils import load_series_scaled

PARAMETERS_JSON = (
    ROOT / "outputs" / "sizing-sweep" / "with-degradation" / "parameters.json"
)
SIZING_ROOT = ROOT / "outputs" / "sizing-sweep" / "with-degradation"
OPERATION_ROOT = ROOT / "outputs" / "operation-sweep" / "with-degradation"
BASELINE_ROOT = ROOT / "outputs" / "baseline-sweep" / "with-degradation"
LOAD_CSV = ROOT / "data" / "load_5min_test.csv"
PV_CSV = ROOT / "data" / "pv_5min_test.csv"
LOAD_TRAIN_CSV = ROOT / "data" / "load_5min_train.csv"
PV_TRAIN_CSV = ROOT / "data" / "pv_5min_train.csv"

SIZING_CASES = ("economic", "critical_50", "full_100")
CONTROLLERS = ("reserve_only", "self_consumption", "load_shifting", "peak_shaving")
REFERENCE_CONTROLLERS = ("ideal", "prototype", "lstm")
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
BASE_OUTAGE_SEED = 42
MONTHLY_OUTAGE_SEEDS = {
    month: BASE_OUTAGE_SEED + index for index, month in enumerate(MONTHS)
}

N_ITERS = 2880
WORKERS = 1  # Rule-based cases are cheap; sequential execution is robust and resumable.
RESERVE_FRACTION = 0.0
PEAK_QUANTILE = 0.75
FRESH = False
CAMPAIGN_VERSION = "independent-paired-rule-based-baselines-v4"


def _boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    _replace_with_retry(temporary, path)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
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


def _mesh_from_params(params: dict) -> dict[str, int]:
    return {
        "h": int(params["time"]["horizon_hours"]),
        "t1": int(params["time"]["timestep_1_min"]),
        "t2": int(params["time"]["timestep_2_min"]),
    }


def _mesh_tag(mesh: dict[str, int]) -> str:
    return f"h{mesh['h']}_t1_{mesh['t1']}_t2_{mesh['t2']}"


def _controller(name: str, params: dict, peak_limit_kw: float):
    if name == "reserve_only":
        return ReserveOnlyController()
    if name == "self_consumption":
        return SelfConsumptionController(reserve_fraction=RESERVE_FRACTION)
    if name == "load_shifting":
        tariffs = [float(value) for value in params["costs"]["EDS"].values()]
        return LoadShiftingController(min(tariffs), max(tariffs), RESERVE_FRACTION)
    if name == "peak_shaving":
        return PeakShavingController(peak_limit_kw, reserve_fraction=RESERVE_FRACTION)
    raise ValueError(f"Unknown baseline controller: {name}")


def _peak_limit(params: dict) -> tuple[float, dict[str, Any]]:
    if not 0.0 < PEAK_QUANTILE < 1.0:
        raise ValueError("PEAK_QUANTILE must be between zero and one")
    scaling = {
        "P_L_nom_kw": float(params["Load"]["Pmax_kw"]),
        "P_PV_nom_kw": float(params["PV"]["Pmax_kw"]),
    }
    load, pv = load_series_scaled(scaling, str(LOAD_TRAIN_CSV), str(PV_TRAIN_CSV))
    aligned = pd.concat({"load": load, "pv": pv}, axis=1, join="inner").dropna()
    net_import = (aligned["load"] - aligned["pv"]).clip(lower=0.0)
    limit = float(net_import.quantile(PEAK_QUANTILE))
    return limit, {
        "method": "training net-load quantile",
        "quantile": PEAK_QUANTILE,
        "import_limit_kw": limit,
        "samples": len(net_import),
        "load_train_sha256": sha256_file(LOAD_TRAIN_CSV),
        "pv_train_sha256": sha256_file(PV_TRAIN_CSV),
    }


def _exogenous_hash(operation: pd.DataFrame) -> str:
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
    values["outage_active"] = _boolean(values["outage_active"]).astype(int)
    return hashlib.sha256(
        values.to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()


def _trajectory_metrics(operation: pd.DataFrame, params: dict) -> dict[str, Any]:
    dt_h = float(params["time"]["timestep"]) / 60.0

    def values(column: str) -> pd.Series:
        if column not in operation:
            return pd.Series(0.0, index=operation.index, dtype=float)
        return pd.to_numeric(operation[column], errors="coerce").fillna(0.0)

    outage = _boolean(operation["outage_active"])
    outage_start = outage & ~outage.shift(1, fill_value=False)
    shed = values("Shedding_kw")
    curtailed = values("Curtailment_kw")
    soc = values("SoC_pct")
    energy = values("E_kwh")
    initial_energy_kwh = float(params["BESS"]["E_init_kwh"])
    terminal_energy_delta_kwh = float(energy.iloc[-1] - initial_energy_kwh)
    tariff = values("TOU_cperkwh")
    available_tariff = tariff[tariff > 0.0]
    mean_available_tariff = float(
        available_tariff.mean() if not available_tariff.empty else tariff.mean()
    )
    eta_c = max(float(params["BESS"]["eta_c"]), 1e-12)
    wear_per_kwh = float(params["costs"]["bess_degradation_per_kwh"])
    terminal_inventory_value_per_kwh = (
        mean_available_tariff + wear_per_kwh
    ) / eta_c
    # Positive delta means useful stored inventory remains and is credited;
    # negative delta is the standardized replacement cost needed for closure.
    terminal_inventory_adjustment_cost = (
        -terminal_energy_delta_kwh * terminal_inventory_value_per_kwh
    )
    return {
        "evaluated_hours": float(len(operation) * dt_h),
        "grid_import_kwh": float(values("P_grid_in_kw").sum() * dt_h),
        "grid_export_kwh": float(values("P_grid_out_kw").sum() * dt_h),
        "peak_grid_import_kw": float(values("P_grid_in_kw").max()),
        "load_demand_kwh": float(values("Load_kw").sum() * dt_h),
        "load_served_kwh": float(values("Load_served_kw").sum() * dt_h),
        "load_shed_kwh": float(shed.sum() * dt_h),
        "outage_load_shed_kwh": float(shed[outage].sum() * dt_h),
        "load_shed_hours": float((shed > 1e-9).sum() * dt_h),
        "outage_load_shed_hours": float(((shed > 1e-9) & outage).sum() * dt_h),
        "max_load_shed_kw": float(shed.max()),
        "pv_available_kwh": float(values("PV_kw").sum() * dt_h),
        "pv_used_kwh": float(values("PV_used_kw").sum() * dt_h),
        "pv_curtailed_kwh": float(curtailed.sum() * dt_h),
        "outage_pv_curtailed_kwh": float(curtailed[outage].sum() * dt_h),
        "bess_charge_kwh": float(values("P_bess_charge_mag_kw").sum() * dt_h),
        "bess_discharge_kwh": float(values("P_bess_discharge_kw").sum() * dt_h),
        "min_soc_pct": float(soc.min()),
        "max_soc_pct": float(soc.max()),
        "final_soc_pct": float(soc.iloc[-1]),
        "initial_energy_kwh": initial_energy_kwh,
        "final_energy_kwh": float(energy.iloc[-1]),
        "terminal_energy_delta_kwh": terminal_energy_delta_kwh,
        "terminal_inventory_value_per_kwh": terminal_inventory_value_per_kwh,
        "terminal_inventory_adjustment_cost": terminal_inventory_adjustment_cost,
        "outage_events": int(outage_start.sum()),
        "outage_hours": float(outage.sum() * dt_h),
        "mean_soc_at_outage_start_pct": (
            float(soc[outage_start].mean()) if outage_start.any() else float("nan")
        ),
    }


def _audit(metrics: dict[str, Any]) -> tuple[bool, str]:
    failures = []
    for name, tolerance in CACHE_PHYSICAL_LIMITS.items():
        value = float(metrics.get(name, 0.0) or 0.0)
        if value > tolerance:
            failures.append(f"{name}={value:.12g}")
    if int(metrics.get("n_solve_fail", 0) or 0):
        failures.append(f"n_solve_fail={metrics['n_solve_fail']}")
    return not failures, " | ".join(failures)


def _load_case(case: str) -> dict[str, Any]:
    artifact_path = SIZING_ROOT / case / "sizing_decision_variables.json"
    for path in (PARAMETERS_JSON, artifact_path, LOAD_CSV, PV_CSV):
        if not path.is_file():
            raise FileNotFoundError(path)
    params = load_sized_parameters_from_artifact(PARAMETERS_JSON, artifact_path)
    return {
        "case": case,
        "params": params,
        "mesh": _mesh_from_params(params),
        "sizing_sha256": sha256_file(artifact_path),
        "baseline_root": BASELINE_ROOT / case,
        "operation_root": OPERATION_ROOT / case,
    }


def _execute_task(task: dict[str, Any]) -> dict[str, Any]:
    params = deepcopy(task["params"])
    controller = _controller(task["controller"], params, task["peak_limit_kw"])
    metrics = simulate_baseline(
        params=params,
        controller=controller,
        start_ts=task["start"],
        n_iters=task["n_iters"],
        out_dir=task["out_dir"],
        load_csv=str(LOAD_CSV),
        pv_csv=str(PV_CSV),
        resume=not FRESH,
    )
    operation = read_operation(task["out_dir"])
    metrics.update(_trajectory_metrics(operation, params))
    metrics["operation_total_cost_inventory_adjusted"] = float(
        metrics["operation_total_cost"]
        + metrics["terminal_inventory_adjustment_cost"]
    )
    audit_pass, audit_reasons = _audit(metrics)
    metrics.update(
        case_id=f"baseline/{task['case']}/{task['month']}/{controller.name}",
        stage="baselines",
        sizing_case=task["case"],
        controller_name=controller.name,
        controller_family="rule_based",
        month=task["month"],
        outage_seed=task["seed"],
        start=task["start"],
        h=task["mesh"]["h"],
        t1=task["mesh"]["t1"],
        t2=task["mesh"]["t2"],
        combo=_mesh_tag(task["mesh"]),
        sizing_source_sha256=task["sizing_sha256"],
        exogenous_sha256=_exogenous_hash(operation),
        pairing_status="not_checked",
        audit_pass=audit_pass,
        audit_reasons=audit_reasons,
    )
    return metrics


def _tasks(case_data: dict[str, Any], peak_limit_kw: float) -> list[dict[str, Any]]:
    tasks = []
    for month in MONTHS:
        params = deepcopy(case_data["params"])
        seed = MONTHLY_OUTAGE_SEEDS[month]
        params["EDS"]["seed"] = seed
        for controller in CONTROLLERS:
            tasks.append(
                {
                    "case": case_data["case"],
                    "month": month,
                    "seed": seed,
                    "start": f"{month}-01",
                    "n_iters": N_ITERS,
                    "mesh": case_data["mesh"],
                    "params": params,
                    "controller": controller,
                    "peak_limit_kw": peak_limit_kw,
                    "sizing_sha256": case_data["sizing_sha256"],
                    "out_dir": case_data["baseline_root"] / month / controller,
                }
            )
    return tasks


def _run_tasks(tasks: list[dict[str, Any]], output: Path) -> pd.DataFrame:
    if WORKERS != 1:
        raise ValueError(
            "3-baselines.py currently requires WORKERS = 1 for deterministic recovery"
        )
    summary_path = output / "summary.csv"
    state_path = output / "pipeline_state.json"
    rows_by_case: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for completed, task in enumerate(tasks, start=1):
        case_id = f"baseline/{task['case']}/{task['month']}/{task['controller']}"
        try:
            row = _execute_task(task)
            status = "ok"
        except Exception as error:  # noqa: BLE001 - isolate failures by case
            row = {
                "case_id": case_id,
                "stage": "baselines",
                "sizing_case": task["case"],
                "controller_name": task["controller"],
                "controller_family": "rule_based",
                "month": task["month"],
                "status": "error",
                "audit_pass": False,
                "pairing_status": "not_checked",
                "error": str(error),
            }
            status = f"FAILED: {error}"
        rows_by_case[case_id] = row
        rows = pd.DataFrame(rows_by_case.values())
        _write_csv(rows, summary_path)
        elapsed = time.perf_counter() - started
        _write_json(
            {
                "stage": "baselines",
                "completed": completed,
                "total": len(tasks),
                "valid": int((rows["status"] != "error").sum()),
                "elapsed_s": elapsed,
                "eta_s": elapsed * (len(tasks) - completed) / completed,
                "updated_at": pd.Timestamp.now().isoformat(),
            },
            state_path,
        )
        print(f"[{completed}/{len(tasks)}] {case_id}: {status}", flush=True)
    return pd.DataFrame(rows_by_case.values())


SUM_FIELDS = (
    "evaluated_hours",
    "operation_grid_reliability_cost",
    "operation_net_grid_cost",
    "operation_load_shedding_cost",
    "operation_pv_curtailment_cost",
    "operation_wear_cost",
    "operation_total_cost",
    "terminal_inventory_adjustment_cost",
    "operation_total_cost_inventory_adjusted",
    "operation_bess_throughput_kwh",
    "grid_import_kwh",
    "grid_export_kwh",
    "load_demand_kwh",
    "load_served_kwh",
    "load_shed_kwh",
    "outage_load_shed_kwh",
    "load_shed_hours",
    "outage_load_shed_hours",
    "pv_available_kwh",
    "pv_used_kwh",
    "pv_curtailed_kwh",
    "outage_pv_curtailed_kwh",
    "bess_charge_kwh",
    "bess_discharge_kwh",
    "outage_events",
    "outage_hours",
)


def _aggregate(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty or "status" not in monthly or (monthly["status"] == "error").any():
        raise RuntimeError("Cannot aggregate an incomplete selection")
    rows = []
    for controller, group in monthly.groupby("controller_name", sort=True):
        row = {
            "sizing_case": str(group["sizing_case"].iloc[0]),
            "controller_name": controller,
            "controller_family": str(group["controller_family"].iloc[0]),
            "windows": int(group["month"].nunique()),
            "trajectory_count": len(group),
            "peak_grid_import_kw": float(group["peak_grid_import_kw"].max()),
            "max_load_shed_kw": float(group["max_load_shed_kw"].max()),
            "min_soc_pct": float(group["min_soc_pct"].min()),
            "max_soc_pct": float(group["max_soc_pct"].max()),
            "mean_final_soc_pct": float(group["final_soc_pct"].mean()),
            "mean_final_energy_kwh": float(group["final_energy_kwh"].mean()),
            "terminal_energy_delta_kwh": float(
                group["terminal_energy_delta_kwh"].sum()
            ),
            "terminal_inventory_value_per_kwh": float(
                group["terminal_inventory_value_per_kwh"].mean()
            ),
            "mean_soc_at_outage_start_pct": float(
                group["mean_soc_at_outage_start_pct"].mean()
            ),
            "total_runtime_h": float(group["total_time_s"].sum() / 3600.0),
            "all_cases_audited": bool(_boolean(group["audit_pass"]).all()),
        }
        if "pairing_pass" in group:
            row["all_cases_paired"] = bool(_boolean(group["pairing_pass"]).all())
        for field in SUM_FIELDS:
            if field in group:
                row[field] = float(
                    pd.to_numeric(group[field], errors="coerce").fillna(0.0).sum()
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _load_available_references(
    case_data: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    summary_path = case_data["operation_root"] / "03-forecast-operation" / "summary.csv"
    warnings = []
    if not summary_path.is_file():
        return pd.DataFrame(), ["forecast summary is not available"]
    summary = pd.read_csv(summary_path)
    required = {
        "month",
        "controller_name",
        "status",
        "audit_pass",
        "sizing_source_sha256",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        return pd.DataFrame(), ["forecast summary lacks columns: " + ", ".join(missing)]

    candidates = summary[
        summary["month"].astype(str).isin(MONTHS)
        & summary["controller_name"].isin(REFERENCE_CONTROLLERS)
        & (summary["status"] != "error")
        & _boolean(summary["audit_pass"])
        & (summary["sizing_source_sha256"] == case_data["sizing_sha256"])
    ].copy()
    duplicated = candidates.duplicated(["month", "controller_name"], keep=False)
    if duplicated.any():
        warnings.append(
            f"discarded {int(duplicated.sum())} ambiguous duplicate reference row(s)"
        )
        candidates = candidates[~duplicated]

    rows = []
    for _, source in candidates.iterrows():
        month = str(source["month"])
        controller = str(source["controller_name"])
        operation_dir = (
            case_data["operation_root"] / "03-forecast-operation" / month / controller
        )
        try:
            operation = read_operation(operation_dir, nrows=N_ITERS)
            if len(operation) != N_ITERS:
                raise ValueError(f"expected {N_ITERS} rows, found {len(operation)}")
        except (FileNotFoundError, ValueError) as error:
            warnings.append(f"{month}/{controller}: {error}")
            continue
        params = deepcopy(case_data["params"])
        params["EDS"]["seed"] = MONTHLY_OUTAGE_SEEDS[month]
        row = source.to_dict()
        row.update(_trajectory_metrics(operation, params))
        row["operation_total_cost_inventory_adjusted"] = float(
            row["operation_total_cost"]
            + row["terminal_inventory_adjustment_cost"]
        )
        row.update(
            sizing_case=case_data["case"],
            controller_family=(
                "perfect_information" if controller == "ideal" else "mpc"
            ),
            exogenous_sha256=_exogenous_hash(operation),
            pairing_pass=True,
        )
        rows.append(row)
    return pd.DataFrame(rows), warnings


def _paired_selection(
    baseline_monthly: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    reference = reference.copy()
    hashes = reference.set_index("month")["exogenous_sha256"].to_dict()
    baseline = baseline_monthly[
        (baseline_monthly["status"] != "error")
        & _boolean(baseline_monthly["audit_pass"])
        & baseline_monthly["month"].isin(hashes)
    ].copy()
    baseline["pairing_pass"] = baseline.apply(
        lambda row: row["exogenous_sha256"] == hashes.get(row["month"]), axis=1
    )
    baseline = baseline[_boolean(baseline["pairing_pass"])]

    common_months = set(reference["month"].astype(str))
    for controller in CONTROLLERS:
        controller_months = set(
            baseline.loc[baseline["controller_name"] == controller, "month"].astype(str)
        )
        common_months &= controller_months
    ordered_months = [month for month in MONTHS if month in common_months]
    return (
        baseline[baseline["month"].isin(ordered_months)].copy(),
        reference[reference["month"].isin(ordered_months)].copy(),
        ordered_months,
    )


def _add_cost_delta(frame: pd.DataFrame, reference_controller: str) -> pd.DataFrame:
    frame = frame.copy()
    reference_cost = float(
        frame.loc[
            frame["controller_name"] == reference_controller, "operation_total_cost"
        ].iloc[0]
    )
    frame["delta_cost_vs_reference"] = frame["operation_total_cost"] - reference_cost
    frame["delta_cost_pct_vs_reference"] = (
        100.0 * frame["delta_cost_vs_reference"] / max(abs(reference_cost), 1e-12)
    )
    adjusted_cost = "operation_total_cost_inventory_adjusted"
    adjusted_reference = float(
        frame.loc[
            frame["controller_name"] == reference_controller, adjusted_cost
        ].iloc[0]
    )
    frame["delta_inventory_adjusted_cost_vs_reference"] = (
        frame[adjusted_cost] - adjusted_reference
    )
    frame["delta_inventory_adjusted_cost_pct_vs_reference"] = (
        100.0
        * frame["delta_inventory_adjusted_cost_vs_reference"]
        / max(abs(adjusted_reference), 1e-12)
    )
    return frame


def _partial_comparison(
    baseline_monthly: pd.DataFrame, reference_monthly: pd.DataFrame
) -> pd.DataFrame:
    if reference_monthly.empty:
        return pd.DataFrame()
    comparisons = []
    for reference_controller in REFERENCE_CONTROLLERS:
        reference = reference_monthly[
            reference_monthly["controller_name"] == reference_controller
        ]
        if reference.empty:
            continue
        baseline, reference, paired_months = _paired_selection(
            baseline_monthly, reference
        )
        if not paired_months:
            continue
        comparison = pd.concat(
            [_aggregate(reference), _aggregate(baseline)], ignore_index=True, sort=False
        )
        comparison = _add_cost_delta(comparison, reference_controller)
        comparison.insert(0, "paired_reference", reference_controller)
        comparison.insert(1, "paired_months", ",".join(paired_months))
        comparison.insert(2, "complete_120d", len(paired_months) == len(MONTHS))
        comparisons.append(comparison)
    return (
        pd.concat(comparisons, ignore_index=True, sort=False)
        if comparisons
        else pd.DataFrame()
    )


def _full_comparison(
    baseline_monthly: pd.DataFrame, reference_monthly: pd.DataFrame
) -> pd.DataFrame:
    if reference_monthly.empty:
        return pd.DataFrame()
    expected_months = set(MONTHS)
    for controller in REFERENCE_CONTROLLERS:
        present = set(
            reference_monthly.loc[
                reference_monthly["controller_name"] == controller, "month"
            ].astype(str)
        )
        if present != expected_months:
            return pd.DataFrame()
    if (reference_monthly.groupby("month")["exogenous_sha256"].nunique() != 1).any():
        return pd.DataFrame()
    prototype = reference_monthly[reference_monthly["controller_name"] == "prototype"]
    baseline, prototype, paired_months = _paired_selection(baseline_monthly, prototype)
    if set(paired_months) != expected_months:
        return pd.DataFrame()
    comparison = pd.concat(
        [_aggregate(reference_monthly), _aggregate(baseline)],
        ignore_index=True,
        sort=False,
    )
    return _add_cost_delta(comparison, "prototype")


def _run_case(case_data: dict[str, Any]) -> dict[str, Any]:
    case = case_data["case"]
    output = case_data["baseline_root"]
    peak_limit_kw, calibration = _peak_limit(case_data["params"])
    print(f"\n=== baselines / {case} / peak limit {peak_limit_kw:.6f} kW ===")
    monthly = _run_tasks(_tasks(case_data, peak_limit_kw), output)
    successful = monthly[
        (monthly["status"] != "error") & _boolean(monthly["audit_pass"])
    ].copy()
    expected = len(MONTHS) * len(CONTROLLERS)
    if len(successful) != expected:
        raise RuntimeError(
            f"{case}: baseline campaign has {len(successful)}/{expected} valid cases"
        )
    baseline_aggregate = _aggregate(successful)
    references, warnings = _load_available_references(case_data)
    partial = _partial_comparison(successful, references)
    full = _full_comparison(successful, references)

    _write_csv(baseline_aggregate, output / "baseline_summary_120d.csv")
    _write_csv(partial, output / "controller_comparison_partial.csv")
    if not full.empty:
        _write_csv(full, output / "controller_comparison_120d.csv")
    coverage = {
        controller: (
            0
            if references.empty
            else int(
                references.loc[
                    references["controller_name"] == controller, "month"
                ].nunique()
            )
        )
        for controller in REFERENCE_CONTROLLERS
    }
    _write_json(
        {
            "campaign_version": CAMPAIGN_VERSION,
            "baseline_model_version": BASELINE_MODEL_VERSION,
            "operation_model_version": OPERATION_MODEL_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sizing_case": case,
            "sizing_source_sha256": case_data["sizing_sha256"],
            "baseline_parameter_mesh": case_data["mesh"],
            "controllers": list(CONTROLLERS),
            "months": list(MONTHS),
            "n_iters_per_month": N_ITERS,
            "reserve_fraction": RESERVE_FRACTION,
            "peak_shaving_calibration": calibration,
            "baseline_cases": len(monthly),
            "audited_baseline_cases": int(_boolean(monthly["audit_pass"]).sum()),
            "operation_reference_coverage_months": coverage,
            "full_120d_comparison_available": not full.empty,
            "comparison_warnings": warnings,
        },
        output / "manifest.json",
    )
    return {
        "baseline": baseline_aggregate,
        "partial": partial,
        "full": full,
        "coverage": coverage,
        "warnings": warnings,
    }


def main() -> None:
    if WORKERS != 1 or N_ITERS < 1:
        raise ValueError("WORKERS must be 1 and N_ITERS must be positive")
    baseline_outputs = []
    partial_outputs = []
    full_outputs = []
    cases = {}
    for case in SIZING_CASES:
        try:
            result = _run_case(_load_case(case))
        except FileNotFoundError as error:
            cases[case] = {"status": "skipped", "reason": str(error)}
            print(f"[skip] {case}: missing required sizing/input artifact: {error}")
            continue
        baseline_outputs.append(result["baseline"])
        if not result["partial"].empty:
            partial_outputs.append(result["partial"])
        if not result["full"].empty:
            full_outputs.append(result["full"])
        cases[case] = {
            "status": "complete",
            "operation_reference_coverage_months": result["coverage"],
            "full_120d_comparison_available": not result["full"].empty,
            "comparison_warnings": result["warnings"],
        }

    if not baseline_outputs:
        raise RuntimeError("No sizing case is ready for baseline evaluation")
    _write_csv(
        pd.concat(baseline_outputs, ignore_index=True, sort=False),
        BASELINE_ROOT / "baseline_summary_120d.csv",
    )
    _write_csv(
        pd.concat(partial_outputs, ignore_index=True, sort=False)
        if partial_outputs
        else pd.DataFrame(),
        BASELINE_ROOT / "baseline_comparison_partial.csv",
    )
    if full_outputs:
        _write_csv(
            pd.concat(full_outputs, ignore_index=True, sort=False),
            BASELINE_ROOT / "baseline_comparison_120d.csv",
        )
    _write_json(
        {
            "campaign_version": CAMPAIGN_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "cases": cases,
        },
        BASELINE_ROOT / "manifest.json",
    )
    print(f"\nBaseline artifacts written to {BASELINE_ROOT}", flush=True)


if __name__ == "__main__":
    main()
