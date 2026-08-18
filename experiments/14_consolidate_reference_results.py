"""Consolidate and audit the annual temporal-mesh results offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opt.artifacts import operation_artifact_path, read_operation
from opt.campaigns import (
    CampaignProvenance,
    campaign_layout,
    load_campaign_provenance,
    validate_parameter_provenance,
    validate_summary_provenance,
)
from opt.operation import _operation_cost_metrics, _operation_physical_metrics


DEFAULT_CAMPAIGN_ROOT = ROOT / "outputs" / "sweeps" / "economic"

EXPECTED_CONTROLLERS = ("ideal", "lstm", "prototype")
EXPECTED_MONTHS = (
    "2009-05", "2009-06", "2009-07", "2009-08", "2009-09", "2009-10",
    "2009-11", "2009-12", "2010-01", "2010-02", "2010-03", "2010-04",
)
EXPECTED_CONFIGURATIONS = 27
EXPECTED_ROWS = len(EXPECTED_CONTROLLERS) * len(EXPECTED_MONTHS) * EXPECTED_CONFIGURATIONS
TOLERANCE = 1e-8

COST_FIELDS = (
    "operation_grid_reliability_cost",
    "operation_bess_throughput_kwh",
    "operation_wear_cost",
    "operation_total_cost",
    "operation_mean_cost",
)
PHYSICAL_FIELDS = (
    "max_abs_power_balance_residual_kw",
    "total_abs_power_balance_residual_kwh",
    "max_grid_exchange_during_outage_kw",
    "max_bess_energy_bound_violation_kwh",
    "max_bess_energy_transition_residual_kwh",
    "max_plant_simultaneous_ch_dis_kw",
)
REQUIRED_ARTIFACTS = (
    "metrics.json", "parameters_used.json", "outage_calendar.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def numeric(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df:
        raise ValueError(f"Missing operation column: {name}")
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0)


def boolean(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df:
        raise ValueError(f"Missing operation column: {name}")
    raw = df[name]
    if pd.api.types.is_bool_dtype(raw):
        return raw.fillna(False)
    return raw.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def exogenous_hash(operation: pd.DataFrame) -> str:
    columns = ["timestamp", "Load_kw", "PV_kw", "outage_active"]
    missing = sorted(set(columns).difference(operation.columns))
    if missing:
        raise ValueError("Missing exogenous columns: " + ", ".join(missing))
    values = operation[columns].copy()
    values["timestamp"] = values["timestamp"].astype(str)
    values["Load_kw"] = pd.to_numeric(values["Load_kw"], errors="raise").map(
        lambda value: format(float(value), ".15g")
    )
    values["PV_kw"] = pd.to_numeric(values["PV_kw"], errors="raise").map(
        lambda value: format(float(value), ".15g")
    )
    values["outage_active"] = boolean(operation, "outage_active").astype(int)
    return hashlib.sha256(
        values.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def case_dir_for(row: pd.Series, may_root: Path, seasonal_root: Path) -> Path:
    controller = str(row["controller_name"])
    combo = str(row["combo"])
    if str(row["month"]) == EXPECTED_MONTHS[0]:
        return may_root / controller / combo
    return seasonal_root / str(row["month"]) / controller / combo


def validate_summary(summary: pd.DataFrame) -> None:
    required = {
        "month", "combo", "h", "t1", "t2", "controller_name", "audit_pass",
        *COST_FIELDS, *PHYSICAL_FIELDS,
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise RuntimeError("Annual summary is missing columns: " + ", ".join(missing))
    if len(summary) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} summary rows, found {len(summary)}")
    if tuple(sorted(summary["month"].unique())) != tuple(sorted(EXPECTED_MONTHS)):
        raise RuntimeError("Annual summary does not contain the expected twelve months")
    if set(summary["controller_name"]) != set(EXPECTED_CONTROLLERS):
        raise RuntimeError("Annual summary must contain ideal, LSTM, and prototype")
    if summary["combo"].nunique() != EXPECTED_CONFIGURATIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_CONFIGURATIONS} configurations, "
            f"found {summary['combo'].nunique()}"
        )
    keys = ["month", "controller_name", "combo"]
    if summary.duplicated(keys).any():
        raise RuntimeError("Duplicate annual-summary case identifiers detected")


def selected_mesh(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if all(name in data for name in ("h", "t1", "t2")):
        return {name: int(data[name]) for name in ("h", "t1", "t2")}
    selected = data.get("selected", [])
    if len(selected) != 1:
        raise RuntimeError(f"Expected one selected mesh in {path}")
    return {name: int(selected[0][name]) for name in ("h", "t1", "t2")}


def additional_metrics(operation: pd.DataFrame, params: dict) -> dict[str, Any]:
    dt_h = float(params.get("time", {}).get("timestep", 5.0)) / 60.0
    outage = boolean(operation, "outage_active")
    outage_start = outage & ~outage.shift(1, fill_value=False)
    shed = numeric(operation, "Shedding_kw").clip(lower=0.0)
    curt = numeric(operation, "Curtailment_kw").clip(lower=0.0)
    grid_in = numeric(operation, "P_grid_in_kw").clip(lower=0.0)
    grid_out = numeric(operation, "P_grid_out_kw").clip(lower=0.0)
    charge = numeric(operation, "P_bess_charge_mag_kw").clip(lower=0.0)
    discharge = numeric(operation, "P_bess_discharge_kw").clip(lower=0.0)
    soc = numeric(operation, "SoC_pct")
    start_soc = soc.loc[outage_start]
    positive_shed = shed > TOLERANCE

    metrics: dict[str, Any] = {
        "evaluated_hours": float(len(operation) * dt_h),
        "grid_import_kwh": float(grid_in.sum() * dt_h),
        "grid_export_kwh": float(grid_out.sum() * dt_h),
        "peak_grid_import_kw": float(grid_in.max()),
        "load_demand_kwh": float(numeric(operation, "Load_kw").sum() * dt_h),
        "load_served_kwh": float(numeric(operation, "Load_served_kw").sum() * dt_h),
        "load_shed_kwh": float(shed.sum() * dt_h),
        "outage_load_shed_kwh": float(shed.loc[outage].sum() * dt_h),
        "load_shed_hours": float(positive_shed.sum() * dt_h),
        "outage_load_shed_hours": float((positive_shed & outage).sum() * dt_h),
        "max_load_shed_kw": float(shed.max()),
        "pv_available_kwh": float(numeric(operation, "PV_kw").sum() * dt_h),
        "pv_used_kwh": float(numeric(operation, "PV_used_kw").sum() * dt_h),
        "pv_curtailed_kwh": float(curt.sum() * dt_h),
        "outage_pv_curtailed_kwh": float(curt.loc[outage].sum() * dt_h),
        "bess_charge_kwh": float(charge.sum() * dt_h),
        "bess_discharge_kwh": float(discharge.sum() * dt_h),
        "min_soc_pct": float(soc.min()),
        "max_soc_pct": float(soc.max()),
        "final_soc_pct": float(soc.iloc[-1]),
        "outage_events": int(outage_start.sum()),
        "outage_hours": float(outage.sum() * dt_h),
        "mean_soc_at_outage_start_pct": (
            float(start_soc.mean()) if len(start_soc) else np.nan
        ),
        "min_soc_at_outage_start_pct": (
            float(start_soc.min()) if len(start_soc) else np.nan
        ),
        "cost_grid": float(numeric(operation, "cost_grid").sum()),
        "cost_shed": float(numeric(operation, "cost_shed").sum()),
        "cost_curt": float(numeric(operation, "cost_curt").sum()),
        "noise_steps": int(boolean(operation, "noise_applied").sum()),
    }
    component_cost = (
        numeric(operation, "cost_grid")
        + numeric(operation, "cost_shed")
        + numeric(operation, "cost_curt")
    )
    metrics["max_stage_cost_component_error"] = float(
        (numeric(operation, "cost_total") - component_cost).abs().max()
    )
    return metrics


def compare_metric(
    record: dict[str, Any], prefix: str, field: str, expected: Any, actual: Any,
) -> None:
    try:
        error = abs(float(expected) - float(actual))
    except (TypeError, ValueError):
        error = 0.0 if expected == actual else np.inf
    record[f"{prefix}_{field}_error"] = float(error)


def audit_case(
    row: pd.Series, may_root: Path, seasonal_root: Path, project_root: Path,
    provenance: CampaignProvenance,
) -> dict[str, Any]:
    case_dir = case_dir_for(row, may_root, seasonal_root)
    missing = [name for name in REQUIRED_ARTIFACTS if not (case_dir / name).exists()]
    try:
        operation_path = operation_artifact_path(case_dir)
    except FileNotFoundError:
        operation_path = case_dir / "operation_final.parquet"
        missing.append("operation_final.parquet|operation_final.csv")
    base = {
        "month": str(row["month"]),
        "controller_name": str(row["controller_name"]),
        "combo": str(row["combo"]),
        "h": int(row["h"]),
        "t1": int(row["t1"]),
        "t2": int(row["t2"]),
        "source_stage": "01-mesh" if str(row["month"]) == EXPECTED_MONTHS[0] else "02-seasonal-mesh",
        "case_dir": case_dir.relative_to(project_root).as_posix(),
        "campaign_id": provenance.campaign_id,
        "sizing_source_sha256": provenance.sizing_sha256,
        "artifacts_complete": not missing,
        "missing_artifacts": ";".join(missing),
    }
    if missing:
        base.update(case_audit_pass=False, audit_failure="missing artifacts")
        return base

    params_path = case_dir / "parameters_used.json"
    metrics_path = case_dir / "metrics.json"
    calendar_path = case_dir / "outage_calendar.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    operation = read_operation(operation_path)
    core_params = {
        key: value for key, value in params.items() if not key.startswith("_operation_")
    }
    validate_parameter_provenance(core_params, provenance, str(case_dir))

    reconstructed = _operation_cost_metrics(operation, params)
    reconstructed.update(_operation_physical_metrics(operation, params))
    reconstructed.update(additional_metrics(operation, params))
    base.update(reconstructed)
    base.update(
        operation_rows=int(len(operation)),
        operation_file_size_bytes=int(operation_path.stat().st_size),
        operation_sha256=sha256_file(operation_path),
        metrics_sha256=sha256_file(metrics_path),
        parameters_sha256=sha256_file(params_path),
        outage_calendar_sha256=sha256_file(calendar_path),
        core_parameters_sha256=canonical_hash(core_params),
        exogenous_sha256=exogenous_hash(operation),
        run_fingerprint_sha256=str(saved_metrics.get("run_fingerprint_sha256", "")),
        parameter_fingerprint_sha256=str(saved_metrics.get("parameter_fingerprint_sha256", "")),
        source_summary_audit_pass=bool(row["audit_pass"]),
    )

    for field in (*COST_FIELDS, *PHYSICAL_FIELDS):
        compare_metric(base, "metrics", field, saved_metrics.get(field), reconstructed[field])
        compare_metric(base, "summary", field, row[field], reconstructed[field])

    error_columns = [name for name in base if name.endswith("_error")]
    max_reconstruction_error = max(float(base[name]) for name in error_columns)
    base["max_reconstruction_error"] = max_reconstruction_error
    base["case_audit_pass"] = bool(
        row["audit_pass"]
        and len(operation) == int(saved_metrics.get("operation_rows", len(operation)))
        and max_reconstruction_error <= TOLERANCE
        and reconstructed["max_stage_cost_component_error"] <= TOLERANCE
        and reconstructed["max_abs_power_balance_residual_kw"] <= TOLERANCE
        and reconstructed["max_grid_exchange_during_outage_kw"] <= TOLERANCE
        and reconstructed["max_bess_energy_bound_violation_kwh"] <= TOLERANCE
        and reconstructed["max_bess_energy_transition_residual_kwh"] <= TOLERANCE
        and reconstructed["max_plant_simultaneous_ch_dis_kw"] <= TOLERANCE
    )
    base["audit_failure"] = "" if base["case_audit_pass"] else "metric or physical audit"
    return base


def pairing_audit(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["month", "combo", "h", "t1", "t2"]
    for key, group in cases.groupby(keys, sort=True):
        controllers = sorted(group["controller_name"].tolist())
        row = dict(zip(keys, key))
        row.update(
            controllers=";".join(controllers),
            controller_count=int(group["controller_name"].nunique()),
            exogenous_hash_count=int(group["exogenous_sha256"].nunique()),
            outage_calendar_hash_count=int(group["outage_calendar_sha256"].nunique()),
            core_parameter_hash_count=int(group["core_parameters_sha256"].nunique()),
            operation_row_count=int(group["operation_rows"].nunique()),
        )
        row["pairing_pass"] = bool(
            controllers == sorted(EXPECTED_CONTROLLERS)
            and row["exogenous_hash_count"] == 1
            and row["outage_calendar_hash_count"] == 1
            and row["core_parameter_hash_count"] == 1
            and row["operation_row_count"] == 1
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_selected(cases: pd.DataFrame) -> pd.DataFrame:
    sum_fields = [
        "evaluated_hours", "operation_grid_reliability_cost", "operation_wear_cost",
        "operation_total_cost", "grid_import_kwh", "grid_export_kwh",
        "load_demand_kwh", "load_served_kwh", "load_shed_kwh",
        "outage_load_shed_kwh", "load_shed_hours", "outage_load_shed_hours",
        "pv_available_kwh", "pv_used_kwh", "pv_curtailed_kwh",
        "outage_pv_curtailed_kwh", "bess_charge_kwh", "bess_discharge_kwh",
        "operation_bess_throughput_kwh", "outage_events", "outage_hours",
        "cost_grid", "cost_shed", "cost_curt", "noise_steps",
    ]
    rows = []
    for controller, group in cases.groupby("controller_name", sort=True):
        row: dict[str, Any] = {
            "campaign_id": str(group["campaign_id"].iloc[0]),
            "sizing_source_sha256": str(group["sizing_source_sha256"].iloc[0]),
            "controller_name": controller,
            "windows": int(group["month"].nunique()),
            "trajectory_count": int(len(group)),
            "mean_solve_time_s": float(group["avg_solve_time_s"].mean()),
            "total_runtime_h": float(group["total_time_s"].sum() / 3600.0),
            "peak_grid_import_kw": float(group["peak_grid_import_kw"].max()),
            "max_load_shed_kw": float(group["max_load_shed_kw"].max()),
            "min_soc_pct": float(group["min_soc_pct"].min()),
            "max_soc_pct": float(group["max_soc_pct"].max()),
            "mean_soc_at_outage_start_pct": float(
                group["mean_soc_at_outage_start_pct"].mean()
            ),
            "all_cases_audited": bool(group["case_audit_pass"].all()),
        }
        row.update({field: float(group[field].sum()) for field in sum_fields})
        rows.append(row)
    return pd.DataFrame(rows)


def write_readme(
    output: Path, mesh: dict[str, int], manifest: dict[str, Any], campaign_root: Path,
) -> None:
    text = f"""# Canonical operational reference set

This directory consolidates the existing annual temporal-mesh campaign without
running any optimization. May (`2009-05`) is referenced from the campaign's
`01-mesh`; the other eleven months are referenced from `02-seasonal-mesh`.
Source trajectories remain in their pipeline-stage directories to avoid data
duplication and broken reproduction paths.

The selected operational mesh is $H={mesh['h']}$ h, $\\Delta t_1={mesh['t1']}$
min, and $\\Delta t_2={mesh['t2']}$ min.

## Files

- `case_manifest.csv`: canonical location, provenance hashes, reconstructed
  costs, energy/reliability metrics, and audit result for all cases.
- `pairing_audit.csv`: verifies that ideal, LSTM, and prototype received the
  same load, PV, outage realization, parameters, and number of samples.
- `selected_mesh_monthly.csv`: the 36 paired trajectories for the selected
  mesh (12 months times three controllers).
- `selected_mesh_120d_summary.csv`: 120-day controller totals used as the
  reference for future baselines.
- `audit_report.json`: machine-readable integrity summary and file hashes.

## Integrity result

- Cases: {manifest['counts']['cases']} / {EXPECTED_ROWS}
- Complete artifacts: {manifest['counts']['complete_artifacts']}
- Passed trajectory audits: {manifest['counts']['case_audits_passed']}
- Passed paired groups: {manifest['counts']['pairing_audits_passed']} / {manifest['counts']['pairing_groups']}

## Reproduction

```powershell
.\\.venv\\Scripts\\python.exe experiments\\14_consolidate_reference_results.py --campaign-root {campaign_root.relative_to(ROOT).as_posix()} --campaign-id {manifest['campaign_id']}
```
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--campaign-id")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--may-root", type=Path)
    parser.add_argument("--seasonal-root", type=Path)
    parser.add_argument("--champion", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    args.campaign_root = args.campaign_root.resolve()
    layout = campaign_layout(args.campaign_root)
    args.summary = (args.summary or layout["summary"]).resolve()
    args.may_root = (args.may_root or layout["may"]).resolve()
    args.seasonal_root = (args.seasonal_root or layout["seasonal"]).resolve()
    args.champion = (args.champion or layout["champion"]).resolve()
    args.output = (args.output or layout["reference"]).resolve()
    provenance = load_campaign_provenance(args.campaign_root, args.campaign_id)

    summary = pd.read_csv(args.summary)
    validate_summary(summary)
    validate_summary_provenance(summary, provenance, "annual mesh summary")
    mesh = selected_mesh(args.champion)
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    ordered = summary.sort_values(["month", "controller_name", "h", "t1", "t2"])
    for index, (_, row) in enumerate(ordered.iterrows(), start=1):
        records.append(
            audit_case(row, args.may_root, args.seasonal_root, ROOT, provenance)
        )
        if index % 100 == 0 or index == len(ordered):
            print(f"Audited {index}/{len(ordered)} cases", flush=True)

    cases = pd.DataFrame(records).sort_values(["month", "controller_name", "h", "t1", "t2"])
    pairs = pairing_audit(cases)
    selected = ((cases["h"] == mesh["h"]) & (cases["t1"] == mesh["t1"])
                & (cases["t2"] == mesh["t2"]))
    chosen = cases[selected].copy()
    reporting = summary[[
        "month", "controller_name", "combo", "avg_solve_time_s", "total_time_s"
    ]]
    chosen = chosen.merge(
        reporting, on=["month", "controller_name", "combo"], validate="one_to_one"
    )
    selected_summary = aggregate_selected(chosen)

    case_path = args.output / "case_manifest.csv"
    pair_path = args.output / "pairing_audit.csv"
    chosen_path = args.output / "selected_mesh_monthly.csv"
    selected_summary_path = args.output / "selected_mesh_120d_summary.csv"
    cases.to_csv(case_path, index=False)
    pairs.to_csv(pair_path, index=False)
    chosen.to_csv(chosen_path, index=False)
    selected_summary.to_csv(selected_summary_path, index=False)

    manifest = {
        "reference_version": "annual-operational-reference-v1",
        "campaign_id": provenance.campaign_id,
        "sizing_source_sha256": provenance.sizing_sha256,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "existing artifacts only; no optimization executed",
        "source_layout": {
            "2009-05": args.may_root.relative_to(ROOT).as_posix(),
            "2009-06_to_2010-04": args.seasonal_root.relative_to(ROOT).as_posix(),
        },
        "sources": {
            "annual_summary": {
                "path": args.summary.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.summary),
            },
            "champion_mesh": {
                "path": args.champion.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(args.champion),
            },
        },
        "selected_mesh": mesh,
        "tolerance": TOLERANCE,
        "counts": {
            "cases": int(len(cases)),
            "complete_artifacts": int(cases["artifacts_complete"].sum()),
            "case_audits_passed": int(cases["case_audit_pass"].sum()),
            "pairing_groups": int(len(pairs)),
            "pairing_audits_passed": int(pairs["pairing_pass"].sum()),
            "selected_mesh_cases": int(len(chosen)),
        },
        "maximum_errors": {
            "reconstruction": float(cases["max_reconstruction_error"].max()),
            "stage_cost_components": float(cases["max_stage_cost_component_error"].max()),
            "power_balance_kw": float(cases["max_abs_power_balance_residual_kw"].max()),
            "grid_exchange_during_outage_kw": float(
                cases["max_grid_exchange_during_outage_kw"].max()
            ),
            "energy_bound_kwh": float(cases["max_bess_energy_bound_violation_kwh"].max()),
            "energy_transition_kwh": float(
                cases["max_bess_energy_transition_residual_kwh"].max()
            ),
            "simultaneous_charge_discharge_kw": float(
                cases["max_plant_simultaneous_ch_dis_kw"].max()
            ),
        },
        "outputs": {
            path.name: {"sha256": sha256_file(path), "rows": int(rows)}
            for path, rows in (
                (case_path, len(cases)),
                (pair_path, len(pairs)),
                (chosen_path, len(chosen)),
                (selected_summary_path, len(selected_summary)),
            )
        },
    }
    report_path = args.output / "audit_report.json"
    report_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_readme(args.output, mesh, manifest, args.campaign_root)

    complete = (
        manifest["counts"]["cases"] == EXPECTED_ROWS
        and manifest["counts"]["complete_artifacts"] == EXPECTED_ROWS
        and manifest["counts"]["case_audits_passed"] == EXPECTED_ROWS
        and manifest["counts"]["pairing_audits_passed"] == manifest["counts"]["pairing_groups"]
        and manifest["counts"]["selected_mesh_cases"] == 36
    )
    print(json.dumps(manifest["counts"], indent=2))
    if not complete:
        raise RuntimeError(f"Reference consolidation failed; inspect {report_path}")
    print(f"Reference package written to {args.output}")


if __name__ == "__main__":
    main()
