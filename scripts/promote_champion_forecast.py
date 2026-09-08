"""Promote validated champion-mesh artifacts into the final forecast stage.

This performs no simulation. It copies byte-identical artifacts produced by
the annual mesh campaign, rebuilds the forecast summary, and records complete
provenance for the promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OPERATION_ROOT = ROOT / "outputs" / "operation-sweep" / "with-degradation"
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
CONTROLLERS = ("ideal", "prototype", "lstm")
BASE_OUTAGE_SEED = 42
REQUIRED_ARTIFACTS = (
    "metrics.json",
    "operation_final.parquet",
    "outage_calendar.json",
    "parameters_used.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    source_hash = _sha256(source)
    if _sha256(temporary) != source_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Hash mismatch while copying {source}")
    os.replace(temporary, destination)
    return source_hash


def _source_directory(case_root: Path, month: str, controller: str, tag: str) -> Path:
    if month == MONTHS[0]:
        return case_root / "01-mesh" / controller / tag
    return case_root / "02-seasonal-mesh" / month / controller / tag


def promote(case: str) -> dict[str, Any]:
    case_root = OPERATION_ROOT / case
    champion_path = case_root / "champion_mesh.json"
    seasonal_summary_path = case_root / "02-seasonal-mesh" / "summary.csv"
    if not champion_path.is_file() or not seasonal_summary_path.is_file():
        raise FileNotFoundError("Champion and seasonal summary are required")

    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    tag = f"h{int(champion['h'])}_t1_{int(champion['t1'])}_t2_{int(champion['t2'])}"
    summary = pd.read_csv(seasonal_summary_path)
    selected = summary[
        summary["combo"].eq(tag)
        & summary["month"].astype(str).isin(MONTHS)
        & summary["controller_name"].isin(CONTROLLERS)
    ].copy()
    expected = len(MONTHS) * len(CONTROLLERS)
    if (
        len(selected) != expected
        or selected.duplicated(["month", "controller_name"]).any()
    ):
        raise RuntimeError(
            f"Champion selection has {len(selected)}/{expected} unique rows"
        )
    valid = (selected["status"] != "error") & _boolean(selected["audit_pass"])
    if not valid.all():
        raise RuntimeError(
            f"Champion selection contains {int((~valid).sum())} invalid rows"
        )

    forecast_root = case_root / "03-forecast-operation"
    forecast_root.mkdir(parents=True, exist_ok=True)
    summary_path = forecast_root / "summary.csv"
    state_path = forecast_root / "pipeline_state.json"
    summary_backup = forecast_root / "summary.before_champion_reuse.csv"
    state_backup = forecast_root / "pipeline_state.before_champion_reuse.json"
    if summary_path.is_file() and not summary_backup.exists():
        shutil.copy2(summary_path, summary_backup)
    if state_path.is_file() and not state_backup.exists():
        shutil.copy2(state_path, state_backup)

    copied = []
    for month in MONTHS:
        seed = BASE_OUTAGE_SEED + MONTHS.index(month)
        for controller in CONTROLLERS:
            source = _source_directory(case_root, month, controller, tag)
            destination = forecast_root / month / controller
            missing = [
                name for name in REQUIRED_ARTIFACTS if not (source / name).is_file()
            ]
            if missing:
                raise FileNotFoundError(f"{source}: missing {', '.join(missing)}")
            metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
            if (
                metrics.get("status") == "error"
                or metrics.get("termination") != "completed"
            ):
                raise RuntimeError(f"Incomplete source metrics: {source}")
            if int(metrics.get("n_iters", 0)) != 2880:
                raise RuntimeError(f"Unexpected operation length in {source}")

            hashes = {}
            for name in REQUIRED_ARTIFACTS:
                hashes[name] = _atomic_copy(source / name, destination / name)
            copied.append(
                {
                    "month": month,
                    "outage_seed": seed,
                    "controller": controller,
                    "source_stage": "01-mesh"
                    if month == MONTHS[0]
                    else "02-seasonal-mesh",
                    "source_directory": str(source.relative_to(ROOT)),
                    "destination_directory": str(destination.relative_to(ROOT)),
                    "sha256": hashes,
                }
            )

    month_order = {month: index for index, month in enumerate(MONTHS)}
    controller_order = {name: index for index, name in enumerate(CONTROLLERS)}
    selected["_month_order"] = selected["month"].map(month_order)
    selected["_controller_order"] = selected["controller_name"].map(controller_order)
    selected.sort_values(["_month_order", "_controller_order"], inplace=True)
    selected.drop(columns=["_month_order", "_controller_order"], inplace=True)
    selected["stage"] = "forecast"
    selected["case_id"] = selected.apply(
        lambda row: (
            f"forecast/{row['month']}/{int(row['outage_seed'])}/"
            f"{row['controller_name']}/{tag}"
        ),
        axis=1,
    )
    selected["artifact_origin"] = "champion_mesh_reuse"
    selected["source_stage"] = selected["month"].map(
        lambda month: "01-mesh" if month == MONTHS[0] else "02-seasonal-mesh"
    )
    selected["source_combo"] = tag
    if "error" in selected:
        selected["error"] = ""
    _atomic_csv(summary_path, selected)

    now = datetime.now(timezone.utc).isoformat()
    provenance = {
        "schema_version": 1,
        "case": case,
        "champion_combo": tag,
        "operation_model_version": champion.get("operation_model_version"),
        "simulation_executed": False,
        "artifact_origin": "champion_mesh_reuse",
        "generated_at_utc": now,
        "source_summary": str(seasonal_summary_path.relative_to(ROOT)),
        "source_summary_sha256": _sha256(seasonal_summary_path),
        "rows": len(selected),
        "audited_rows": int(_boolean(selected["audit_pass"]).sum()),
        "copies": copied,
    }
    _atomic_json(forecast_root / "champion_reuse_manifest.json", provenance)
    _atomic_json(
        state_path,
        {
            "stage": "forecast",
            "processed": expected,
            "successful": expected,
            "failed": 0,
            "completed": expected,
            "total": expected,
            "complete": True,
            "operation_model_version": champion.get("operation_model_version"),
            "artifact_origin": "champion_mesh_reuse",
            "simulation_executed": False,
            "updated_at": now,
        },
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    result = promote(args.case)
    print(
        f"Promoted {result['rows']} audited {result['champion_combo']} artifacts "
        f"for {result['case']} without simulation."
    )


if __name__ == "__main__":
    main()
