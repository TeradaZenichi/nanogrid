"""Re-run only failed seasonal operation cases and repair campaign summaries."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from types import ModuleType

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OPERATION_SCRIPT = ROOT / "2-operation.py"
SIZING_ROOT = ROOT / "outputs" / "sizing-sweep" / "with-degradation"
OPERATION_ROOT = ROOT / "outputs" / "operation-sweep" / "with-degradation"
TARGET_CASES = ("economic", "critical_50", "full_100")


def _load_operation_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("operation_campaign", OPERATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {OPERATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _is_valid(frame: pd.DataFrame) -> pd.Series:
    audit = frame["audit_pass"]
    if not pd.api.types.is_bool_dtype(audit):
        audit = audit.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
    return frame["status"].astype(str).ne("error") & audit.fillna(False)


def _replace_row(path: Path, result: dict) -> None:
    frame = pd.read_csv(path)
    frame = frame[frame["case_id"].astype(str) != str(result["case_id"])]
    frame = pd.concat([frame, pd.DataFrame([result])], ignore_index=True, sort=False)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.repair.tmp")
    frame.to_csv(temporary, index=False)
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


def _task_from_row(operation, case: str, row: pd.Series) -> dict:
    month = str(row["month"])
    controller = str(row["controller_name"])
    match = re.fullmatch(r"h(\d+)_t1_(\d+)_t2_(\d+)", str(row["combo"]))
    if match is None:
        raise ValueError(f"Invalid mesh tag: {row['combo']}")
    mesh = dict(zip(("h", "t1", "t2"), map(int, match.groups()), strict=True))
    seed = int(row["outage_seed"])
    out_dir = (
        OPERATION_ROOT
        / case
        / "02-seasonal-mesh"
        / month
        / controller
        / str(row["combo"])
    )
    return operation._task(
        "seasonal",
        controller,
        mesh,
        f"{month}-01 00:00:00",
        operation.DEFAULT_N_ITERS,
        out_dir,
        labels={"month": month, "outage_seed": seed},
        seed=seed,
        resume=True,
        parameters_json=str(SIZING_ROOT / "parameters.json"),
        sizing_artifact=str(
            SIZING_ROOT / case / "sizing_decision_variables.json"
        ),
        campaign_id=f"with-degradation-{case}",
    )


def _refresh_selection(operation, case: str) -> None:
    case_root = OPERATION_ROOT / case
    stage_root = case_root / "02-seasonal-mesh"
    summary_path = stage_root / "summary.csv"
    summary = pd.read_csv(summary_path)
    operation.write_mesh_effects(summary, stage_root / "factor_effects.csv")
    champion = operation.select_meshes(
        summary_path,
        stage_root / "champion_mesh.json",
        1,
        expected_runs_per_controller=len(operation.MONTHLY_VALIDATION_WINDOWS),
    )[0]
    payload = {
        "operation_model_version": operation.OPERATION_MODEL_VERSION,
        "mesh_selection_version": operation.MESH_SELECTION_VERSION,
        "required_controllers": list(operation.SELECTION_CONTROLLERS),
        "validation_months": list(operation.MONTHLY_VALIDATION_WINDOWS),
        **champion,
    }
    (case_root / "champion_mesh.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"[{case}] refreshed champion: {champion}", flush=True)


def main() -> None:
    operation = _load_operation_module()
    total = 0
    failures = []
    for case in TARGET_CASES:
        stage_root = OPERATION_ROOT / case / "02-seasonal-mesh"
        summary_path = stage_root / "summary.csv"
        incremental_path = stage_root / "incremental_summary.csv"
        summary = pd.read_csv(summary_path)
        failed = summary.loc[~_is_valid(summary)].copy()
        print(f"[{case}] {len(failed)} failed case(s) selected", flush=True)
        for position, (_, row) in enumerate(failed.iterrows(), start=1):
            task = _task_from_row(operation, case, row)
            print(
                f"[{case} {position}/{len(failed)}] running {task['case_id']}",
                flush=True,
            )
            try:
                result = operation.execute_task(task)
                if not bool(result.get("audit_pass", False)):
                    print(
                        f"[{case}] cached artifact failed audit; forcing fresh run",
                        flush=True,
                    )
                    task["resume"] = False
                    result = operation.execute_task(task)
                _replace_row(incremental_path, result)
                _replace_row(summary_path, result)
                if not bool(result.get("audit_pass", False)):
                    raise RuntimeError(result.get("audit_reasons", "audit failed"))
                print(f"[{case}] completed {task['case_id']}", flush=True)
                total += 1
            except Exception as error:  # noqa: BLE001 - continue independent repairs
                failures.append(f"{task['case_id']}: {error}")
                print(f"[{case}] FAILED {task['case_id']}: {error}", flush=True)
        if not failures:
            _refresh_selection(operation, case)

    print(f"Repaired {total} case(s).", flush=True)
    if failures:
        raise RuntimeError("\n".join(failures))


if __name__ == "__main__":
    main()
