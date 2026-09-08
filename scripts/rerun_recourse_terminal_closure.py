"""Rebuild recourse experiments with a common finite-window terminal energy."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OPERATION_SCRIPT = ROOT / "2-operation.py"
SIZING_ROOT = ROOT / "outputs" / "sizing-sweep" / "with-degradation"
OPERATION_ROOT = ROOT / "outputs" / "operation-sweep" / "with-degradation"
TARGET_CASES = ("full_100", "critical_50", "economic")


def _load_operation_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("operation_campaign", OPERATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {OPERATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(operation, case: str) -> SimpleNamespace:
    return SimpleNamespace(
        stage="recourse",
        out_root=OPERATION_ROOT / case,
        parameters=SIZING_ROOT / "parameters.json",
        sizing_artifact=(
            SIZING_ROOT / case / "sizing_decision_variables.json"
        ),
        campaign_id=f"with-degradation-{case}",
        workers=1,
        n_iters=operation.DEFAULT_N_ITERS,
        smoke_iters=operation.SMOKE_ITERS,
        top_k=operation.TOP_K,
        forecast_controllers=operation.FORECAST_CONTROLLERS,
        fresh=False,
    )


def _validate(case: str) -> None:
    path = OPERATION_ROOT / case / "04-recourse" / "summary.csv"
    summary = pd.read_csv(path)
    residual = pd.to_numeric(
        summary["evaluation_terminal_energy_residual_kwh"], errors="coerce"
    )
    valid = (
        summary["status"].astype(str).ne("error")
        & summary["audit_pass"].astype(str).str.lower().isin(("true", "1"))
        & residual.notna()
        & residual.le(1e-6)
    )
    if len(summary) != 12 or not valid.all():
        raise RuntimeError(
            f"{case}: terminal-closure audit passed for {int(valid.sum())}/{len(summary)} cases"
        )
    print(f"[{case}] terminal closure valid for 12/12 recourse cases", flush=True)


def main() -> None:
    operation = _load_operation_module()
    requested = tuple(sys.argv[1:]) or TARGET_CASES
    unknown = sorted(set(requested).difference(TARGET_CASES))
    if unknown:
        raise ValueError("Unknown sizing case(s): " + ", ".join(unknown))
    for case in requested:
        print(f"\n=== recourse terminal closure / {case} ===", flush=True)
        operation._run_pipeline(_args(operation, case))
        _validate(case)


if __name__ == "__main__":
    main()
