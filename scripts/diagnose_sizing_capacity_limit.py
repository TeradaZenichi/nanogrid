"""Certify whether the 6 h full-service sizing exceeds the 25 kWh design cap."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
SIZING_SCRIPT = ROOT / "1-sizing.py"
PARAMETERS = (
    ROOT
    / "outputs"
    / "paper-analysis"
    / "sizing-autonomy"
    / "with-degradation"
    / "h6"
    / "parameters.json"
)
OUTPUT = (
    ROOT
    / "outputs"
    / "paper-analysis"
    / "sizing-autonomy"
    / "capacity_limit_diagnostic.json"
)
ORIGINAL_CAP_KWH = 25.0
DIAGNOSTIC_CAP_KWH = 30.0


def _load_sizing_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sizing_capacity_diagnostic", SIZING_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {SIZING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    sizing = _load_sizing_module()
    base = json.loads(PARAMETERS.read_text(encoding="utf-8"))
    outage_hours = float(base["EDS"]["outage_duration_hours"])
    pmax_kw = float(base["Load"]["Pmax_kw"])
    tails = {
        "train": sizing._tail_statistics(sizing.LOAD_TRAIN_PATH, pmax_kw, outage_hours),
        "test": sizing._tail_statistics(sizing.LOAD_TEST_PATH, pmax_kw, outage_hours),
    }
    pv, load, joint, stress = sizing._build_sizing_inputs(tails["train"])
    pv_credit = sizing._pv_resilience_credit(pv, outage_hours)
    case = next(item for item in sizing.CASES if item.case_id == "full_100")
    config = sizing._effective_config(base, case, tails["train"], pv_credit)
    config["sizing"]["E_BESS_size_max_kwh"] = DIAGNOSTIC_CAP_KWH

    started = time.perf_counter()
    design = sizing.MicrogridDesign(config, df_pv=pv, df_load=load, df_prob_joint=joint)
    model = design.build()
    model.Objective.deactivate()
    model.MinimumInstalledBESS = pyo.Objective(expr=model.E_hat_BESS, sense=pyo.minimize)
    solved = design.optimize(
        tee=True,
        time_limit=sizing.TIME_LIMIT_SECONDS,
        threads=sizing.THREADS,
        solver_name=sizing.SOLVER,
    )
    status = str(solved.solver.status)
    termination = str(solved.solver.termination_condition)
    loaded = status.lower() == "ok" and termination.lower() in {
        "optimal",
        "locallyoptimal",
        "feasible",
    }
    minimum = pyo.value(model.E_hat_BESS, exception=False) if loaded else None
    last_year = pyo.value(model.E_BESS_year[model.Y.last()], exception=False) if loaded else None
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "minimum-capacity feasibility certificate for full_100 at 6 h",
        "objective": "minimize installed BESS energy subject to all sizing constraints",
        "degradation_variant": "with-degradation",
        "outage_duration_h": outage_hours,
        "service_fraction": case.service_fraction,
        "original_capacity_cap_kwh": ORIGINAL_CAP_KWH,
        "diagnostic_capacity_cap_kwh": DIAGNOSTIC_CAP_KWH,
        "solver_status": status,
        "termination_condition": termination,
        "has_loaded_solution": loaded,
        "minimum_feasible_bess_kwh": float(minimum) if minimum is not None else None,
        "last_year_available_bess_kwh": float(last_year) if last_year is not None else None,
        "original_cap_shortfall_kwh": (
            float(minimum) - ORIGINAL_CAP_KWH if minimum is not None else None
        ),
        "original_cap_is_infeasible": (
            bool(float(minimum) > ORIGINAL_CAP_KWH + 1e-6)
            if minimum is not None
            else None
        ),
        "model_size": {
            "variables": int(model.nvariables()),
            "constraints": int(model.nconstraints()),
        },
        "runtime_seconds": time.perf_counter() - started,
        "stress_scenarios": stress,
        "tail_statistics": tails,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not loaded:
        raise RuntimeError(f"Capacity diagnostic failed: {status}/{termination}")


if __name__ == "__main__":
    main()
