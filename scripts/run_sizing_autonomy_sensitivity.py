"""Run the paper's outage-duration sensitivity for resilient PV-BESS sizing."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIZING_SCRIPT = ROOT / "1-sizing.py"
PARAMETERS = ROOT / "data" / "parameters.json"
OUTPUT_ROOT = ROOT / "outputs" / "paper-analysis" / "sizing-autonomy"
CAPACITY_DIAGNOSTIC_SCRIPT = ROOT / "scripts" / "diagnose_sizing_capacity_limit.py"
CAPACITY_DIAGNOSTIC = OUTPUT_ROOT / "capacity_limit_diagnostic.json"
DURATIONS_H = (1.0, 2.0, 4.0, 6.0)
CASES = ("critical_50", "full_100")
VARIANTS = {
    "with-degradation": {},
    "without-degradation": {
        "sizing": {
            "bess_calendar_fade_per_year": 0.0,
            "bess_cyclic_fade_per_kwh": 0.0,
            "pv_degradation_year1_frac": 0.0,
            "pv_degradation_linear_frac": 0.0,
        }
    },
}


def _load_sizing_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sizing_autonomy", SIZING_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {SIZING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_parameters(duration_h: float, overrides: dict, destination: Path) -> None:
    params = json.loads(PARAMETERS.read_text(encoding="utf-8"))
    params.setdefault("EDS", {})["outage_duration_hours"] = duration_h
    for section, values in overrides.items():
        params.setdefault(section, {}).update(values)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _duration_label(duration_h: float) -> str:
    return f"h{duration_h:g}".replace(".", "p")


def _capacity_limit_certified() -> bool:
    if not CAPACITY_DIAGNOSTIC.exists():
        return False
    certificate = json.loads(CAPACITY_DIAGNOSTIC.read_text(encoding="utf-8"))
    return bool(
        certificate.get("has_loaded_solution")
        and certificate.get("original_cap_is_infeasible")
        and certificate.get("degradation_variant") == "with-degradation"
        and certificate.get("outage_duration_h") == 6.0
        and certificate.get("service_fraction") == 1.0
    )


def _run_capacity_diagnostic() -> None:
    spec = importlib.util.spec_from_file_location(
        "sizing_capacity_diagnostic_from_runner", CAPACITY_DIAGNOSTIC_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {CAPACITY_DIAGNOSTIC_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main()


def _collect_summary(sizing: ModuleType) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        for duration_h in DURATIONS_H:
            output = OUTPUT_ROOT / variant / _duration_label(duration_h)
            for case in CASES:
                artifact = output / case / "sizing_decision_variables.json"
                if not artifact.exists():
                    continue
                result = json.loads(artifact.read_text(encoding="utf-8"))
                row = sizing._summary_row(result)
                row = {
                    "degradation_variant": variant,
                    "outage_duration_h": duration_h,
                    **row,
                }
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    sizing = _load_sizing_module()
    total = len(VARIANTS) * len(DURATIONS_H)
    completed = 0
    for variant, overrides in VARIANTS.items():
        for duration_h in DURATIONS_H:
            completed += 1
            output = OUTPUT_ROOT / variant / _duration_label(duration_h)
            parameters = output / "parameters.json"
            _write_parameters(duration_h, overrides, parameters)
            print(
                f"\n=== autonomy [{completed}/{total}] {variant} / {duration_h:g} h ===",
                flush=True,
            )
            selected_cases = list(CASES)
            if (
                variant == "with-degradation"
                and duration_h == 6.0
                and _capacity_limit_certified()
            ):
                selected_cases = ["critical_50"]
                print("full_100: certified infeasible at the 25 kWh cap", flush=True)
            args = SimpleNamespace(
                parameters=parameters,
                results_root=output,
                case=selected_cases,
                time_limit=sizing.TIME_LIMIT_SECONDS,
                threads=sizing.THREADS,
                solver=sizing.SOLVER,
                tee=sizing.TEE,
                force=False,
                dry_run=False,
            )
            status = sizing._run_campaign(args)
            if status:
                raise RuntimeError(
                    f"Sizing campaign failed: {variant}, {duration_h:g} h"
                )
            combined = _collect_summary(sizing)
            combined.to_csv(OUTPUT_ROOT / "summary.csv", index=False)

    combined = _collect_summary(sizing)
    expected = total * len(CASES)
    valid = combined[
        combined["solver_status"].astype(str).str.lower().eq("ok")
        & combined["termination_condition"].astype(str).str.lower().isin(
            ("optimal", "locallyoptimal", "feasible")
        )
    ]
    if len(combined) != expected:
        raise RuntimeError(
            f"Autonomy sensitivity has {len(combined)}/{expected} rows"
        )
    if len(valid) == expected:
        print(f"\nAutonomy sensitivity valid for {len(valid)}/{expected} cases")
    else:
        failed = combined.loc[~combined.index.isin(valid.index)]
        expected_failure = (
            len(valid) == expected - 1
            and len(failed) == 1
            and failed.iloc[0]["degradation_variant"] == "with-degradation"
            and float(failed.iloc[0]["outage_duration_h"]) == 6.0
            and failed.iloc[0]["case_id"] == "full_100"
        )
        if expected_failure and not _capacity_limit_certified():
            print("\nRunning expanded-cap minimum-capacity certificate", flush=True)
            _run_capacity_diagnostic()
        if not expected_failure or not _capacity_limit_certified():
            raise RuntimeError(
                f"Autonomy sensitivity has {len(valid)}/{expected} valid rows"
            )
        print(
            f"\nAutonomy sensitivity: {len(valid)} optimal cases and "
            "1 certified capacity-limit infeasible case"
        )
    print((OUTPUT_ROOT / "summary.csv").as_posix())


if __name__ == "__main__":
    main()
