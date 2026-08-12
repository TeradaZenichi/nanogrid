"""One-at-a-time sensitivity analysis for degradation-aware PV/BESS sizing.

The script is intentionally lighter than ``01_sizing.py``: it does not export
the complete operation table for every planning year.  Each case stores the
sizing decisions, annual trajectories, expected energy flows, costs, and model
audits required to choose a defensible sizing before running the operational
experiments.

Examples (run from the repository root):

    python experiments/01_1_sizing.py --list-cases
    python experiments/01_1_sizing.py --dry-run
    python experiments/01_1_sizing.py                  # priority factors
    python experiments/01_1_sizing.py --group extended
    python experiments/01_1_sizing.py --all
    python experiments/01_1_sizing.py --case baseline --case c_shed__25

Existing successful cases are resumed when their effective configuration has
not changed.  Use ``--force`` to solve them again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import pandas as pd
import pyomo.environ as pyo

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sizing import MicrogridDesign


DEFAULT_RESULTS_ROOT = ROOT / "Results" / "sizing-sensitivity"
DEFAULT_PARAMETERS = ROOT / "data" / "parameters.json"
GULLIVER_FONT_PATH = ROOT / "data" / "Gulliver.otf"


@dataclass(frozen=True)
class Factor:
    key: str
    label: str
    path: tuple[str, ...] | None
    unit: str
    levels: tuple[float, ...]
    group: str


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    factor_key: str | None
    factor_label: str
    factor_value: float | None
    factor_unit: str


FACTORS: tuple[Factor, ...] = (
    Factor(
        "c_shed",
        "Value of lost load",
        ("costs", "c_shed_per_kwh"),
        "USD/kWh",
        (2.0, 5.0, 10.0, 25.0),
        "priority",
    ),
    Factor(
        "grid_import",
        "Grid import limit",
        ("EDS", "Pmax"),
        "kW",
        (2.0, 3.5, 5.0),
        "priority",
    ),
    Factor(
        "outage_probability",
        "Daily outage probability",
        ("EDS", "outage_probability_pct"),
        "%",
        (2.0, 5.0, 10.0),
        "priority",
    ),
    Factor(
        "outage_duration",
        "Outage duration",
        ("EDS", "outage_duration_hours"),
        "h",
        (1.0, 2.0, 4.0),
        "priority",
    ),
    Factor(
        "bess_capex",
        "BESS CAPEX",
        ("BESS", "capex_per_kwh"),
        "USD/kWh",
        (750.0, 1043.0, 1300.0),
        "extended",
    ),
    Factor(
        "pv_capex",
        "PV CAPEX",
        ("PV", "capex_per_kw"),
        "USD/kW",
        (2200.0, 2740.0, 3300.0),
        "extended",
    ),
    Factor(
        "discount_rate",
        "Discount rate",
        ("sizing", "discount_rate"),
        "fraction",
        (0.05, 0.08, 0.12),
        "extended",
    ),
    Factor(
        "cycle_life",
        "BESS cycle life",
        None,
        "full cycles",
        (3000.0, 5475.0, 8000.0),
        "extended",
    ),
    Factor(
        "calendar_fade",
        "BESS calendar fade",
        ("sizing", "bess_calendar_fade_per_year"),
        "fraction/year",
        (0.0, 0.005, 0.01),
        "extended",
    ),
)

FACTOR_BY_KEY = {factor.key: factor for factor in FACTORS}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _get_nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = data
    for key in path:
        value = value[key]
    return value


def _set_nested(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = data
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value


def _number_slug(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-12):
        return str(int(round(value)))
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def _base_factor_value(params: dict[str, Any], factor: Factor) -> float:
    if factor.key == "cycle_life":
        return float(params["BESS"]["cycle_life_full"])
    if factor.path is None:
        raise ValueError(f"No path or coupled update defined for factor {factor.key}")
    return float(_get_nested(params, factor.path))


def _apply_factor(params: dict[str, Any], factor: Factor, value: float) -> dict[str, Any]:
    cfg = deepcopy(params)
    if factor.key != "cycle_life":
        if factor.path is None:
            raise ValueError(f"Missing parameter path for {factor.key}")
        _set_nested(cfg, factor.path, float(value))
        return cfg

    # Cycle life changes both physical fade and the marginal throughput cost.
    # Keeping the two derived quantities coupled avoids an inconsistent case.
    cycle_life = float(value)
    if cycle_life <= 0.0:
        raise ValueError("BESS cycle life must be positive")
    cfg.setdefault("BESS", {})["cycle_life_full"] = cycle_life
    sizing = cfg.setdefault("sizing", {})
    costs = cfg.setdefault("costs", {})
    eol_fraction = float(sizing.get("bess_eol_capacity_frac", 0.8))
    replacement_cost = float(cfg["BESS"].get("replacement_cost_per_kwh", 0.0))
    sizing["bess_cyclic_fade_per_kwh"] = (1.0 - eol_fraction) / (2.0 * cycle_life)
    costs["bess_degradation_per_kwh"] = replacement_cost / (2.0 * cycle_life)
    return cfg


def _build_case_catalog(params: dict[str, Any]) -> list[CaseSpec]:
    cases = [CaseSpec("baseline", None, "Baseline", None, "")]
    for factor in FACTORS:
        base_value = _base_factor_value(params, factor)
        for value in factor.levels:
            if math.isclose(value, base_value, rel_tol=1e-10, abs_tol=1e-12):
                continue
            cases.append(
                CaseSpec(
                    case_id=f"{factor.key}__{_number_slug(value)}",
                    factor_key=factor.key,
                    factor_label=factor.label,
                    factor_value=float(value),
                    factor_unit=factor.unit,
                )
            )
    return cases


def _select_cases(args: argparse.Namespace, catalog: list[CaseSpec]) -> list[CaseSpec]:
    by_id = {case.case_id: case for case in catalog}
    if args.case:
        unknown = sorted(set(args.case) - set(by_id))
        if unknown:
            raise SystemExit(f"Unknown case(s): {', '.join(unknown)}. Use --list-cases.")
        return [by_id[case_id] for case_id in args.case]

    groups = {"priority", "extended"} if args.all else set(args.group or ["priority"])
    selected = [by_id["baseline"]]
    selected.extend(
        case
        for case in catalog
        if case.factor_key is not None and FACTOR_BY_KEY[case.factor_key].group in groups
    )
    return selected


def _configure_gulliver() -> str:
    if not GULLIVER_FONT_PATH.exists():
        raise FileNotFoundError(f"Gulliver font not found: {GULLIVER_FONT_PATH}")
    font_manager.fontManager.addfont(str(GULLIVER_FONT_PATH))
    family = font_manager.FontProperties(fname=str(GULLIVER_FONT_PATH)).get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.sans-serif": [family],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return family


def _safe_value(component: Any) -> float | None:
    value = pyo.value(component, exception=False)
    return None if value is None else float(value)


def _expected_day_metrics(design: MicrogridDesign, year: int) -> dict[str, float]:
    m = design.model
    if m is None:
        return {}
    if year not in m.Y:
        raise KeyError(f"Year {year} is not present in the sizing model")

    totals = {
        "load_kwh_day": 0.0,
        "pv_available_kwh_day": 0.0,
        "pv_curtailment_kwh_day": 0.0,
        "grid_import_kwh_day": 0.0,
        "grid_export_kwh_day": 0.0,
        "load_shed_kwh_day": 0.0,
        "bess_charge_kwh_day": 0.0,
        "bess_discharge_kwh_day": 0.0,
        "grid_cost_usd_day": 0.0,
        "load_shed_cost_usd_day": 0.0,
        "pv_curtailment_cost_usd_day": 0.0,
        "bess_throughput_cost_usd_day": 0.0,
    }
    outage_shed_weighted = 0.0
    outage_probability = sum(float(pyo.value(m.pi_c[c])) for c in m.C if str(c) != "c0")

    for t in m.T:
        dt = float(pyo.value(m.dt_h[t]))
        tariff = float(pyo.value(m.c_EDS[t]))
        for s in m.S:
            for c in m.C:
                probability = float(pyo.value(m.pi_s[s] * m.pi_c[c]))
                weight = probability * dt
                load = float(pyo.value(m.P_L[t, s]))
                pv = float(pyo.value(m.P_PV_avail[t, s, year]))
                curt = float(pyo.value(m.P_PV_curt[t, s, c, year]))
                shed = float(pyo.value(m.P_L_shed[t, s, c, year]))
                charge = float(pyo.value(m.P_BESS_c[t, s, c, year]))
                discharge = float(pyo.value(m.P_BESS_d[t, s, c, year]))
                imported = float(pyo.value(m.P_EDS_in[t, s, c, year]))
                exported = float(pyo.value(m.P_EDS_out[t, s, c, year]))

                totals["load_kwh_day"] += weight * load
                totals["pv_available_kwh_day"] += weight * pv
                totals["pv_curtailment_kwh_day"] += weight * curt
                totals["grid_import_kwh_day"] += weight * imported
                totals["grid_export_kwh_day"] += weight * exported
                totals["load_shed_kwh_day"] += weight * shed
                totals["bess_charge_kwh_day"] += weight * charge
                totals["bess_discharge_kwh_day"] += weight * discharge
                totals["grid_cost_usd_day"] += weight * tariff * imported
                totals["load_shed_cost_usd_day"] += weight * float(pyo.value(m.c_L_shed)) * shed
                totals["pv_curtailment_cost_usd_day"] += weight * float(pyo.value(m.c_PV_curt)) * curt
                totals["bess_throughput_cost_usd_day"] += (
                    weight * float(pyo.value(m.c_BESS_deg)) * (charge + discharge)
                )
                if str(c) != "c0":
                    outage_shed_weighted += weight * shed

    totals["pv_used_kwh_day"] = (
        totals["pv_available_kwh_day"] - totals["pv_curtailment_kwh_day"]
    )
    totals["bess_throughput_kwh_day"] = (
        totals["bess_charge_kwh_day"] + totals["bess_discharge_kwh_day"]
    )
    totals["load_shed_conditional_outage_kwh_day"] = (
        outage_shed_weighted / outage_probability if outage_probability > 0.0 else 0.0
    )
    totals["stage_cost_usd_day"] = sum(
        totals[key]
        for key in (
            "grid_cost_usd_day",
            "load_shed_cost_usd_day",
            "pv_curtailment_cost_usd_day",
            "bess_throughput_cost_usd_day",
        )
    )
    return totals


def _input_snapshot(cfg: dict[str, Any]) -> dict[str, float]:
    return {
        "c_shed_usd_per_kwh": float(cfg["costs"]["c_shed_per_kwh"]),
        "grid_import_limit_kw": float(cfg["EDS"]["Pmax"]),
        "outage_probability_pct": float(cfg["EDS"]["outage_probability_pct"]),
        "outage_duration_h": float(cfg["EDS"]["outage_duration_hours"]),
        "bess_capex_usd_per_kwh": float(cfg["BESS"]["capex_per_kwh"]),
        "pv_capex_usd_per_kw": float(cfg["PV"]["capex_per_kw"]),
        "discount_rate": float(cfg["sizing"]["discount_rate"]),
        "cycle_life_full": float(cfg["BESS"]["cycle_life_full"]),
        "calendar_fade_per_year": float(cfg["sizing"].get("bess_calendar_fade_per_year", 0.0)),
        "cyclic_fade_per_kwh": float(cfg["sizing"].get("bess_cyclic_fade_per_kwh", 0.0)),
        "bess_degradation_usd_per_kwh": float(cfg["costs"]["bess_degradation_per_kwh"]),
    }


def _run_case(
    base_params: dict[str, Any],
    base_hash: str,
    case: CaseSpec,
    case_dir: Path,
    time_limit: float,
    threads: int,
    solver: str | None,
    tee: bool,
) -> dict[str, Any]:
    factor = FACTOR_BY_KEY.get(case.factor_key) if case.factor_key else None
    cfg = (
        _apply_factor(base_params, factor, float(case.factor_value))
        if factor is not None and case.factor_value is not None
        else deepcopy(base_params)
    )
    config_hash = _canonical_hash(cfg)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "case_id": case.case_id,
        "factor_key": case.factor_key,
        "factor_label": case.factor_label,
        "factor_value": case.factor_value,
        "factor_unit": case.factor_unit,
        "base_config_sha256": base_hash,
        "effective_config_sha256": config_hash,
        "inputs": _input_snapshot(cfg),
    }

    try:
        design = MicrogridDesign(cfg)
        design.build()
        if design.model is None:
            raise RuntimeError("Sizing model was not built")
        m = design.model
        result["model_size"] = {
            "time_periods": len(m.T),
            "representative_scenarios": len(m.S),
            "contingencies": len(m.C),
            "years": len(m.Y),
        }
        solved = design.optimize(
            tee=tee,
            time_limit=time_limit,
            threads=threads,
            solver_name=solver,
        )
        status = str(solved.solver.status)
        termination = str(solved.solver.termination_condition)
        has_solution = status.lower() == "ok" and termination.lower() in {
            "optimal",
            "locallyoptimal",
            "feasible",
        }
        result.update(
            {
                "solver_status": status,
                "termination_condition": termination,
                "has_loaded_solution": has_solution,
            }
        )
        if not has_solution:
            return result

        out = design.get_results()
        capacity = {int(k): v for k, v in (out.get("E_BESS_year_kwh") or {}).items()}
        initial = {int(k): v for k, v in (out.get("E_BESS_init_by_year_kwh") or {}).items()}
        years = sorted(capacity)
        first_year = years[0]
        last_year = years[-1]
        bess_size = float(out["E_hat_BESS_kwh"])
        crate = float(cfg["BESS"]["crate_per_h"])
        days_per_year = float(cfg["sizing"]["days_per_year"])
        y1_metrics = _expected_day_metrics(design, first_year)
        ylast_metrics = _expected_day_metrics(design, last_year)

        result.update(
            {
                "decision_variables": {
                    "pv_size_kw": out.get("P_hat_PV_kw"),
                    "bess_size_kwh": bess_size,
                    "bess_power_limit_kw": crate * bess_size,
                    "bess_initial_energy_kwh": out.get("E_BESS_init_kwh"),
                    "bess_initial_soc_fraction": out.get("BESS_initial_soc_fraction"),
                },
                "lifetime": {
                    "first_year": first_year,
                    "last_year": last_year,
                    "bess_capacity_by_year_kwh": capacity,
                    "bess_initial_energy_by_year_kwh": initial,
                    "bess_last_year_retention_fraction": (
                        float(capacity[last_year]) / bess_size if bess_size > 0.0 else None
                    ),
                    "pv_retention_by_year": out.get("d_PV_y", {}),
                },
                "costs": {
                    "capex_usd": out.get("CAPEX"),
                    "npv_opex_usd": out.get("NPV_OPEX"),
                    "objective_usd": out.get("Objective"),
                    "opex_day_by_year_usd": out.get("OPEX_day_by_year", {}),
                },
                "year_1_metrics": y1_metrics,
                "last_year_metrics": ylast_metrics,
                "expected_load_shed_kwh_year_1": (
                    y1_metrics["load_shed_kwh_day"] * days_per_year
                ),
                "model_audit": {
                    "cycle_closure_max_abs_kwh": out.get("BESS_cycle_closure_max_abs_kwh"),
                    "simultaneous_charge_discharge_max_kw": out.get(
                        "BESS_simultaneous_charge_discharge_max_kw"
                    ),
                },
            }
        )
        return result
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # preserve a failed case and continue the sweep
        result.update(
            {
                "solver_status": "error",
                "termination_condition": "exception",
                "has_loaded_solution": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        return result
    finally:
        result["runtime_seconds"] = time.perf_counter() - started
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _load_resumable_result(case_dir: Path, effective_hash: str) -> dict[str, Any] | None:
    path = case_dir / "result.json"
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if result.get("effective_config_sha256") != effective_hash:
        return None
    if not result.get("has_loaded_solution", False):
        return None
    return result


def _flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    decision = result.get("decision_variables", {})
    costs = result.get("costs", {})
    y1 = result.get("year_1_metrics", {})
    ylast = result.get("last_year_metrics", {})
    lifetime = result.get("lifetime", {})
    audit = result.get("model_audit", {})
    row = {
        "case_id": result.get("case_id"),
        "factor_key": result.get("factor_key"),
        "factor_label": result.get("factor_label"),
        "factor_value": result.get("factor_value"),
        "factor_unit": result.get("factor_unit"),
        "solver_status": result.get("solver_status"),
        "termination_condition": result.get("termination_condition"),
        "has_loaded_solution": result.get("has_loaded_solution"),
        "runtime_seconds": result.get("runtime_seconds"),
        "pv_size_kw": decision.get("pv_size_kw"),
        "bess_size_kwh": decision.get("bess_size_kwh"),
        "bess_power_limit_kw": decision.get("bess_power_limit_kw"),
        "bess_initial_energy_kwh": decision.get("bess_initial_energy_kwh"),
        "bess_initial_soc_fraction": decision.get("bess_initial_soc_fraction"),
        "bess_last_year_retention_fraction": lifetime.get(
            "bess_last_year_retention_fraction"
        ),
        "capex_usd": costs.get("capex_usd"),
        "npv_opex_usd": costs.get("npv_opex_usd"),
        "objective_usd": costs.get("objective_usd"),
        "load_shed_kwh_day_y1": y1.get("load_shed_kwh_day"),
        "load_shed_kwh_year_y1": result.get("expected_load_shed_kwh_year_1"),
        "load_shed_conditional_outage_kwh_day_y1": y1.get(
            "load_shed_conditional_outage_kwh_day"
        ),
        "grid_import_kwh_day_y1": y1.get("grid_import_kwh_day"),
        "bess_throughput_kwh_day_y1": y1.get("bess_throughput_kwh_day"),
        "load_shed_kwh_day_last": ylast.get("load_shed_kwh_day"),
        "cycle_closure_max_abs_kwh": audit.get("cycle_closure_max_abs_kwh"),
        "simultaneous_charge_discharge_max_kw": audit.get(
            "simultaneous_charge_discharge_max_kw"
        ),
        "error_type": result.get("error_type"),
        "error_message": result.get("error_message"),
    }
    row.update(result.get("inputs", {}))
    return row


def _collect_current_results(results_root: Path, base_hash: str) -> list[dict[str, Any]]:
    results = []
    for path in sorted(results_root.glob("*/result.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("base_config_sha256") == base_hash:
            results.append(result)
    return results


def _save_report(
    results_root: Path,
    rows: list[dict[str, Any]],
    params: dict[str, Any],
    font_family: str,
) -> Path | None:
    successful = [row for row in rows if row.get("has_loaded_solution")]
    if not successful:
        return None
    by_id = {row["case_id"]: row for row in successful}
    baseline = by_id.get("baseline")
    report_path = results_root / "sizing_sensitivity_report.pdf"

    with PdfPages(
        report_path,
        metadata={
            "Title": "Sizing sensitivity analysis",
            "Author": "UNICAMP",
            "Subject": f"One-at-a-time sizing sensitivity rendered with {font_family}",
        },
    ) as pdf:
        fig, ax = plt.subplots(figsize=(10.5, 7.5))
        ax.axis("off")
        ax.set_title("Degradation-aware sizing sensitivity", fontsize=17, pad=18)
        summary = [
            ["Completed cases", str(len(successful))],
            ["Method", "One factor at a time"],
            ["Planning horizon", f"{params['sizing']['npv_years']} years"],
            ["Font", font_family],
        ]
        if baseline:
            summary.extend(
                [
                    ["Baseline PV", f"{baseline['pv_size_kw']:.4f} kW"],
                    ["Baseline BESS", f"{baseline['bess_size_kwh']:.4f} kWh"],
                    ["Baseline expected shed", f"{baseline['load_shed_kwh_year_y1']:.3f} kWh/year"],
                    ["Baseline objective", f"USD {baseline['objective_usd']:,.2f}"],
                ]
            )
        table = ax.table(cellText=summary, colLabels=["Item", "Value"], loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 1.55)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        metrics = (
            ("pv_size_kw", "PV capacity", "kW"),
            ("bess_size_kwh", "BESS capacity", "kWh"),
            ("load_shed_kwh_year_y1", "Expected unserved energy", "kWh/year"),
            ("objective_usd", "Discounted objective", "USD"),
        )
        for factor in FACTORS:
            factor_rows = [row for row in successful if row.get("factor_key") == factor.key]
            if baseline is None or not factor_rows:
                continue
            baseline_point = dict(baseline)
            baseline_point["factor_value"] = _base_factor_value(params, factor)
            plot_rows = sorted([baseline_point, *factor_rows], key=lambda row: row["factor_value"])
            x = [float(row["factor_value"]) for row in plot_rows]

            fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
            for ax, (key, title, ylabel) in zip(axes.flat, metrics):
                y = [float(row[key]) for row in plot_rows]
                ax.plot(x, y, marker="o", linewidth=2, color="#4C78A8")
                ax.scatter(
                    [_base_factor_value(params, factor)],
                    [float(baseline[key])],
                    marker="*",
                    s=150,
                    color="#F58518",
                    zorder=3,
                    label="Baseline",
                )
                ax.set_title(title)
                ax.set_xlabel(f"{factor.label} ({factor.unit})")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.25)
                ax.legend(frameon=False)
            fig.suptitle(f"Sensitivity to {factor.label}", fontsize=16)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    if not report_path.exists() or report_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to generate PDF report: {report_path}")
    return report_path


def _save_consolidated_outputs(
    results_root: Path,
    results: list[dict[str, Any]],
    params: dict[str, Any],
    base_hash: str,
    font_family: str,
) -> None:
    rows = [_flatten_result(result) for result in results]
    rows.sort(key=lambda row: (row["factor_key"] or "", row["factor_value"] or 0.0))
    results_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results_root / "sizing_sensitivity_summary.csv", index=False)
    (results_root / "sizing_sensitivity_results.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at_utc": _utc_now(),
                "base_config_sha256": base_hash,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report = _save_report(results_root, rows, params, font_family)
    manifest = {
        "generated_at_utc": _utc_now(),
        "base_config_sha256": base_hash,
        "completed_cases": sum(bool(row.get("has_loaded_solution")) for row in rows),
        "failed_cases": sum(not bool(row.get("has_loaded_solution")) for row in rows),
        "summary_csv": (results_root / "sizing_sensitivity_summary.csv").as_posix(),
        "results_json": (results_root / "sizing_sensitivity_results.json").as_posix(),
        "report_pdf": report.as_posix() if report else None,
        "font_family": font_family,
        "font_file": GULLIVER_FONT_PATH.as_posix(),
    }
    (results_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _print_cases(cases: Iterable[CaseSpec]) -> None:
    print(f"{'case_id':<34} {'factor':<24} {'value':>12}  unit")
    print("-" * 88)
    for case in cases:
        value = "-" if case.factor_value is None else f"{case.factor_value:g}"
        print(
            f"{case.case_id:<34} {(case.factor_key or 'baseline'):<24} "
            f"{value:>12}  {case.factor_unit}"
        )


def _effective_config_for_case(params: dict[str, Any], case: CaseSpec) -> dict[str, Any]:
    if case.factor_key is None:
        return deepcopy(params)
    return _apply_factor(params, FACTOR_BY_KEY[case.factor_key], float(case.factor_value))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable one-at-a-time sensitivity analysis of degradation-aware sizing."
    )
    parser.add_argument("--parameters", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--group",
        action="append",
        choices=("priority", "extended"),
        help="Factor group to run; may be repeated. Default: priority.",
    )
    parser.add_argument("--all", action="store_true", help="Run priority and extended factors.")
    parser.add_argument(
        "--case", action="append", help="Run only this case ID; may be repeated."
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print selected cases without solving.")
    parser.add_argument("--force", action="store_true", help="Re-solve matching completed cases.")
    parser.add_argument("--time-limit", type=float, default=3600.0, help="Solver limit per case (s).")
    parser.add_argument("--threads", type=int, default=8, help="Solver threads per case.")
    parser.add_argument("--solver", default=None, help="Optional explicit Pyomo solver name.")
    parser.add_argument("--tee", action="store_true", help="Show solver output.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    parameters_path = args.parameters.resolve()
    results_root = args.results_root.resolve()
    params = json.loads(parameters_path.read_text(encoding="utf-8"))
    base_hash = _canonical_hash(params)
    catalog = _build_case_catalog(params)

    if args.list_cases:
        _print_cases(catalog)
        return 0

    selected = _select_cases(args, catalog)
    if args.dry_run:
        _print_cases(selected)
        print(f"\nSelected cases: {len(selected)}")
        print(f"Results root: {results_root}")
        return 0

    font_family = _configure_gulliver()
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "case_catalog.json").write_text(
        json.dumps(
            {
                "generated_at_utc": _utc_now(),
                "parameters_file": parameters_path.as_posix(),
                "base_config_sha256": base_hash,
                "selected_cases": [case.__dict__ for case in selected],
                "factors": [factor.__dict__ for factor in FACTORS],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    for index, case in enumerate(selected, start=1):
        effective_cfg = _effective_config_for_case(params, case)
        effective_hash = _canonical_hash(effective_cfg)
        case_dir = results_root / case.case_id
        existing = None if args.force else _load_resumable_result(case_dir, effective_hash)
        if existing is not None:
            print(f"[{index}/{len(selected)}] {case.case_id}: resumed")
            continue

        print(f"[{index}/{len(selected)}] {case.case_id}: solving")
        result = _run_case(
            base_params=params,
            base_hash=base_hash,
            case=case,
            case_dir=case_dir,
            time_limit=args.time_limit,
            threads=args.threads,
            solver=args.solver,
            tee=args.tee,
        )
        print(
            f"[{index}/{len(selected)}] {case.case_id}: "
            f"{result.get('solver_status')}/{result.get('termination_condition')} "
            f"({result.get('runtime_seconds', 0.0):.1f} s)"
        )

        current_results = _collect_current_results(results_root, base_hash)
        _save_consolidated_outputs(
            results_root, current_results, params, base_hash, font_family
        )

    current_results = _collect_current_results(results_root, base_hash)
    _save_consolidated_outputs(results_root, current_results, params, base_hash, font_family)
    print((results_root / "sizing_sensitivity_summary.csv").as_posix())
    print((results_root / "sizing_sensitivity_report.pdf").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
