"""Run the paper's PV/BESS sizing sensitivity campaign."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import pandas as pd
import pyomo.environ as pyo

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sizing import MicrogridDesign


PARAMETERS_PATH = ROOT / "data" / "parameters.json"
RESULTS_ROOT = ROOT / "outputs" / "sizing-sweep"
LOAD_TRAIN_PATH = ROOT / "data" / "load_5min_train.csv"
LOAD_TEST_PATH = ROOT / "data" / "load_5min_test.csv"
LOAD_PROTOTYPES_PATH = ROOT / "data" / "sizing" / "prototypes_load_dtw_all_train.csv"
PV_PROTOTYPES_PATH = ROOT / "data" / "sizing" / "prototypes_pv_dtw_train.csv"
JOINT_PROBABILITY_PATH = ROOT / "data" / "sizing" / "prob_joint_load_pv.csv"
GULLIVER_PATH = ROOT / "data" / "Gulliver.otf"


@dataclass(frozen=True)
class Case:
    case_id: str
    label: str
    service_fraction: float
    pv_credit_mode: str = "none"


CASES = (
    Case("economic", "Economic sizing", 0.0),
    Case("critical_50", "Critical-load resilience", 0.5),
    Case("full_100", "Full-load resilience", 1.0),
    Case("full_100_solar", "Full-load resilience with PV credit", 1.0, "daylight"),
)

DEGRADATION_SWEEP = (
    ("with-degradation", {}),
    (
        "without-degradation",
        {
            "sizing": {
                "bess_calendar_fade_per_year": 0.0,
                "bess_cyclic_fade_per_kwh": 0.0,
                "pv_degradation_year1_frac": 0.0,
                "pv_degradation_linear_frac": 0.0,
            }
        },
    ),
)
SELECTED_CASES = ("economic", "critical_50", "full_100")
TIME_LIMIT_SECONDS = 7200.0
THREADS = 8
SOLVER = "gurobi"
TEE = False
FORCE = True
DRY_RUN = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_value(component: Any) -> float | None:
    value = pyo.value(component, exception=False)
    return None if value is None else float(value)


def _load_complete_days(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["timestamp", "p_norm"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["p_norm"] = pd.to_numeric(frame["p_norm"], errors="coerce")
    frame = frame.dropna().drop_duplicates("timestamp").sort_values("timestamp")
    frame["day"] = frame["timestamp"].dt.normalize()
    counts = frame.groupby("day")["timestamp"].transform("size")
    complete = frame[counts == 288].copy()
    if complete.empty:
        raise ValueError(f"No complete 5-min days found in {path}")
    return complete


def _tail_statistics(path: Path, pmax_kw: float, outage_hours: float) -> dict[str, Any]:
    frame = _load_complete_days(path)
    window = max(1, int(round(outage_hours * 12.0)))
    night = frame[(frame["timestamp"].dt.hour < 6) | (frame["timestamp"].dt.hour >= 18)]
    peak_row = night.loc[night["p_norm"].idxmax()]
    rolling = frame.groupby("day", sort=False)["p_norm"].transform(
        lambda values: values.rolling(window, min_periods=window).sum()
    )
    energy_kwh = rolling * pmax_kw / 12.0
    energy_idx = energy_kwh.idxmax()
    energy_row = frame.loc[energy_idx]
    return {
        "source": path.as_posix(),
        "complete_days": int(frame["day"].nunique()),
        "night_peak_kw": float(peak_row["p_norm"] * pmax_kw),
        "night_peak_timestamp": peak_row["timestamp"].isoformat(),
        "night_peak_day": peak_row["day"].date().isoformat(),
        "maximum_outage_energy_kwh": float(energy_kwh.loc[energy_idx]),
        "maximum_outage_energy_end": energy_row["timestamp"].isoformat(),
        "maximum_outage_energy_day": energy_row["day"].date().isoformat(),
        "outage_duration_hours": float(outage_hours),
    }


def _aggregate_day(path: Path, day: str, slots: int) -> list[float]:
    frame = _load_complete_days(path)
    selected = frame[frame["day"] == pd.Timestamp(day)].copy()
    if selected.empty:
        raise ValueError(f"Day {day} is not available in {path}")
    minutes = selected["timestamp"].dt.hour * 60 + selected["timestamp"].dt.minute
    selected["slot"] = (minutes * slots // 1440).astype(int)
    profile = selected.groupby("slot")["p_norm"].mean().reindex(range(slots))
    if profile.isna().any():
        raise ValueError(f"Day {day} cannot be aggregated into {slots} complete slots")
    return [float(value) for value in profile]


def _build_sizing_inputs(train_tail: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    load = pd.read_csv(LOAD_PROTOTYPES_PATH)
    pv = pd.read_csv(PV_PROTOTYPES_PATH)
    joint = pd.read_csv(JOINT_PROBABILITY_PATH)
    load = load[load["split"] == "train"].copy()
    pv = pv[pv["split"] == "train"].copy()
    joint = joint[(joint["split"] == "train") & (joint["probability"] > 0.0)].copy()

    slots = int(load["slot"].nunique())
    zero_pv_cluster = int(pv["cluster"].max()) + 1
    zero_pv = pd.DataFrame(
        {"group": "PV_STRESS", "split": "train", "cluster": zero_pv_cluster, "slot": range(slots), "value": 0.0}
    )
    pv = pd.concat([pv, zero_pv], ignore_index=True)

    requested_days = [
        ("night_peak", train_tail["night_peak_day"]),
        ("maximum_outage_energy", train_tail["maximum_outage_energy_day"]),
    ]
    day_labels: dict[str, list[str]] = {}
    for label, day in requested_days:
        day_labels.setdefault(day, []).append(label)

    first_cluster = int(load["cluster"].max()) + 1
    stress_rows: list[pd.DataFrame] = []
    joint_rows: list[dict[str, Any]] = []
    stress_clusters: list[int] = []
    stress_profiles: list[dict[str, Any]] = []
    for offset, (day, labels) in enumerate(day_labels.items()):
        cluster = first_cluster + offset
        values = _aggregate_day(LOAD_TRAIN_PATH, day, slots)
        stress_rows.append(
            pd.DataFrame(
                {"group": "LOAD_STRESS", "split": "train", "cluster": cluster, "slot": range(slots), "value": values}
            )
        )
        row = {column: None for column in joint.columns}
        row.update(
            split="train",
            group_load="LOAD_STRESS",
            group_pv="PV_STRESS",
            cluster_load=cluster,
            cluster_pv=zero_pv_cluster,
            probability=0.0,
        )
        if "n_days" in row:
            row["n_days"] = 0
        if "n_total" in row:
            row["n_total"] = int(joint["n_total"].max())
        joint_rows.append(row)
        stress_clusters.append(cluster)
        stress_profiles.append({"cluster_load": cluster, "day": day, "tail_modes": labels})

    load = pd.concat([load, *stress_rows], ignore_index=True)
    joint = pd.concat([joint, pd.DataFrame(joint_rows)], ignore_index=True)
    metadata = {
        "method": "zero-probability feasibility scenarios from training data",
        "pv_assumption": "zero available PV throughout the stress day",
        "slots": slots,
        "load_clusters": stress_clusters,
        "pv_cluster": zero_pv_cluster,
        "profiles": stress_profiles,
        "objective_probability_mass": float(joint["probability"].sum()),
    }
    return pv, load, joint, metadata


def _pv_resilience_credit(pv: pd.DataFrame, outage_hours: float) -> dict[str, Any]:
    train = pv[pv["split"] == "train"].copy()
    if train.empty:
        return {"cluster": None, "peak_factor": 0.0, "energy_factor": 0.0, "window_slots": 0}

    slots = int(train["slot"].nunique())
    window = max(1, int(round(outage_hours * slots / 24.0)))
    best: dict[str, Any] | None = None
    for cluster, group in train.groupby("cluster", sort=False):
        profile = group.sort_values("slot")["value"].astype(float).reindex(range(slots))
        if profile.isna().any():
            continue
        peak_factor = float(profile.max())
        energy_factor = float(profile.rolling(window, min_periods=window).sum().max() * (24.0 / slots))
        candidate = {
            "cluster": int(cluster),
            "peak_factor": peak_factor,
            "energy_factor": energy_factor,
            "window_slots": window,
        }
        if best is None or candidate["energy_factor"] > best["energy_factor"]:
            best = candidate

    if best is None:
        return {"cluster": None, "peak_factor": 0.0, "energy_factor": 0.0, "window_slots": window}
    return best


def _effective_config(
    base: dict[str, Any], case: Case, train_tail: dict[str, Any], pv_credit: dict[str, Any]
) -> dict[str, Any]:
    config = deepcopy(base)
    campaign = config.get("resilience_sizing", {})
    sizing = config.setdefault("sizing", {})
    sizing["include_zero_prob_scenarios"] = True
    sizing["minimum_outage_load_served_fraction"] = case.service_fraction
    sizing["resilience_peak_load_kw"] = float(config["Load"]["Pmax_kw"])
    sizing["resilience_outage_energy_kwh"] = float(train_tail["maximum_outage_energy_kwh"])
    sizing["E_BESS_size_max_kwh"] = float(campaign.get("E_BESS_size_max_kwh", 25.0))
    if case.pv_credit_mode == "daylight":
        sizing["resilience_pv_peak_factor"] = float(pv_credit["peak_factor"])
        sizing["resilience_pv_energy_factor"] = float(pv_credit["energy_factor"])
        sizing["resilience_pv_credit_cluster"] = int(pv_credit["cluster"])
    return config


def _expected_day_metrics(design: MicrogridDesign, year: int) -> dict[str, float]:
    model = design.model
    if model is None:
        return {}
    totals = {
        "load_kwh": 0.0,
        "grid_import_kwh": 0.0,
        "load_shed_kwh": 0.0,
        "bess_throughput_kwh": 0.0,
        "operating_cost_usd": 0.0,
    }
    outage_shed = 0.0
    outage_probability = sum(float(pyo.value(model.pi_c[c])) for c in model.C if str(c) != "c0")
    for t in model.T:
        dt = float(pyo.value(model.dt_h[t]))
        for s in model.S:
            for c in model.C:
                probability = float(pyo.value(model.pi_s[s] * model.pi_c[c]))
                if probability <= 0.0:
                    continue
                weight = probability * dt
                load = float(pyo.value(model.P_L[t, s]))
                shed = float(pyo.value(model.P_L_shed[t, s, c, year]))
                imported = float(pyo.value(model.P_EDS_in[t, s, c, year]))
                charge = float(pyo.value(model.P_BESS_c[t, s, c, year]))
                discharge = float(pyo.value(model.P_BESS_d[t, s, c, year]))
                curt = float(pyo.value(model.P_PV_curt[t, s, c, year]))
                stage_cost = (
                    float(pyo.value(model.c_EDS[t])) * imported
                    + float(pyo.value(model.c_L_shed)) * shed
                    + float(pyo.value(model.c_PV_curt)) * curt
                    + float(pyo.value(model.c_BESS_deg)) * (charge + discharge)
                )
                totals["load_kwh"] += weight * load
                totals["grid_import_kwh"] += weight * imported
                totals["load_shed_kwh"] += weight * shed
                totals["bess_throughput_kwh"] += weight * (charge + discharge)
                totals["operating_cost_usd"] += weight * stage_cost
                if str(c) != "c0":
                    outage_shed += weight * shed
    totals["conditional_outage_shed_kwh"] = outage_shed / outage_probability if outage_probability > 0.0 else 0.0
    return totals


def _service_audit(
    design: MicrogridDesign, year: int, stress_clusters: set[int], cluster_map: dict[str, int]
) -> dict[str, Any]:
    model = design.model
    if model is None:
        return {}
    minimum_all = 1.0
    minimum_stress = 1.0
    worst: dict[str, Any] | None = None
    observations = 0
    for c in model.C:
        if str(c) == "c0":
            continue
        for t in model.W[c]:
            for s in model.S:
                load = float(pyo.value(model.P_L[t, s]))
                if load <= 1e-9:
                    continue
                shed = float(pyo.value(model.P_L_shed[t, s, c, year]))
                served = 1.0 - shed / load
                observations += 1
                minimum_all = min(minimum_all, served)
                is_stress = int(cluster_map[str(s)]) in stress_clusters
                if is_stress:
                    minimum_stress = min(minimum_stress, served)
                if worst is None or served < worst["served_fraction"]:
                    worst = {
                        "scenario": str(s),
                        "contingency": str(c),
                        "slot": int(t),
                        "load_kw": load,
                        "shed_kw": shed,
                        "served_fraction": served,
                        "stress_scenario": is_stress,
                    }
    return {
        "year": year,
        "observations": observations,
        "minimum_served_fraction": minimum_all,
        "minimum_stress_served_fraction": minimum_stress,
        "worst_observation": worst,
    }


def _tail_adequacy(
    config: dict[str, Any],
    case: Case,
    capacity_by_year: dict[int, float],
    tails: dict[str, Any],
    pv_size_kw: float,
    pv_credit: dict[str, Any],
) -> dict[str, Any]:
    crate = float(config["BESS"]["crate_per_h"])
    dod = float(config["BESS"]["DoD_frac"])
    eta_d = float(config["BESS"]["eta_d"])
    sizing = config.get("sizing", {})
    pv_year1 = float(sizing.get("pv_degradation_year1_frac", config["PV"].get("degradation_year1_frac", 0.01)))
    pv_linear = float(sizing.get("pv_degradation_linear_frac", config["PV"].get("degradation_linear_frac", 0.004)))
    pv_peak_factor = float(pv_credit.get("peak_factor", 0.0))
    pv_energy_factor = float(pv_credit.get("energy_factor", 0.0))

    def pv_retention(year: int) -> float:
        if year <= 1:
            return 1.0
        return max(0.0, (1.0 - pv_year1) * ((1.0 - pv_linear) ** (year - 1)))

    years = sorted(capacity_by_year)
    audits: dict[str, Any] = {}
    for split, tail in tails.items():
        rows = {}
        for year in (years[0], years[-1]):
            capacity = capacity_by_year[year]
            retention = pv_retention(year)
            pv_power = pv_size_kw * pv_peak_factor * retention
            pv_energy = pv_size_kw * pv_energy_factor * retention
            power_margin = crate * capacity + pv_power - case.service_fraction * tail["night_peak_kw"]
            energy_margin = (
                dod * eta_d * capacity + pv_energy - case.service_fraction * tail["maximum_outage_energy_kwh"]
            )
            rows[str(year)] = {
                "available_capacity_kwh": capacity,
                "pv_power_credit_kw": pv_power,
                "pv_energy_credit_kwh": pv_energy,
                "power_margin_kw": power_margin,
                "energy_margin_kwh": energy_margin,
                "passes": power_margin >= -1e-7 and energy_margin >= -1e-7,
            }
        audits[split] = {"tail_statistics": tail, "year_audit": rows}
    return audits


def _run_case(
    base: dict[str, Any],
    case: Case,
    pv: pd.DataFrame,
    load: pd.DataFrame,
    joint: pd.DataFrame,
    stress: dict[str, Any],
    tails: dict[str, Any],
    pv_credit: dict[str, Any],
    args: SimpleNamespace,
) -> dict[str, Any]:
    config = _effective_config(base, case, tails["train"], pv_credit)
    effective_hash = _canonical_hash({"config": config, "stress": stress, "tails": tails})
    case_dir = args.results_root / case.case_id
    result_path = case_dir / "sizing_decision_variables.json"
    if result_path.exists() and not args.force:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("effective_config_sha256") == effective_hash:
            print(f"{case.case_id}: resumed")
            return existing

    started = time.perf_counter()
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "case_id": case.case_id,
        "case_label": case.label,
        "minimum_outage_load_served_fraction": case.service_fraction,
        "effective_config_sha256": effective_hash,
        "stress_scenarios": stress,
        "tail_statistics": tails,
        "replacement_policy": "none; adequacy is imposed in every modeled year",
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "effective_parameters.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    try:
        design = MicrogridDesign(config, df_pv=pv, df_load=load, df_prob_joint=joint)
        model = design.build()
        result["model_size"] = {
            "time_periods": len(model.T),
            "representative_scenarios": len(model.S),
            "contingencies": len(model.C),
            "years": len(model.Y),
            "variables": int(model.nvariables()),
            "constraints": int(model.nconstraints()),
        }
        solved = design.optimize(
            tee=args.tee, time_limit=args.time_limit, threads=args.threads, solver_name=args.solver
        )
        status = str(solved.solver.status)
        termination = str(solved.solver.termination_condition)
        loaded = status.lower() == "ok" and termination.lower() in {"optimal", "locallyoptimal", "feasible"}
        result.update(solver_status=status, termination_condition=termination, has_loaded_solution=loaded)
        if not loaded:
            return result

        out = design.get_results()
        capacity = {int(year): float(value) for year, value in out["E_BESS_year_kwh"].items()}
        years = sorted(capacity)
        bess_size = float(out["E_hat_BESS_kwh"])
        cluster_map = out["metadata"]["cluster_load_of_s"]
        result.update(
            decision_variables={
                "P_hat_PV_kw": out["P_hat_PV_kw"],
                "E_hat_BESS_kwh": bess_size,
                "P_hat_BESS_kw": float(config["BESS"]["crate_per_h"]) * bess_size,
                "E_BESS_init_kwh": out["E_BESS_init_kwh"],
                "BESS_initial_soc_fraction": out["BESS_initial_soc_fraction"],
            },
            lifetime={
                "E_BESS_year_kwh": capacity,
                "E_BESS_init_by_year_kwh": out["E_BESS_init_by_year_kwh"],
                "last_year_retention_fraction": capacity[years[-1]] / bess_size,
            },
            costs={"CAPEX_usd": out["CAPEX"], "NPV_OPEX_usd": out["NPV_OPEX"], "Objective_usd": out["Objective"]},
            year_1_metrics=_expected_day_metrics(design, years[0]),
            last_year_metrics=_expected_day_metrics(design, years[-1]),
            service_audit={
                "year_1": _service_audit(design, years[0], set(stress["load_clusters"]), cluster_map),
                "last_year": _service_audit(design, years[-1], set(stress["load_clusters"]), cluster_map),
            },
            tail_adequacy=_tail_adequacy(config, case, capacity, tails, float(out["P_hat_PV_kw"] or 0.0), pv_credit),
            model_audit={
                "cycle_closure_max_abs_kwh": out["BESS_cycle_closure_max_abs_kwh"],
                "simultaneous_charge_discharge_max_kw": (out["BESS_simultaneous_charge_discharge_max_kw"]),
                "resilience_power_margin_by_year_kw": out["resilience_power_margin_by_year_kw"],
                "resilience_energy_margin_by_year_kwh": out["resilience_energy_margin_by_year_kwh"],
            },
            metadata=out["metadata"],
        )
        return result
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        result.update(
            solver_status="error",
            termination_condition="exception",
            has_loaded_solution=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return result
    finally:
        result["runtime_seconds"] = time.perf_counter() - started
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
    decision = result.get("decision_variables", {})
    costs = result.get("costs", {})
    metrics = result.get("year_1_metrics", {})
    audit = result.get("service_audit", {}).get("last_year", {})
    test = result.get("tail_adequacy", {}).get("test", {}).get("year_audit", {})
    test_last = test[sorted(test, key=int)[-1]] if test else {}
    return {
        "case_id": result.get("case_id"),
        "minimum_service_fraction": result.get("minimum_outage_load_served_fraction"),
        "pv_size_kw": decision.get("P_hat_PV_kw"),
        "bess_size_kwh": decision.get("E_hat_BESS_kwh"),
        "bess_power_kw": decision.get("P_hat_BESS_kw"),
        "capex_usd": costs.get("CAPEX_usd"),
        "npv_opex_usd": costs.get("NPV_OPEX_usd"),
        "objective_usd": costs.get("Objective_usd"),
        "expected_shed_kwh_day_y1": metrics.get("load_shed_kwh"),
        "conditional_outage_shed_kwh_y1": metrics.get("conditional_outage_shed_kwh"),
        "minimum_service_fraction_last_year": audit.get("minimum_served_fraction"),
        "test_power_margin_kw_last_year": test_last.get("power_margin_kw"),
        "test_energy_margin_kwh_last_year": test_last.get("energy_margin_kwh"),
        "test_tail_pass_last_year": test_last.get("passes"),
        "runtime_seconds": result.get("runtime_seconds"),
        "solver_status": result.get("solver_status"),
        "termination_condition": result.get("termination_condition"),
    }


def _configure_gulliver() -> str:
    if not GULLIVER_PATH.exists():
        raise FileNotFoundError(f"Gulliver font not found: {GULLIVER_PATH}")
    font_manager.fontManager.addfont(str(GULLIVER_PATH))
    family = font_manager.FontProperties(fname=str(GULLIVER_PATH)).get_name()
    plt.rcParams.update({"font.family": family, "font.sans-serif": [family], "pdf.fonttype": 42, "ps.fonttype": 42})
    return family


def _save_outputs(root: Path, results: list[dict[str, Any]], campaign: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    solved = [result for result in results if result.get("has_loaded_solution")]
    summary = pd.DataFrame([_summary_row(result) for result in results])
    summary.to_csv(root / "summary.csv", index=False)
    (root / "manifest.json").write_text(
        json.dumps({**campaign, "updated_at_utc": _utc_now(), "cases": [_summary_row(r) for r in results]}, indent=2),
        encoding="utf-8",
    )
    if not solved:
        return

    family = _configure_gulliver()
    report = root / "sizing_resilience_report.pdf"
    labels = [result["case_label"] for result in solved]
    with PdfPages(report, metadata={"Title": "Resilience sizing comparison", "Font": family}) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        axes[0].bar(labels, [r["decision_variables"]["P_hat_PV_kw"] for r in solved], color="#E4A11B")
        axes[0].set_ylabel("PV capacity (kW)", fontsize=9)
        axes[1].bar(labels, [r["decision_variables"]["E_hat_BESS_kwh"] for r in solved], color="#3F7CAC")
        axes[1].set_ylabel("BESS capacity (kWh)", fontsize=9)
        for ax in axes:
            ax.tick_params(axis="x", labelrotation=18, labelsize=8)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle("Sizing decisions under alternative resilience requirements", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        for result in solved:
            capacity = result["lifetime"]["E_BESS_year_kwh"]
            years = sorted(int(year) for year in capacity)
            ax.plot(
                years,
                [capacity[str(y)] if str(y) in capacity else capacity[y] for y in years],
                label=result["case_label"],
            )
        ax.set_xlabel("Planning year", fontsize=9)
        ax.set_ylabel("Available BESS capacity (kWh)", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _run_campaign(args: SimpleNamespace) -> int:
    args.parameters = args.parameters.resolve()
    args.results_root = args.results_root.resolve()
    base = json.loads(args.parameters.read_text(encoding="utf-8"))
    outage_hours = float(base["EDS"]["outage_duration_hours"])
    pmax_kw = float(base["Load"]["Pmax_kw"])
    tails = {
        "train": _tail_statistics(LOAD_TRAIN_PATH, pmax_kw, outage_hours),
        "test": _tail_statistics(LOAD_TEST_PATH, pmax_kw, outage_hours),
    }
    pv, load, joint, stress = _build_sizing_inputs(tails["train"])
    pv_credit = _pv_resilience_credit(pv, outage_hours)
    selected = [case for case in CASES if not args.case or case.case_id in args.case]
    campaign = {
        "schema_version": 1,
        "parameters": args.parameters.as_posix(),
        "parameters_sha256": hashlib.sha256(args.parameters.read_bytes()).hexdigest(),
        "policy": "minimum pointwise served-load fraction during every modeled outage",
        "stress_scenarios": stress,
        "tail_statistics": tails,
        "selected_cases": [case.__dict__ for case in selected],
    }
    if args.dry_run:
        print(json.dumps(campaign, indent=2, ensure_ascii=False))
        return 0

    args.results_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, case in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {case.case_id}: solving", flush=True)
        result = _run_case(base, case, pv, load, joint, stress, tails, pv_credit, args)
        results.append(result)
        _save_outputs(args.results_root, results, campaign)
        print(
            f"[{index}/{len(selected)}] {case.case_id}: "
            f"{result.get('solver_status')}/{result.get('termination_condition')} "
            f"({result.get('runtime_seconds', 0.0):.1f} s)",
            flush=True,
        )
    _save_outputs(args.results_root, results, campaign)
    print((args.results_root / "summary.csv").as_posix())
    print((args.results_root / "sizing_resilience_report.pdf").as_posix())
    return 0


def _write_parameters(overrides: dict[str, Any], destination: Path) -> None:
    parameters = json.loads(PARAMETERS_PATH.read_text(encoding="utf-8"))
    for section, values in overrides.items():
        parameters.setdefault(section, {}).update(values)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(parameters, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    for degradation_name, overrides in DEGRADATION_SWEEP:
        degradation_root = RESULTS_ROOT / degradation_name
        parameters = degradation_root / "parameters.json"
        _write_parameters(overrides, parameters)
        print(f"\n=== {degradation_name} ===", flush=True)
        args = SimpleNamespace(
            parameters=parameters,
            results_root=degradation_root,
            case=list(SELECTED_CASES),
            time_limit=TIME_LIMIT_SECONDS,
            threads=THREADS,
            solver=SOLVER,
            tee=TEE,
            force=FORCE,
            dry_run=DRY_RUN,
        )
        status = _run_campaign(args)
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
