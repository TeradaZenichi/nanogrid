"""Run the 25-year PV/BESS sizing cases and generate their report."""

import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sizing import MicrogridDesign

SOLVER_TIME_LIMIT_S = 3600
SOLVER_THREADS = 8
ROOT = Path(__file__).resolve().parents[1]
GULLIVER_FONT_PATH = ROOT / "data" / "Gulliver.otf"


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


GULLIVER_FONT_FAMILY = _configure_gulliver()


def _to_year_map(year_data):
    if not year_data:
        return {}
    return {int(k): float(v) for k, v in year_data.items() if v is not None}


def _extract_operation_rows(
    design: MicrogridDesign, yearly_capacity: dict[int, float]
) -> list[dict]:
    m = design.model
    if m is None:
        return []
    meta = design.get_results().get("metadata", {})
    cl_map = meta.get("cluster_load_of_s", {})
    cp_map = meta.get("cluster_pv_of_s", {})

    def _safe(v):
        val = pyo.value(v, exception=False)
        return None if val is None else float(val)

    rows = []
    has_gamma = hasattr(m, "gamma_BESS_c") and hasattr(m, "gamma_BESS_d")
    for y in m.Y:
        y_int = int(y)
        for t in m.T:
            for s in m.S:
                for c in m.C:
                    rows.append(
                        {
                            "year": y_int,
                            "slot": int(t),
                            "scenario": str(s),
                            "contingency": str(c),
                            "cluster_load": int(cl_map.get(str(s), -1)),
                            "cluster_pv": int(cp_map.get(str(s), -1)),
                            "P_L_kw": _safe(m.P_L[t, s]),
                            "P_PV_avail_kw": _safe(m.P_PV_avail[t, s, y]),
                            "P_PV_curt_kw": _safe(m.P_PV_curt[t, s, c, y]),
                            "P_BESS_c_kw": _safe(m.P_BESS_c[t, s, c, y]),
                            "P_BESS_d_kw": _safe(m.P_BESS_d[t, s, c, y]),
                            "E_BESS_kwh": _safe(m.E_BESS[t, s, c, y]),
                            "P_EDS_in_kw": _safe(m.P_EDS_in[t, s, c, y]),
                            "P_EDS_out_kw": _safe(m.P_EDS_out[t, s, c, y]),
                            "P_L_shed_kw": _safe(m.P_L_shed[t, s, c, y]),
                            "gamma_BESS_c": (_safe(m.gamma_BESS_c[t, s, c, y]) if has_gamma else None),
                            "gamma_BESS_d": (_safe(m.gamma_BESS_d[t, s, c, y]) if has_gamma else None),
                            "E_BESS_year_kwh": yearly_capacity.get(y_int),
                            "E_BESS_init_year_kwh": _safe(m.E_BESS_init[y]),
                            "initial_soc_fraction": (
                                _safe(m.E_BESS_init[y]) / yearly_capacity[y_int]
                                if yearly_capacity.get(y_int)
                                else None
                            ),
                        }
                    )
    return rows


def _save_figure(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to generate PDF: {pdf_path}")


def _save_plots(out_dir: Path, payload: dict, discount_rate: float) -> None:
    yearly_capacity = payload.get("bess_capacity_by_year_kwh", {}) or {}
    years = sorted(int(y) for y in yearly_capacity.keys())
    caps = [yearly_capacity.get(y, yearly_capacity.get(str(y))) for y in years]

    if years and caps:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(years, caps, marker="o", linewidth=2, label="Available capacity")
        initial_map = payload.get("bess_initial_energy_by_year_kwh", {}) or {}
        initial = [initial_map.get(y, initial_map.get(str(y))) for y in years]
        if initial and all(value is not None for value in initial):
            ax.plot(years, initial, marker="s", linewidth=1.8, label="Cyclic initial energy")
            ax.legend(frameon=False)
        ax.set_title("BESS capacity and cyclic initial energy")
        ax.set_xlabel("Year")
        ax.set_ylabel("Energy (kWh)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        _save_figure(fig, out_dir, "bess_capacity_by_year")

    opex_by_year = payload.get("objective_breakdown", {}).get("OPEX_annual_by_year", {}) or {}
    opex_by_year = {int(k): float(v) for k, v in opex_by_year.items() if v is not None}
    if years and opex_by_year:
        discounted = [opex_by_year[y] / ((1.0 + discount_rate) ** y) for y in years]
    else:
        opex_annual = payload.get("objective_breakdown", {}).get("OPEX_annual")
        discounted = (
            [opex_annual / ((1.0 + discount_rate) ** y) for y in years]
            if years and opex_annual is not None else []
        )

    if discounted:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(years, discounted)
        ax.set_title("Discounted OPEX by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Discounted OPEX")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        _save_figure(fig, out_dir, "discounted_opex_by_year")


def _save_comparison_report(root: Path, comparison: dict, discount_rate: float) -> Path:
    labels = ["No degradation", "With degradation"]
    cases = [comparison["alpha_eq_0"], comparison["alpha_gt_0"]]
    if not all(case.get("has_loaded_solution", True) for case in cases):
        raise RuntimeError("Cannot generate the sizing PDF report without two loaded solutions")
    pv = [case["decision_variables"]["P_hat_PV_kw"] for case in cases]
    bess = [case["decision_variables"]["E_hat_BESS_kwh"] for case in cases]
    capex = [case["objective_breakdown"]["CAPEX"] for case in cases]
    objective = [case["objective_breakdown"]["Objective"] for case in cases]
    colors = ["#4C78A8", "#F58518"]
    report_path = root / "sizing_report.pdf"

    with PdfPages(
        report_path,
        metadata={
            "Title": "PV and BESS sizing report",
            "Author": "UNICAMP",
            "Subject": f"Sizing results rendered with {GULLIVER_FONT_FAMILY}",
        },
    ) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
        for ax, values, title, ylabel in (
            (axes[0, 0], pv, "Installed PV capacity", "kW"),
            (axes[0, 1], bess, "Installed BESS capacity", "kWh"),
            (axes[1, 0], capex, "CAPEX", "Cost"),
            (axes[1, 1], objective, "Total discounted objective", "Cost"),
        ):
            bars = ax.bar(labels, values, color=colors)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
            ax.set_ylim(0.0, 1.15 * max(values))
            ax.bar_label(bars, fmt="%.3f", padding=3)
        fig.suptitle("Degradation-aware residential energy system sizing", fontsize=16)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(root / "sizing_comparison_alpha.png", bbox_inches="tight", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
        for label, case, color in zip(labels, cases, colors):
            capacity = _to_year_map(case.get("bess_capacity_by_year_kwh", {}))
            initial = _to_year_map(case.get("bess_initial_energy_by_year_kwh", {}))
            years = sorted(capacity)
            axes[0].plot(years, [capacity[y] for y in years], marker="o", color=color, label=label)
            axes[0].plot(
                years,
                [initial[y] for y in years],
                linestyle="--",
                marker="s",
                color=color,
                alpha=0.85,
            )
            annual = _to_year_map(case["objective_breakdown"].get("OPEX_annual_by_year", {}))
            axes[1].plot(
                years,
                [annual[y] / ((1.0 + discount_rate) ** y) for y in years],
                marker="o",
                color=color,
                label=label,
            )
        axes[0].set_title("Available capacity (solid) and cyclic initial energy (dashed)")
        axes[0].set_ylabel("Energy (kWh)")
        axes[0].legend(frameon=False)
        axes[0].grid(alpha=0.25)
        axes[1].set_title("Discounted OPEX by planning year")
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("Discounted OPEX")
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(root / "sizing_trajectories.png", bbox_inches="tight", dpi=180)
        plt.close(fig)

    if not report_path.exists() or report_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to generate sizing report: {report_path}")
    return report_path


def _run_case(params: dict, case_dir: Path, degradation_on: bool):
    cfg = deepcopy(params)
    cfg.setdefault("sizing", {})
    cfg["sizing"]["npv_years"] = 25
    pv_cap = cfg["sizing"].get("P_PV_size_max_kw", None)
    pv_cap_str = "sem_teto" if pv_cap is None else str(pv_cap)

    mode_name = "com_degradacao" if degradation_on else "sem_degradacao"
    if not degradation_on:
        cfg["sizing"]["bess_calendar_fade_per_year"] = 0.0
        cfg["sizing"]["bess_cyclic_fade_per_kwh"] = 0.0
        cfg["sizing"]["pv_degradation_year1_frac"] = 0.0
        cfg["sizing"]["pv_degradation_linear_frac"] = 0.0

    case_name = case_dir.name
    print(
        f"[sizing] Executando caso '{case_name}' | "
        f"modelo=BESS Extn-LP (sem binarias) | "
        f"modo={mode_name} | "
        f"alpha_bess_cal={cfg['sizing'].get('bess_calendar_fade_per_year', 'auto')} | "
        f"alpha_bess_cyc={cfg['sizing'].get('bess_cyclic_fade_per_kwh', 'auto(cycle_life)')} | "
        f"alpha_pv_y1={cfg['sizing'].get('pv_degradation_year1_frac', 'default(0.01)')} | "
        f"alpha_pv_lin={cfg['sizing'].get('pv_degradation_linear_frac', 'default(0.004)')} | "
        f"pv_cap_kw={pv_cap_str} | "
        f"npv_years={cfg['sizing'].get('npv_years')}"
    )

    design = MicrogridDesign(cfg)
    design.build()
    m = design.model
    if m is not None:
        print(
            f"[sizing] Estrutura do modelo '{case_name}': "
            f"|T|={len(m.T)} |S|={len(m.S)} |C|={len(m.C)} |Y|={len(m.Y)}"
        )

    results = design.optimize(tee=False, time_limit=SOLVER_TIME_LIMIT_S, threads=SOLVER_THREADS)
    out = design.get_results()
    status = str(results.solver.status)
    term = str(results.solver.termination_condition)
    has_solution = status.lower() == "ok" and term.lower() in {
        "optimal", "locallyoptimal", "feasible"
    }

    yearly_capacity = _to_year_map(out.get("E_BESS_year_kwh", {}))
    yearly_values = [v for v in yearly_capacity.values() if v is not None]
    max_capacity_years = max(yearly_values) if yearly_values else None

    payload = {
        "solver_status": status,
        "termination_condition": term,
        "has_loaded_solution": bool(has_solution),
        "degradation_mode": mode_name,
        "decision_variables": {
            "P_hat_PV_kw": out.get("P_hat_PV_kw"),
            "E_hat_BESS_kwh": out.get("E_hat_BESS_kwh"),
            "E_BESS_init_kwh": out.get("E_BESS_init_kwh"),
            "BESS_initial_soc_fraction": out.get("BESS_initial_soc_fraction"),
            "E_BESS_min_life_kwh": out.get("E_BESS_min_life_kwh"),
        },
        "bess_capacity_by_year_kwh": yearly_capacity,
        "bess_initial_energy_by_year_kwh": out.get("E_BESS_init_by_year_kwh", {}),
        "bess_initial_soc_fraction_by_year": out.get("BESS_initial_soc_fraction_by_year", {}),
        "pv_retention_by_year": out.get("d_PV_y", {}),
        "E_BESS_max_over_years_kwh": max_capacity_years,
        "model_audit": {
            "cyclic_daily_soc": True,
            "optimized_cyclic_initial_soc": True,
            "cycle_closure_max_abs_kwh": out.get("BESS_cycle_closure_max_abs_kwh"),
            "simultaneous_charge_discharge_max_kw": out.get(
                "BESS_simultaneous_charge_discharge_max_kw"
            ),
            "simultaneous_charge_discharge_by_year_kw": out.get(
                "BESS_simultaneous_charge_discharge_by_year_kw", {}
            ),
            "weighted_simultaneous_overlap_by_year_kwh_day": out.get(
                "BESS_weighted_simultaneous_overlap_by_year_kwh_day", {}
            ),
        },
        "artifacts": {
            "capacity_pdf": (case_dir / "bess_capacity_by_year.pdf").as_posix(),
            "discounted_opex_pdf": (case_dir / "discounted_opex_by_year.pdf").as_posix(),
            "font": GULLIVER_FONT_FAMILY,
        },
        "objective_breakdown": {
            "CAPEX": out.get("CAPEX"),
            "OPEX_day": out.get("OPEX_day"),
            "OPEX_annual": out.get("OPEX_annual"),
            "OPEX_day_by_year": out.get("OPEX_day_by_year"),
            "OPEX_annual_by_year": out.get("OPEX_annual_by_year"),
            "NPV_OPEX": out.get("NPV_OPEX"),
            "Objective": out.get("Objective"),
        },
        "metadata": out.get("metadata", {}),
    }

    case_dir.mkdir(parents=True, exist_ok=True)
    decision_path = case_dir / "sizing_decision_variables.json"
    decision_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    discount_rate = float(cfg.get("sizing", {}).get("discount_rate", 0.08))
    _save_plots(case_dir, payload, discount_rate=discount_rate)

    rows = _extract_operation_rows(design, yearly_capacity) if has_solution else []
    op_dir = case_dir / "operations"
    op_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(
            op_dir / "operations_all_years.parquet",
            engine="pyarrow", compression="zstd", index=False,
        )
    elif not has_solution:
        (op_dir / "README.txt").write_text(
            (
                "No operation CSV exported because solver did not return a loaded optimal/feasible solution.\n"
                f"solver_status={status}\n"
                f"termination_condition={term}\n"
            ),
            encoding="utf-8",
        )

    print(
        f"[sizing] Caso '{case_name}' concluido | "
        f"status={payload['solver_status']} term={payload['termination_condition']}"
    )

    return payload


def main() -> None:
    params = json.loads(Path("data/parameters.json").read_text(encoding="utf-8"))
    root = Path("outputs/sizing")
    root.mkdir(parents=True, exist_ok=True)
    comparison = {
        "alpha_eq_0": _run_case(params, root / "alpha_eq_0", False),
        "alpha_gt_0": _run_case(params, root / "alpha_gt_0", True),
    }
    comparison_path = root / "comparison_alpha_cases.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    discount_rate = float(params.get("sizing", {}).get("discount_rate", 0.08))
    report_path = _save_comparison_report(root, comparison, discount_rate)
    manifest = {
        "report_pdf": report_path.as_posix(),
        "font_family": GULLIVER_FONT_FAMILY,
        "font_file": GULLIVER_FONT_PATH.as_posix(),
        "model_version": 2,
    }
    (root / "sizing_report_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(comparison_path.as_posix())
    print(report_path.as_posix())


if __name__ == "__main__":
    main()
