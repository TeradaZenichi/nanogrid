import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo

from sizing import MicrogridDesign


def _to_year_map(year_data):
    if not year_data:
        return {}
    return {int(k): float(v) for k, v in year_data.items() if v is not None}


def _extract_operation_rows(design: MicrogridDesign, yearly_capacity: dict[int, float]) -> list[dict]:
    m = design.model
    if m is None:
        return []
    meta = design.get_results().get("metadata", {})
    cl_map = meta.get("cluster_load_of_s", {})
    cp_map = meta.get("cluster_pv_of_s", {})

    rows = []
    years = sorted(yearly_capacity.keys())
    for y in years:
        for t in m.T:
            for s in m.S:
                for c in m.C:
                    rows.append(
                        {
                            "year": int(y),
                            "slot": int(t),
                            "scenario": str(s),
                            "contingency": str(c),
                            "cluster_load": int(cl_map.get(str(s), -1)),
                            "cluster_pv": int(cp_map.get(str(s), -1)),
                            "P_L_kw": pyo.value(m.P_L[t, s]),
                            "P_PV_avail_kw": pyo.value(m.P_PV_avail[t, s]),
                            "P_PV_curt_kw": pyo.value(m.P_PV_curt[t, s, c]),
                            "P_BESS_c_kw": pyo.value(m.P_BESS_c[t, s, c]),
                            "P_BESS_d_kw": pyo.value(m.P_BESS_d[t, s, c]),
                            "E_BESS_kwh": pyo.value(m.E_BESS[t, s, c]),
                            "P_EDS_in_kw": pyo.value(m.P_EDS_in[t, s, c]),
                            "P_EDS_out_kw": pyo.value(m.P_EDS_out[t, s, c]),
                            "P_L_shed_kw": pyo.value(m.P_L_shed[t, s, c]),
                            "gamma_BESS_c": int(round(pyo.value(m.gamma_BESS_c[t, s, c]))),
                            "gamma_BESS_d": int(round(pyo.value(m.gamma_BESS_d[t, s, c]))),
                            "E_BESS_year_kwh": yearly_capacity[y],
                        }
                    )
    return rows


def _save_plots(out_dir: Path, payload: dict, discount_rate: float) -> None:
    yearly_capacity = payload.get("bess_capacity_by_year_kwh", {}) or {}
    years = sorted(int(y) for y in yearly_capacity.keys())
    caps = [yearly_capacity.get(y, yearly_capacity.get(str(y))) for y in years]

    if years and caps:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(years, caps, marker="o", linewidth=2)
        ax.set_title("BESS Capacity Along Years")
        ax.set_xlabel("Year")
        ax.set_ylabel("Capacity (kWh)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "bess_capacity_by_year.png", dpi=180)
        plt.close(fig)

    opex_annual = payload.get("objective_breakdown", {}).get("OPEX_annual")
    if years and (opex_annual is not None):
        discounted = [opex_annual / ((1.0 + discount_rate) ** y) for y in years]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(years, discounted)
        ax.set_title("Discounted OPEX by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Discounted OPEX")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "discounted_opex_by_year.png", dpi=180)
        plt.close(fig)


def _run_case(params: dict, case_dir: Path, alpha_cal: float, alpha_cyc: float | None):
    cfg = deepcopy(params)
    cfg.setdefault("sizing", {})
    cfg["sizing"]["npv_years"] = 25
    cfg["sizing"]["bess_calendar_fade_per_year"] = float(alpha_cal)
    if alpha_cyc is None:
        cfg["sizing"].pop("bess_cyclic_fade_per_kwh", None)
    else:
        cfg["sizing"]["bess_cyclic_fade_per_kwh"] = float(alpha_cyc)

    design = MicrogridDesign(cfg)
    design.build()
    results = design.optimize(solver="gurobi", tee=False, TimeLimit=180, MIPGap=0.01)
    out = design.get_results()

    yearly_capacity = _to_year_map(out.get("E_BESS_year_kwh", {}))
    yearly_values = [v for v in yearly_capacity.values() if v is not None]
    max_capacity_years = max(yearly_values) if yearly_values else None

    payload = {
        "solver_status": str(results.solver.status),
        "termination_condition": str(results.solver.termination_condition),
        "decision_variables": {
            "P_hat_PV_kw": out.get("P_hat_PV_kw"),
            "E_hat_BESS_kwh": out.get("E_hat_BESS_kwh"),
            "E_BESS_min_life_kwh": out.get("E_BESS_min_life_kwh"),
        },
        "bess_capacity_by_year_kwh": yearly_capacity,
        "E_BESS_max_over_years_kwh": max_capacity_years,
        "objective_breakdown": {
            "CAPEX": out.get("CAPEX"),
            "OPEX_day": out.get("OPEX_day"),
            "OPEX_annual": out.get("OPEX_annual"),
            "NPV_OPEX": out.get("NPV_OPEX"),
            "Objective": out.get("Objective"),
        },
        "metadata": out.get("metadata", {}),
    }

    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "sizing_decision_variables.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    discount_rate = float(cfg.get("sizing", {}).get("discount_rate", 0.08))
    _save_plots(case_dir, payload, discount_rate=discount_rate)

    # Save operation CSVs by year, scenario, contingency dimensions.
    rows = _extract_operation_rows(design, yearly_capacity)
    op_dir = case_dir / "operations"
    op_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows)
        for year, df_y in df.groupby("year", sort=True):
            df_y.sort_values(["slot", "scenario", "contingency"]).to_csv(
                op_dir / f"operations_year_{int(year):02d}.csv",
                index=False,
            )
        df.to_csv(op_dir / "operations_all_years.csv", index=False)

    return payload


if __name__ == "__main__":
    with open("data/parameters.json", "r", encoding="utf-8") as f:
        params = json.load(f)

    root = Path("Results/sizing")
    root.mkdir(parents=True, exist_ok=True)

    # Case 1: alpha = 0 (no calendar/cyclic fade)
    case_alpha_0 = _run_case(
        params=params,
        case_dir=root / "alpha_eq_0",
        alpha_cal=0.0,
        alpha_cyc=0.0,
    )

    # Case 2: alpha > 0 (use cyclic fade from cycle_life_full, keep calendar as configured/default)
    default_alpha_cal = float(params.get("sizing", {}).get("bess_calendar_fade_per_year", 0.0))
    case_alpha_gt = _run_case(
        params=params,
        case_dir=root / "alpha_gt_0",
        alpha_cal=default_alpha_cal,
        alpha_cyc=None,
    )

    comparison = {
        "alpha_eq_0": case_alpha_0,
        "alpha_gt_0": case_alpha_gt,
    }
    (root / "comparison_alpha_cases.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print((root / "comparison_alpha_cases.json").as_posix())
