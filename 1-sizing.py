import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

from sizing import MicrogridDesign

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency for plotting only
    plt = None


# Solver: escolhido automaticamente (Gurobi se disponivel, senao HiGHS); LPs
# grandes como este usam barrier/IPM sem crossover — ver opt.utils.solve_model.
SOLVER_TIME_LIMIT_S = 3600
SOLVER_THREADS = 8


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
                        }
                    )
    return rows


def _save_plots(out_dir: Path, payload: dict, discount_rate: float) -> None:
    if plt is None:
        print("[sizing] matplotlib nao encontrado; pulando geracao de plots.")
        return

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

    opex_by_year = payload.get("objective_breakdown", {}).get("OPEX_annual_by_year", {}) or {}
    opex_by_year = {int(k): float(v) for k, v in opex_by_year.items() if v is not None}
    if years and opex_by_year:
        discounted = [opex_by_year[y] / ((1.0 + discount_rate) ** y) for y in years]
    else:
        opex_annual = payload.get("objective_breakdown", {}).get("OPEX_annual")
        discounted = [opex_annual / ((1.0 + discount_rate) ** y) for y in years] if (years and opex_annual is not None) else []

    if discounted:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(years, discounted)
        ax.set_title("Discounted OPEX by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Discounted OPEX")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "discounted_opex_by_year.png", dpi=180)
        plt.close(fig)


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

    results = design.optimize(
        tee=False,
        time_limit=SOLVER_TIME_LIMIT_S,
        threads=SOLVER_THREADS,
    )
    out = design.get_results()
    status = str(results.solver.status)
    term = str(results.solver.termination_condition)
    has_solution = (status.lower() == "ok") and (term.lower() in {"optimal", "locallyoptimal", "feasible"})

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
            "E_BESS_min_life_kwh": out.get("E_BESS_min_life_kwh"),
        },
        "bess_capacity_by_year_kwh": yearly_capacity,
        "pv_retention_by_year": out.get("d_PV_y", {}),
        "E_BESS_max_over_years_kwh": max_capacity_years,
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
    (case_dir / "sizing_decision_variables.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    discount_rate = float(cfg.get("sizing", {}).get("discount_rate", 0.08))
    _save_plots(case_dir, payload, discount_rate=discount_rate)

    rows = _extract_operation_rows(design, yearly_capacity) if has_solution else []
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


if __name__ == "__main__":
    with open("data/parameters.json", "r", encoding="utf-8") as f:
        params = json.load(f)

    root = Path("Results/sizing")
    root.mkdir(parents=True, exist_ok=True)

    case_alpha_0 = _run_case(
        params=params,
        case_dir=root / "alpha_eq_0",
        degradation_on=False,
    )

    case_alpha_gt = _run_case(
        params=params,
        case_dir=root / "alpha_gt_0",
        degradation_on=True,
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
