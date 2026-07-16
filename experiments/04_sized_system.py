"""Controller comparison on the system sized by 01_sizing.py.

Replaces the catalog PV/BESS with the optimal capacities from
Results/sizing/<case>/ and repeats the comparison. Outputs go to
Results/4-sized_system/<case>/.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting import PerfectForecast, PrototypeForecast
from opt import apply_sizing_case, simulate_mpc, simulate_stochastic
from opt.utils import load_series_scaled

START_TS = pd.Timestamp("2009-05-01 00:00:00")
N_ITERS = 2880
SIZING_CASE = "alpha_gt_0"  # "alpha_eq_0" = no-degradation counterfactual
OUT_ROOT = f"Results/4-sized_system/{SIZING_CASE}"
PARAMS_JSON = "data/parameters.json"

if __name__ == "__main__":
    params_catalog = json.load(open(PARAMS_JSON, encoding="utf-8"))

    params = apply_sizing_case(params_catalog, SIZING_CASE)
    info = params["sizing_case_applied"]
    print(
        f"sized system '{SIZING_CASE}': "
        f"PV={info['P_hat_PV_kw']:.3f} kW | BESS={info['E_hat_BESS_kwh']:.3f} kWh "
        f"(Pmax={info['BESS_Pmax_kw']:.3f} kW)"
    )

    # Forecaster scaling follows the sized Pmax values.
    scaling = {"P_L_nom_kw": params["Load"]["Pmax_kw"], "P_PV_nom_kw": params["PV"]["Pmax_kw"]}
    load_s, pv_s = load_series_scaled(scaling, "data/load_5min_test.csv", "data/pv_5min_test.csv")
    pv_kw = float(params["PV"]["Pmax_kw"])
    load_kw = float(params["Load"]["Pmax_kw"])

    results = []

    print("=== ideal ===")
    results.append(simulate_mpc(
        params, PerfectForecast(load_s, pv_s),
        START_TS, N_ITERS, f"{OUT_ROOT}/ideal", forecaster_name="ideal",
    ))

    print("=== stochastic ===")
    results.append(simulate_stochastic(params, START_TS, N_ITERS, f"{OUT_ROOT}/stochastic"))

    print("=== prototype-prefix ===")
    results.append(simulate_mpc(
        params, PrototypeForecast(None, load_s, pv_s, pv_kw, load_kw, strategy="prefix"),
        START_TS, N_ITERS, f"{OUT_ROOT}/prototype_prefix", forecaster_name="prototype-prefix",
    ))

    summary = pd.DataFrame(results)
    summary.to_csv(f"{OUT_ROOT}/summary.csv", index=False)
    cols = [c for c in ("controller", "forecaster", "operation_total_cost", "n_solve_fail") if c in summary.columns]
    print("\n", summary[cols].to_string(index=False))
    print(f"\nsummary: {OUT_ROOT}/summary.csv")
