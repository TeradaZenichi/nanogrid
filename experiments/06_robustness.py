"""Robustness sweeps: actuator noise, outage probability, outage seeds.

Each variant is built in memory from data/parameters.json and run for the
stochastic plan and the prototype MPC on the same window. Outputs go to
Results/6-robustness/<variant>/.
"""

import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting import PrototypeForecast
from opt import simulate_mpc, simulate_stochastic
from opt.utils import load_series_scaled

START_TS = pd.Timestamp("2009-05-01 00:00:00")
N_ITERS = 2880
OUT_ROOT = "Results/6-robustness"
PARAMS_JSON = "data/parameters.json"

NOISE_LEVELS = [0.02, 0.05, 0.10]
OUTAGE_PCTS = [2, 5, 10]
SEEDS = list(range(42, 52))


def run_pair(params: dict, variant: str) -> list[dict]:
    scaling = {"P_L_nom_kw": params["Load"]["Pmax_kw"], "P_PV_nom_kw": params["PV"]["Pmax_kw"]}
    load_s, pv_s = load_series_scaled(scaling, "data/load_5min_test.csv", "data/pv_5min_test.csv")
    forecaster = PrototypeForecast(
        None, load_s, pv_s,
        float(params["PV"]["Pmax_kw"]), float(params["Load"]["Pmax_kw"]),
        strategy="prefix",
    )

    out = []
    print(f"--- {variant}: stochastic ---")
    m = simulate_stochastic(params, START_TS, N_ITERS, f"{OUT_ROOT}/{variant}/stochastic")
    m["variant"] = variant
    out.append(m)

    print(f"--- {variant}: prototype ---")
    m = simulate_mpc(params, forecaster, START_TS, N_ITERS,
                     f"{OUT_ROOT}/{variant}/prototype", forecaster_name="prototype-prefix")
    m["variant"] = variant
    out.append(m)
    return out


if __name__ == "__main__":
    base = json.load(open(PARAMS_JSON, encoding="utf-8"))
    rows = []

    for std in NOISE_LEVELS:
        params = deepcopy(base)
        params["BESS"]["noisy"] = True
        params["BESS"]["noise"]["std_frac"] = std
        rows += run_pair(params, f"noise_{std:.2f}")

    for pct in OUTAGE_PCTS:
        params = deepcopy(base)
        params["EDS"]["outage_probability_pct"] = pct
        rows += run_pair(params, f"outage_{pct}pct")

    for seed in SEEDS:
        params = deepcopy(base)
        params["EDS"]["seed"] = seed
        rows += run_pair(params, f"seed_{seed}")

    summary = pd.DataFrame(rows)
    summary.to_csv(f"{OUT_ROOT}/summary.csv", index=False)
    cols = [c for c in ("variant", "controller", "forecaster", "operation_total_cost") if c in summary.columns]
    print("\n", summary[cols].to_string(index=False))
    print(f"\nsummary: {OUT_ROOT}/summary.csv")
