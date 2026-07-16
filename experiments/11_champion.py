"""LSTM and hybrid on the best mesh from the sweep (h=36, dt1=10, dt2=60).

The mesh sweep only covered ideal/prototype/stochastic; this closes the
table for the remaining forecasters. Outputs go to Results/11-champion/.
"""

import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting import HybridForecast, PrototypeForecast
from opt import simulate_mpc
from opt.utils import load_series_scaled

START_TS = pd.Timestamp("2009-05-01 00:00:00")
N_ITERS = 2880
OUT_ROOT = "Results/11-champion"
PARAMS_JSON = "data/parameters.json"

H, T1, T2 = 36, 10, 60

if __name__ == "__main__":
    base = json.load(open(PARAMS_JSON, encoding="utf-8"))
    params = deepcopy(base)
    params["time"].update(horizon_hours=H, timestep_1_min=T1, timestep_2_min=T2)

    scaling = {"P_L_nom_kw": base["Load"]["Pmax_kw"], "P_PV_nom_kw": base["PV"]["Pmax_kw"]}
    load_s, pv_s = load_series_scaled(scaling, "data/load_5min_test.csv", "data/pv_5min_test.csv")
    pv_kw = float(base["PV"]["Pmax_kw"])
    load_kw = float(base["Load"]["Pmax_kw"])

    from forecasting.get_forecasting import ForecastMPC  # pulls TensorFlow

    lstm = ForecastMPC({}, load_s, pv_s, pv_kw, load_kw)
    proto = PrototypeForecast(None, load_s, pv_s, pv_kw, load_kw, strategy="prefix")

    results = []
    print(f"===== lstm h{H}/{T1}/{T2} =====")
    results.append(simulate_mpc(params, lstm, START_TS, N_ITERS,
                                f"{OUT_ROOT}/lstm", forecaster_name="lstm"))
    print(f"===== hybrid h{H}/{T1}/{T2} =====")
    results.append(simulate_mpc(params, HybridForecast(fine=lstm, coarse=proto), START_TS, N_ITERS,
                                f"{OUT_ROOT}/hybrid", forecaster_name="hybrid"))

    summary = pd.DataFrame(results)
    summary.to_csv(f"{OUT_ROOT}/summary.csv", index=False)
    print(summary[["forecaster", "operation_total_cost", "avg_solve_time_s"]].round(3).to_string(index=False))
    print(f"\nsummary: {OUT_ROOT}/summary.csv")
