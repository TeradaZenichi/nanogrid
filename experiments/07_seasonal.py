"""Monthly coverage: one 10-day window per month of the test year.

Four controllers per window (ideal, stochastic, prototype-prefix, LSTM),
May 2009 through April 2010. Outputs go to Results/7-seasonal/<month>/.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting import PerfectForecast, PrototypeForecast
from opt import simulate_mpc, simulate_stochastic
from opt.utils import load_series_scaled

N_ITERS = 2880
OUT_ROOT = "Results/7-seasonal"
PARAMS_JSON = "data/parameters.json"

MONTHS = [
    "2009-05", "2009-06", "2009-07", "2009-08", "2009-09", "2009-10",
    "2009-11", "2009-12", "2010-01", "2010-02", "2010-03", "2010-04",
]

if __name__ == "__main__":
    params = json.load(open(PARAMS_JSON, encoding="utf-8"))
    scaling = {"P_L_nom_kw": params["Load"]["Pmax_kw"], "P_PV_nom_kw": params["PV"]["Pmax_kw"]}
    load_s, pv_s = load_series_scaled(scaling, "data/load_5min_test.csv", "data/pv_5min_test.csv")
    pv_kw = float(params["PV"]["Pmax_kw"])
    load_kw = float(params["Load"]["Pmax_kw"])

    proto = PrototypeForecast(None, load_s, pv_s, pv_kw, load_kw, strategy="prefix")
    oracle = PerfectForecast(load_s, pv_s)

    results = []
    lstm = None

    for month in MONTHS:
        start = pd.Timestamp(f"{month}-01 00:00:00")
        out = f"{OUT_ROOT}/{month}"
        print(f"===== {month} =====")

        print("--- ideal ---")
        m = simulate_mpc(params, oracle, start, N_ITERS, f"{out}/ideal", forecaster_name="ideal")
        results.append({**m, "month": month})

        print("--- stochastic ---")
        m = simulate_stochastic(params, start, N_ITERS, f"{out}/stochastic")
        results.append({**m, "month": month})

        print("--- prototype-prefix ---")
        m = simulate_mpc(params, proto, start, N_ITERS, f"{out}/prototype_prefix",
                         forecaster_name="prototype-prefix")
        results.append({**m, "month": month})

        print("--- lstm ---")
        if lstm is None:
            from forecasting.get_forecasting import ForecastMPC  # deferred: pulls TensorFlow

            lstm = ForecastMPC({}, load_s, pv_s, pv_kw, load_kw)
        m = simulate_mpc(params, lstm, start, N_ITERS, f"{out}/lstm", forecaster_name="lstm")
        results.append({**m, "month": month})

    summary = pd.DataFrame(results)
    summary.to_csv(f"{OUT_ROOT}/summary.csv", index=False)
    cols = [c for c in ("month", "controller", "forecaster", "operation_total_cost") if c in summary.columns]
    print("\n", summary[cols].to_string(index=False))
    print(f"\nsummary: {OUT_ROOT}/summary.csv")
