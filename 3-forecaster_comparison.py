# -*- coding: utf-8 -*-
"""E1 — Comparativo de controladores/forecasters em malha fechada (sistema de catalogo).

Seis estrategias sobre a MESMA janela e os MESMOS outages (EDS.seed em
data/parameters.json):

  ideal       : MPC com previsao perfeita (limite superior, ~WS)
  stochastic  : plano por cenarios resolvido uma vez (RP), malha aberta
  prototype-* : MPC com dia analogo (equivalente-certeza, ~EEV)
  lstm        : MPC com previsao LSTM
  hybrid      : LSTM nos passos finos + prototipo nos grossos

Gap ideal-real = valor da previsao (EVPI); prototype - stochastic = VSS.
"""

import json

import pandas as pd

from forecasting import HybridForecast, PerfectForecast, PrototypeForecast
from opt import simulate_mpc, simulate_stochastic
from opt.utils import load_series_scaled

START_TS = pd.Timestamp("2009-05-01 00:00:00")
N_ITERS = 2880  # 10 dias em passos de 5 min
OUT_ROOT = "Results/3-forecaster_comparison"
PARAMS_JSON = "data/parameters.json"

if __name__ == "__main__":
    params = json.load(open(PARAMS_JSON, encoding="utf-8"))

    # Series reais em kW (mesmas que o GridEnv usa) para os forecasters.
    scaling = {"P_L_nom_kw": params["Load"]["Pmax_kw"], "P_PV_nom_kw": params["PV"]["Pmax_kw"]}
    load_s, pv_s = load_series_scaled(scaling, "data/load_5min_test.csv", "data/pv_5min_test.csv")
    pv_kw = float(params["PV"]["Pmax_kw"])
    load_kw = float(params["Load"]["Pmax_kw"])

    results = []

    print("=== ideal (previsao perfeita) ===")
    results.append(simulate_mpc(
        params, PerfectForecast(load_s, pv_s),
        START_TS, N_ITERS, f"{OUT_ROOT}/ideal", forecaster_name="ideal",
    ))

    print("=== stochastic (plano por cenarios, malha aberta) ===")
    results.append(simulate_stochastic(params, START_TS, N_ITERS, f"{OUT_ROOT}/stochastic"))

    print("=== prototype-prefix ===")
    proto_prefix = PrototypeForecast(None, load_s, pv_s, pv_kw, load_kw, strategy="prefix")
    results.append(simulate_mpc(
        params, proto_prefix,
        START_TS, N_ITERS, f"{OUT_ROOT}/prototype_prefix", forecaster_name="prototype-prefix",
    ))

    print("=== prototype-calendar ===")
    results.append(simulate_mpc(
        params, PrototypeForecast(None, load_s, pv_s, pv_kw, load_kw, strategy="calendar"),
        START_TS, N_ITERS, f"{OUT_ROOT}/prototype_calendar", forecaster_name="prototype-calendar",
    ))

    print("=== lstm ===")
    from forecasting.get_forecasting import ForecastMPC  # import tardio: carrega TensorFlow

    lstm = ForecastMPC({}, load_s, pv_s, pv_kw, load_kw)
    results.append(simulate_mpc(
        params, lstm, START_TS, N_ITERS, f"{OUT_ROOT}/lstm", forecaster_name="lstm",
    ))

    print("=== hybrid (lstm fino + prototipo grosso) ===")
    results.append(simulate_mpc(
        params, HybridForecast(fine=lstm, coarse=proto_prefix),
        START_TS, N_ITERS, f"{OUT_ROOT}/hybrid", forecaster_name="hybrid",
    ))

    summary = pd.DataFrame(results)
    summary.to_csv(f"{OUT_ROOT}/summary.csv", index=False)
    cols = [c for c in ("controller", "forecaster", "operation_total_cost", "n_solve_fail") if c in summary.columns]
    print("\n", summary[cols].to_string(index=False))
    print(f"\nresumo salvo em {OUT_ROOT}/summary.csv")
