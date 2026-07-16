"""Full mesh grid (h x dt1 x dt2) replicated in three more seasons.

Together with the May sweep this gives the 4-season mesh dataset. Runs
prototype and ideal in parallel workers; outputs go to
Results/7-seasonal/mesh/<season>/<controller>/<combo>/.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting import PerfectForecast, PrototypeForecast
from opt import simulate_mpc
from opt.operation import validate_time_mesh
from opt.utils import load_series_scaled

N_ITERS = 2880
WORKERS = 4
OUT_ROOT = "Results/7-seasonal/mesh"
PARAMS_JSON = "data/parameters.json"

SEASONS = {
    "2009-08": pd.Timestamp("2009-08-01 00:00:00"),
    "2009-11": pd.Timestamp("2009-11-01 00:00:00"),
    "2010-02": pd.Timestamp("2010-02-01 00:00:00"),
}
HORIZONS_H = [6, 12, 24, 36]
T1_MIN = [5, 10, 15]
T2_MIN = [30, 60, 120]
CONTROLLERS = ["prototype", "ideal"]


def run_combo(season: str, start_iso: str, controller: str, h: int, t1: int, t2: int) -> dict:
    params = json.load(open(PARAMS_JSON, encoding="utf-8"))
    params["time"].update(horizon_hours=h, timestep_1_min=t1, timestep_2_min=t2)

    scaling = {"P_L_nom_kw": params["Load"]["Pmax_kw"], "P_PV_nom_kw": params["PV"]["Pmax_kw"]}
    load_s, pv_s = load_series_scaled(scaling, "data/load_5min_test.csv", "data/pv_5min_test.csv")
    if controller == "ideal":
        forecaster = PerfectForecast(load_s, pv_s)
    else:
        forecaster = PrototypeForecast(None, load_s, pv_s,
                                       float(params["PV"]["Pmax_kw"]), float(params["Load"]["Pmax_kw"]),
                                       strategy="prefix")

    combo = f"h{h}_t1_{t1}_t2_{t2}"
    out_dir = f"{OUT_ROOT}/{season}/{controller}/{combo}"
    metrics = simulate_mpc(params, forecaster, pd.Timestamp(start_iso), N_ITERS, out_dir,
                           forecaster_name=controller)
    metrics.update(season=season, controller_name=controller, combo=combo, h=h, t1=t1, t2=t2)
    return metrics


if __name__ == "__main__":
    base = json.load(open(PARAMS_JSON, encoding="utf-8"))

    tasks = []
    for season, start in SEASONS.items():
        for h in HORIZONS_H:
            for t1 in T1_MIN:
                for t2 in T2_MIN:
                    trial = deepcopy(base)
                    trial["time"].update(horizon_hours=h, timestep_1_min=t1, timestep_2_min=t2)
                    try:
                        validate_time_mesh(trial)
                    except Exception:
                        continue
                    for ctrl in CONTROLLERS:
                        tasks.append((season, str(start), ctrl, h, t1, t2))
    print(f"{len(tasks)} runs")

    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_combo, *t): t for t in tasks}
        for k, fut in enumerate(as_completed(futures), start=1):
            season, _, ctrl, h, t1, t2 = futures[fut]
            tag = f"{season} {ctrl} h{h}_t1_{t1}_t2_{t2}"
            try:
                rows.append(fut.result())
                print(f"[{k}/{len(tasks)}] ok: {tag}")
            except Exception as e:
                print(f"[{k}/{len(tasks)}] FAILED: {tag}: {e}")
                rows.append({"season": season, "controller_name": ctrl,
                             "combo": f"h{h}_t1_{t1}_t2_{t2}", "status": "error", "error": str(e)})

    summary = pd.DataFrame(rows)
    summary.to_csv(f"{OUT_ROOT}/summary.csv", index=False)
    print(f"\nsummary ({len(rows)} runs): {OUT_ROOT}/summary.csv")
