"""Extra dt2 points (15/20/40 min) on the Pareto line h=36, dt1=10.

Complements the {30,60,120} grid of 05_mesh_sweep.py and rebuilds that
sweep's summary.csv from every metrics.json on disk.
"""

import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting import PerfectForecast, PrototypeForecast
from opt import simulate_mpc
from opt.utils import load_series_scaled

START_TS = pd.Timestamp("2009-05-01 00:00:00")  # same window as 05_mesh_sweep
N_ITERS = 2880
OUT_ROOT = "Results/5-mesh_sweep"
PARAMS_JSON = "data/parameters.json"

H = 36
T1 = 10
NEW_T2 = [15, 20, 40]  # (H - 2h) * 60 = 2040 min is divisible by each
CONTROLLERS = ["prototype", "ideal"]


def rebuild_summary() -> pd.DataFrame:
    rows = []
    for ctrl_dir in sorted(Path(OUT_ROOT).iterdir()):
        if not ctrl_dir.is_dir():
            continue
        for case in sorted(ctrl_dir.iterdir()):
            mj = case / "metrics.json"
            if mj.exists():
                m = json.loads(mj.read_text(encoding="utf-8"))
                m["controller_name"] = ctrl_dir.name
                m["combo"] = case.name
                rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_ROOT}/summary.csv", index=False)
    return df


if __name__ == "__main__":
    base = json.load(open(PARAMS_JSON, encoding="utf-8"))
    scaling = {"P_L_nom_kw": base["Load"]["Pmax_kw"], "P_PV_nom_kw": base["PV"]["Pmax_kw"]}
    load_s, pv_s = load_series_scaled(scaling, "data/load_5min_test.csv", "data/pv_5min_test.csv")
    pv_kw = float(base["PV"]["Pmax_kw"])
    load_kw = float(base["Load"]["Pmax_kw"])

    forecasters = {
        "prototype": PrototypeForecast(None, load_s, pv_s, pv_kw, load_kw, strategy="prefix"),
        "ideal": PerfectForecast(load_s, pv_s),
    }

    for t2 in NEW_T2:
        params = deepcopy(base)
        params["time"].update(horizon_hours=H, timestep_1_min=T1, timestep_2_min=t2)
        for ctrl in CONTROLLERS:
            combo = f"h{H}_t1_{T1}_t2_{t2}"
            print(f"===== {ctrl} {combo} =====")
            simulate_mpc(params, forecasters[ctrl], START_TS, N_ITERS,
                         f"{OUT_ROOT}/{ctrl}/{combo}", forecaster_name=ctrl)

    df = rebuild_summary()
    print(f"\nsummary rebuilt: {len(df)} cases in {OUT_ROOT}/summary.csv")
