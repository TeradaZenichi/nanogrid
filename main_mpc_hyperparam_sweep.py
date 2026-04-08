import json
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd

from opt.ongrid import OnGridMPC
from opt.utils import build_dt_vector, build_time_grid, slice_forecasts


PARAMS_JSON = Path("data/parameters.json")
LOAD_CSV = Path("data/load_5min_test.csv")
PV_CSV = Path("data/pv_5min_test.csv")
RESULTS_DIR = Path("Results/mpc")

HORIZONS_H = [6, 12, 24, 36]
TIMESTEP_1_MIN = [5, 10, 15]
TIMESTEP_2_MIN = [30, 60, 120, 180]

START_TS = pd.Timestamp("2009-05-01 00:00:00")
SOLVER_NAME = "gurobi"
SOLVER_TIME_LIMIT_S = 60
SOLVER_MIP_GAP = 0.02


def _load_scaled_series(load_csv: Path, pv_csv: Path, p_load_max: float, p_pv_max: float):
    load_df = pd.read_csv(load_csv)
    pv_df = pd.read_csv(pv_csv)
    for name, df in [("load", load_df), ("pv", pv_df)]:
        if "timestamp" not in df.columns or "p_norm" not in df.columns:
            raise ValueError(f"{name} CSV must contain columns: timestamp, p_norm")

    load_df["timestamp"] = pd.to_datetime(load_df["timestamp"])
    pv_df["timestamp"] = pd.to_datetime(pv_df["timestamp"])
    load_df = load_df.set_index("timestamp").sort_index()
    pv_df = pv_df.set_index("timestamp").sort_index()

    load_kw = load_df["p_norm"].astype(float).clip(0.0, 1.0) * float(p_load_max)
    pv_kw = pv_df["p_norm"].astype(float).clip(0.0, 1.0) * float(p_pv_max)
    return load_kw, pv_kw


def _is_success(status: str, term: str) -> bool:
    return (status.lower() == "ok") and (term.lower() in {"optimal", "feasible"})


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(PARAMS_JSON, "r", encoding="utf-8") as f:
        base_params = json.load(f)

    p_load_max = float(base_params["Load"]["Pmax_kw"])
    p_pv_max = float(base_params["PV"]["Pmax_kw"])
    e_init = float(base_params["BESS"].get("E_init_kwh", base_params["BESS"]["Emax_kwh"]))
    load_kw_s, pv_kw_s = _load_scaled_series(LOAD_CSV, PV_CSV, p_load_max, p_pv_max)

    rows = []
    detailed = []
    case_id = 0

    for h in HORIZONS_H:
        for t1 in TIMESTEP_1_MIN:
            for t2 in TIMESTEP_2_MIN:
                case_id += 1
                tag = f"h{h}_t1_{t1}_t2_{t2}"
                row = {
                    "case_id": case_id,
                    "tag": tag,
                    "horizon_hours": h,
                    "timestep_1_min": t1,
                    "timestep_2_min": t2,
                    "status": "not_run",
                    "termination": "",
                    "build_time_s": None,
                    "solve_time_s": None,
                    "objective": None,
                    "P_bess_kw": None,
                    "X_L": None,
                    "X_PV": None,
                    "n_T": None,
                    "n_C": None,
                    "error": "",
                }

                try:
                    params = deepcopy(base_params)
                    params["time"]["horizon_hours"] = int(h)
                    params["time"]["timestep_1_min"] = int(t1)
                    params["time"]["timestep_2_min"] = int(t2)

                    outage_h = int(params.get("EDS", {}).get("outage_duration_hours", 0))
                    dt_vector = build_dt_vector(
                        horizon_hours=int(h),
                        outage_duration_hours=outage_h,
                        dt1_min=int(t1),
                        dt2_min=int(t2),
                    )

                    times = build_time_grid(START_TS, dt_vector)
                    forecasts = slice_forecasts(times, load_kw_s, pv_kw_s)

                    tmp_params_path = RESULTS_DIR / f"params_{tag}.json"
                    with open(tmp_params_path, "w", encoding="utf-8") as f:
                        json.dump(params, f, indent=2)

                    mpc = OnGridMPC(params_path=str(tmp_params_path), relaxation=True)

                    t0 = time.perf_counter()
                    mpc.build(
                        start_dt=START_TS,
                        forecasts=forecasts,
                        E_hat_kwh=e_init,
                        P_bess_hat_kw=0.0,
                    )
                    t1_build = time.perf_counter()

                    mpc.solve(
                        solver_name=SOLVER_NAME,
                        tee=False,
                        TimeLimit=SOLVER_TIME_LIMIT_S,
                        MIPGap=SOLVER_MIP_GAP,
                    )
                    t2_solve = time.perf_counter()

                    status = str(mpc.results.solver.status)
                    term = str(mpc.results.solver.termination_condition)
                    row["status"] = status
                    row["termination"] = term
                    row["build_time_s"] = t1_build - t0
                    row["solve_time_s"] = t2_solve - t1_build
                    row["n_T"] = len(mpc.model.T)
                    row["n_C"] = len(mpc.model.C)

                    if _is_success(status, term):
                        step0 = mpc.extract_first_step(scenario="c0")
                        row["objective"] = float(step0.get("obj", 0.0))
                        row["P_bess_kw"] = float(step0.get("P_bess_kw", 0.0))
                        row["X_L"] = float(step0.get("X_L", 0.0))
                        row["X_PV"] = float(step0.get("X_PV", 0.0))

                        detailed.append(
                            {
                                "tag": tag,
                                "params_path": str(tmp_params_path),
                                "first_step": step0,
                                "all_first_steps": mpc.extract_first_step_all(),
                            }
                        )
                    else:
                        row["error"] = f"solver status={status}, term={term}"

                except Exception as e:
                    row["status"] = "error"
                    row["termination"] = "exception"
                    row["error"] = str(e)

                rows.append(row)
                print(f"[{case_id:02d}/48] {tag} -> {row['status']} ({row['termination']})")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "mpc_hyperparam_sweep_summary.csv", index=False)

    with open(RESULTS_DIR / "mpc_hyperparam_sweep_details.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(detailed), f, indent=2, ensure_ascii=False, default=str)

    agg = {
        "total_cases": int(len(df)),
        "success_cases": int(((df["status"].str.lower() == "ok") & (df["termination"].str.lower().isin(["optimal", "feasible"]))).sum()),
        "error_cases": int((df["status"].str.lower() == "error").sum()),
        "invalid_or_other_cases": int(len(df) - (((df["status"].str.lower() == "ok") & (df["termination"].str.lower().isin(["optimal", "feasible"]))).sum()) - (df["status"].str.lower() == "error").sum()),
        "avg_solve_time_s_success": float(df.loc[(df["status"].str.lower() == "ok") & (df["termination"].str.lower().isin(["optimal", "feasible"])), "solve_time_s"].mean()),
    }
    with open(RESULTS_DIR / "mpc_hyperparam_sweep_aggregate.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    print((RESULTS_DIR / "mpc_hyperparam_sweep_summary.csv").as_posix())
