import os
import csv
import pandas as pd

from opt.plot_results import plot_identity_and_soc, plot_power_balance_stacked
from forecasting.get_forecasting import get_window
from opt.offgrid import OffGridMPC
from opt.ongrid import OnGridMPC
from env.grid_env import GridEnv
import numpy as np
import json

# --- data paths ---
LOAD_CSV = "data/load_5min_test.csv"
PV_CSV   = "data/pv_5min_test.csv"
PARAMS_JSON = "data/parameters.json"


# REQUIRED simulation start (GridEnv aligns to next available timestamp if necessary)
START_TS = pd.Timestamp("2009-04-26 00:00:00")
NUMBER_ITERS = 288

# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def save_operation_csv(df: pd.DataFrame, path="outputs/operation_real.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)
    print(f"[csv] Operation log saved at: {path}")



if __name__ == "__main__":
    with open(PARAMS_JSON, "r") as f:
        params = json.load(f)

    mpc_ongrid = OnGridMPC(PARAMS_JSON)
    mpc_offgrid = OffGridMPC(PARAMS_JSON)
    
    env = GridEnv(
        params=params,
        load_csv=LOAD_CSV,
        pv_csv=PV_CSV,
        start_dt0=START_TS,
        n_iters=NUMBER_ITERS,
        debug=True,
        clamp_soc_pct=True,
    )

    # --- LOG DE CENÁRIOS ---
    scen_path = "outputs/mpc_firststep_scenarios.csv"
    os.makedirs(os.path.dirname(scen_path), exist_ok=True)
    
    # MPC loop (rolling horizon)
    while not env.done():
        forecasts = get_window(env.timestamp, env.load_kw_s, env.pv_kw_s, env.dt_min)
   
        if np.random.rand() < env.outage_prob:
            env.mode = "offgrid"
            mpc_offgrid.build(start_dt=env.timestamp, forecasts=forecasts, E_hat_kwh=env.E_meas)
            mpc_offgrid.solve("gurobi", tee=False)
            step0 = mpc_offgrid.extract_first_step()
        else:
            env.mode = "ongrid"
            mpc_ongrid.build(start_dt=env.timestamp, forecasts=forecasts, E_hat_kwh=env.E_meas)
            mpc_ongrid.solve("gurobi", tee=False)
            step0 = mpc_ongrid.extract_first_step()


        if step0 is None:
            print("[FALLBACK] Applying safe step: P_bess=0, X_L/X_PV=None; GridEnv will close the balance.")
            P_bess, X_L, X_PV, obj = 0.0, None, None, None
        else:
            P_bess = float(step0.get("P_bess_kw", 0.0))
            X_L = float(step0.get("X_L", 0.0))
            X_PV = float(step0.get("X_PV", 0.0))
            obj = float(step0.get("obj", 0.0))

        # Apply to the environment
        row, done = env.step(P_bess_kw=P_bess, X_L=X_L, X_PV=X_PV, obj=obj)

        if done:
            break

    # Return the final DataFrame
    df = env.to_dataframe()
    print(f"[csv] Scenario log saved at: {scen_path}")
    save_operation_csv(df, path="outputs/operation_real.csv")
    plot_identity_and_soc(df, save_path="outputs/operation_plot.png")
    plot_power_balance_stacked(df, save_path="outputs/operation_balance.png")
