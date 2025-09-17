import os
import json
import time
import pandas as pd
from pathlib import Path

# --- Local Imports ---
from env.grid_env import GridEnv
from opt.ongrid import OnGridMPC as OnGridStochasticMPC
from forecasting.get_forecasting import get_window
from opt.plot_results import (
    plot_identity_and_soc,
    plot_power_balance_stacked,
    plot_custom_dispatch_with_soc
)

# ==========================
# Configuration
# ==========================
LOAD_CSV = "data/load_5min_test_smoothed_normalized.csv"
PV_CSV   = "data/pv_5min_test.csv"
PARAMS_JSON = "data/parameters.json"
OUTPUT_DIR = Path("outputs")
MPC_PLANS_DIR = OUTPUT_DIR / "mpc_solutions"

# Simulation window
START_TS = pd.Timestamp("2009-05-01 00:00:00")
NUMBER_ITERS = 288 * 10  # e.g., 10 days @ 5-min

# Solver settings for on-grid MPC
SOLVER_NAME = "gurobi"
SOLVER_TEE = False

# ==========================
# I/O Helpers
# ==========================
def save_operation_csv(df: pd.DataFrame, path="outputs/operation_real.csv"):
    """Save rolling operation DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)
    print(f"[csv] Operation log saved at: {path}")

# ==========================
# Main Simulation
# ==========================
if __name__ == "__main__":
    # Prepare directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    MPC_PLANS_DIR.mkdir(exist_ok=True)

    # Load parameters
    with open(PARAMS_JSON, "r") as f:
        params = json.load(f)

    # Initialize on-grid controller only (no off-grid MPC anywhere)
    mpc_ongrid = OnGridStochasticMPC(PARAMS_JSON, relaxation=True)

    # Initialize environment (env manages outage state internally)
    env = GridEnv(
        params=params,
        load_csv=LOAD_CSV,
        pv_csv=PV_CSV,
        start_dt0=START_TS,
        n_iters=NUMBER_ITERS,
        debug=True,
    )

    # Track previous-step BESS net power to anchor ramp on the first on-grid step
    pbess_prev_kw = 0.0

    # Rolling-horizon loop
    while not env.done():
        loop_start_time = time.perf_counter()

        # Current state & forecasts for the window
        current_time = env.timestamp
        forecasts = get_window(current_time, env.load_kw_s, env.pv_kw_s, env.dt_min)

        # Mode decided by env in the previous step's tail
        current_mode = env.mode

        if current_mode == "offgrid":
            print(f"\n--- [Off-Grid Mode] Timestamp: {current_time} ---")
            # No off-grid MPC: env will control BESS internally in step()
            step0 = None  # placeholder so we send neutral commands
            full_plan = {
                "mode": "offgrid",
                "autonomous_bess": True,
                "timestamp": current_time.isoformat(),
                "note": "Env controls BESS; PV surplus used to charge; XL/XPV forced internally."
            }
        else:
            print(f"\n--- [On-Grid Mode] Timestamp: {current_time} ---")
            # Build on-grid stochastic MPC with first-step ramp anchored to pbess_prev_kw
            try:
                mpc_ongrid.build(
                    start_dt=current_time,
                    forecasts=forecasts,
                    E_hat_kwh=env.E_meas,
                    P_bess_hat_kw=pbess_prev_kw  # anchor first-step ramp
                )
                mpc_ongrid.solve(SOLVER_NAME, tee=SOLVER_TEE)
                step0 = mpc_ongrid.extract_first_step(scenario="c0")
                full_plan = mpc_ongrid.extract_full_solution()
            except Exception as e:
                print(f"[WARN] OnGridMPC failed: {e}. Applying safe fallback.")
                step0 = None
                full_plan = {
                    "mode": "ongrid",
                    "timestamp": current_time.isoformat(),
                    "note": "Safe fallback after MPC error."
                }

        # Save plan (on-grid real plan or off-grid stub) for this timestamp
        plan_filename = f"mpc_plan_{current_time.strftime('%Y%m%dT%H%M%S')}.json"
        plan_path = MPC_PLANS_DIR / plan_filename
        try:
            with open(plan_path, 'w') as f:
                json.dump(full_plan, f, indent=4)
            print(f"Full MPC plan saved to: {plan_path}")
        except Exception as e:
            print(f"[WARN] Could not save plan file: {e}")

        # Decide action to apply to env
        if step0 is None:
            # Off-grid: env will override; send placeholders
            # On-grid fallback: env will still try to balance per its logic
            P_bess, X_L, X_PV, obj = 0.0, None, None, None
        else:
            P_bess = float(step0.get("P_bess_kw", 0.0))
            X_L    = float(step0.get("X_L", 0.0))
            X_PV   = float(step0.get("X_PV", 0.0))
            obj    = float(step0.get("obj", 0.0))

        loop_end_time = time.perf_counter()
        execution_time = loop_end_time - loop_start_time
        print(f"Loop execution time: {execution_time:.4f} seconds")

        # Apply action (env will ignore/override in off-grid as designed)
        row, done = env.step(
            P_bess_kw=P_bess,
            X_L=X_L,
            X_PV=X_PV,
            obj=obj,
            exec_time_sec=execution_time
        )

        # Update previous-step BESS setpoint with actually applied/measured net power
        if isinstance(row, dict) and ("P_bess_kw" in row):
            try:
                pbess_prev_kw = float(row["P_bess_kw"])
            except Exception:
                pbess_prev_kw = P_bess
        else:
            pbess_prev_kw = P_bess

        if done:
            break

        # Rolling log (can be throttled if needed)
        df_operation = env.to_dataframe()
        save_operation_csv(df_operation, path=OUTPUT_DIR / "operation_real.csv")
    print("\nSimulation finished.")
