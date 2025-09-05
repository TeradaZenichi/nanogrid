import os
import json
import time # <-- Import the time module
import pandas as pd
from pathlib import Path

# --- Local Imports ---
# Ensure the paths to your modules are correct
from env.grid_env import GridEnv
from opt.offgrid import OffGridMPC
from opt.ongrid import OnGridMPC as OnGridStochasticMPC
from forecasting.get_forecasting import get_window
from opt.plot_results import (
    plot_identity_and_soc,
    plot_power_balance_stacked,
    plot_custom_dispatch_with_soc
)

# --- Configuration ---
LOAD_CSV = "data/load_5min_test.csv"
PV_CSV = "data/pv_5min_test.csv"
PARAMS_JSON = "data/parameters.json"
OUTPUT_DIR = Path("outputs")
MPC_PLANS_DIR = OUTPUT_DIR / "mpc_solutions" # Directory to store MPC plans

# REQUIRED simulation start
START_TS = pd.Timestamp("2009-05-01 00:00:00")
# NUMBER_ITERS = 288 # Corresponds to 24 hours with a 5-min timestep
NUMBER_ITERS = 288 * 10 # Corresponds to 24 hours with a 15-min timestep

# ---------------------------------------------------------------------
# I/O Helpers
# ---------------------------------------------------------------------

def save_operation_csv(df: pd.DataFrame, path="outputs/operation_real.csv"):
    """Saves the final operation DataFrame to a CSV file."""
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)
    print(f"[csv] Operation log saved at: {path}")

# ---------------------------------------------------------------------
# Main Simulation
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Create output directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    MPC_PLANS_DIR.mkdir(exist_ok=True)

    # Load shared parameters
    with open(PARAMS_JSON, "r") as f:
        params = json.load(f)

    # --- Initialize Models and Environment ---
    # Initialize both MPC controllers. The relaxation parameter can be toggled here.
    mpc_ongrid = OnGridStochasticMPC(PARAMS_JSON, relaxation=False)
    mpc_offgrid = OffGridMPC(PARAMS_JSON, relaxation=False)
    
    # Initialize the simulation environment
    # The environment will now manage its own outage state
    env = GridEnv(
        params=params,
        load_csv=LOAD_CSV,
        pv_csv=PV_CSV,
        start_dt0=START_TS,
        n_iters=NUMBER_ITERS,
        debug=True,
    )

    # --- MPC Loop (Rolling Horizon) ---
    while not env.done():
        # --- Start timing the loop iteration ---
        loop_start_time = time.perf_counter()

        # Get current state and forecasts for the optimization window
        current_time = env.timestamp
        forecasts = get_window(current_time, env.load_kw_s, env.pv_kw_s, env.dt_min)
   
        # The environment determines its own mode ('ongrid' or 'offgrid') internally
        if env.mode == "offgrid":
            print(f"\n--- [Off-Grid Mode] Solving MPC for timestamp: {current_time} ---")
            active_mpc = mpc_offgrid
            active_mpc.build(start_dt=current_time, forecasts=forecasts, E_hat_kwh=env.E_meas)
            active_mpc.solve("gurobi", tee=False)
            step0 = active_mpc.extract_first_step()
        else: # ongrid mode
            print(f"\n--- [On-Grid Mode] Solving MPC for timestamp: {current_time} ---")
            active_mpc = mpc_ongrid
            active_mpc.build(start_dt=current_time, forecasts=forecasts, E_hat_kwh=env.E_meas)
            active_mpc.solve("gurobi", tee=False)
            # For the stochastic on-grid MPC, we use the base case "c0" for the action
            step0 = active_mpc.extract_first_step(scenario="c0")

        # --- Extract and Save the Full MPC Plan for the current step ---
        full_plan = active_mpc.extract_full_solution()
        plan_filename = f"mpc_plan_{current_time.strftime('%Y%m%dT%H%M%S')}.json"
        plan_path = MPC_PLANS_DIR / plan_filename
        with open(plan_path, 'w') as f:
            json.dump(full_plan, f, indent=4)
        print(f"Full MPC plan saved to: {plan_path}")

        # Determine the action to apply to the environment
        if step0 is None:
            print("[FALLBACK] Solver failed. Applying safe step: P_bess=0. Environment will balance.")
            P_bess, X_L, X_PV, obj = 0.0, None, None, None
        else:
            P_bess = float(step0.get("P_bess_kw", 0.0))
            X_L = float(step0.get("X_L", 0.0))
            X_PV = float(step0.get("X_PV", 0.0))
            obj = float(step0.get("obj", 0.0))
        
        # --- Stop timing and calculate duration ---
        loop_end_time = time.perf_counter()
        execution_time = loop_end_time - loop_start_time
        print(f"Loop execution time: {execution_time:.4f} seconds")

        # Apply action to the environment and advance one time step
        # Pass the calculated execution time to the environment
        row, done = env.step(
            P_bess_kw=P_bess,
            X_L=X_L,
            X_PV=X_PV,
            obj=obj,
            exec_time_sec=execution_time
        )
        
        if done:
            break

        # --- Final Results ---
        df_operation = env.to_dataframe()
        save_operation_csv(df_operation, path=OUTPUT_DIR / "operation_real.csv")
    
    # Generate final plots from the resulting operation data
    print("\nGenerating final plots...")
    e_nom_kwh = env.bess.get("E_nom")
    plot_identity_and_soc(df_operation, save_path=OUTPUT_DIR / "operation_plot.png", e_nom_kwh=e_nom_kwh)
    plot_power_balance_stacked(df_operation, save_path=OUTPUT_DIR / "operation_balance.png", e_nom_kwh=e_nom_kwh)
    plot_custom_dispatch_with_soc(df_operation, save_path=OUTPUT_DIR / "custom_dispatch.png", e_nom_kwh=e_nom_kwh)

    print("\nSimulation finished.")