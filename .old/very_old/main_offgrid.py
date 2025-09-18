# -*- coding: utf-8 -*-
"""
main.py

Runs the MPC inside GridEnv (OFF-GRID or ON-GRID), saves realized operation CSV,
and produces the two plots:
  (Figure 1)
    - Bars: PV_used + BESS(discharge)
    - Line: Load_served + BESS(charge)
    - Bottom: SoC [%]
  (Figure 2)
    - Side-by-side stacked bars per timestep:
        * Supply: PV_used + BESS_discharge  (negative bar for BESS charge)
        * Demand: Load_served + Shedding
"""

import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from opt.offgrid import OffGridMPC
from env.grid_env import GridEnv

# --- data paths ---
LOAD_CSV = "data/load_5min_test.csv"
PV_CSV   = "data/pv_5min_test.csv"
PARAMS_JSON = "data/parameters.json"

# Choose mode manually: "offgrid" or "ongrid"
MODE = "offgrid"   # change to "ongrid" to enable grid import + TOU costs

# REQUIRED simulation start (GridEnv aligns to the next available timestamp if needed)
START_TS = pd.Timestamp("2009-04-26 00:00:00")  # adjust to your dataset if necessary
NUMBER_ITERS = 1440


# ---------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------

def _try_extract_first_step(mpc, times):
    """Return dict with MPC first-step decisions or None if infeasible/failed."""
    try:
        return mpc.extract_first_step(times)
    except Exception as e:
        print("[WARN] MPC infeasible or extract_first_step failed:", str(e))
        return None


def run_and_log() -> pd.DataFrame:
    """Run rolling horizon using GridEnv + your MPC; return realized operation (first step each window)."""
    # Load parameters.json via MPC (keeps your current workflow)
    mpc = OffGridMPC(PARAMS_JSON)
    p = mpc.params

    env = GridEnv(
        params=p,
        load_csv=LOAD_CSV,
        pv_csv=PV_CSV,
        start_dt0=START_TS,         # <-- now mandatory
        n_iters=NUMBER_ITERS,                # e.g., 24h with 5-min steps
        mode=MODE,
        debug=True,                 # set False to silence per-step log
        clamp_soc_pct=True,
    )

    # Rolling horizon loop
    while not env.done():
        times, forecasts = env.get_window()
        if times is None:
            print("End of data — stopping.")
            break

        # Build & solve your MPC using the environment's measured state
        mpc.build(times, forecasts, E_hat_kwh=env.E_meas)
        mpc.solve("gurobi", tee=False)

        # First-step decisions from MPC (robust to infeasibility)
        step0 = _try_extract_first_step(mpc, times)

        if step0 is None:
            # Safe fallback -> GridEnv will enforce JSON & close balance
            P_bess = 0.0
            X_L    = None   # let env auto-fix if needed
            X_PV   = None
            obj    = None
            print("[FALLBACK] Applying safe step: P_bess=0, X_L/X_PV=None; GridEnv will close the balance.")
        else:
            P_bess = float(step0.get("P_bess_kw", 0.0))
            X_L    = float(step0.get("X_L", 0.0))
            X_PV   = float(step0.get("X_PV", 0.0))
            obj    = float(step0.get("obj", 0.0))

        # Apply to environment (it will enforce JSON constraints & safety)
        _, done = env.step(P_bess_kw=P_bess, X_L=X_L, X_PV=X_PV, obj=obj)
        if done:
            break

    # Final realized operation
    df = env.to_dataframe()
    return df


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def save_operation_csv(df: pd.DataFrame, path="outputs/operation_real.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)
    print(f"[csv] operação real salva em: {path}")


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def _bar_width_and_offsets(index) -> tuple:
    """Compute bar width (in days) and lateral offset for side-by-side bars."""
    if len(index) > 1:
        delta = (index[1] - index[0]).total_seconds()
    else:
        delta = 300.0  # fallback: 5 min
    bar_width_days = 0.8 * (delta / 86400.0)
    offset_days = 0.5 * bar_width_days
    return bar_width_days, offset_days


def plot_identity_and_soc(df: pd.DataFrame, save_path: str = "outputs/operation_plot.png", e_nom_kwh: Optional[float] = None):
    """Figure 1 — Bars: PV_used + BESS(discharge); Line: LOAD(1−X) + BESS(charge); Bottom: SoC[%]."""
    if df.empty:
        print("Nada a plotar: DataFrame vazio.")
        return

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # target line = Load served + BESS charging power
    load_served = df["Load_served_kw"].clip(lower=0.0)
    bess_charge_mag = df["P_bess_charge_mag_kw"].clip(lower=0.0) if "P_bess_charge_mag_kw" in df.columns else 0.0
    target_kw = load_served + bess_charge_mag

    pv_used   = df["PV_used_kw"].clip(lower=0.0)
    bess_dis  = df["P_bess_discharge_kw"].clip(lower=0.0)
    bar_width_days, _ = _bar_width_and_offsets(df.index)

    fig, (ax0, ax1) = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.2], "hspace": 0.15}
    )

    # Top subplot: stacked supply vs target line
    ax0.bar(df.index, pv_used,  width=bar_width_days, label="PV·(1−X_PV)", alpha=0.65)
    ax0.bar(df.index, bess_dis, bottom=pv_used, width=bar_width_days, label="BESS (discharge)", alpha=0.65)
    ax0.plot(df.index, target_kw, linewidth=2.0, label="LOAD·(1−X_L) + BESS (charge)", zorder=3)

    ax0.set_ylabel("Power [kW]")
    ax0.grid(True, axis='y', linestyle="--", alpha=0.6)
    ax0.legend(ncol=2, fontsize=9, frameon=False)

    # Bottom subplot: SoC [%]
    soc_series = None
    if "SoC_pct" in df.columns:
        soc_series = pd.to_numeric(df["SoC_pct"], errors="coerce")
        if soc_series.max(skipna=True) is not None and soc_series.max(skipna=True) > 1000:
            # safety fix if someone multiplied by 100 twice upstream
            soc_series = soc_series / 100.0
    elif "E_kwh" in df.columns and e_nom_kwh:
        soc_series = 100.0 * (pd.to_numeric(df["E_kwh"], errors="coerce") / max(e_nom_kwh, 1e-9))
        print("[plot_identity_and_soc] Using SoC derived from E_kwh/E_nom_kwh.")

    if soc_series is None:
        soc_series = pd.Series(index=df.index, dtype=float)

    ax1.plot(df.index, soc_series, linewidth=2.0, label="SoC", zorder=3)
    smin = soc_series.min(skipna=True)
    smax = soc_series.max(skipna=True)
    if pd.notna(smin) and pd.notna(smax):
        lo = 0 if smin >= 0 else min(-5, int(smin) - 5)
        hi = 100 if smax <= 100 else max(105, int(smax) + 5)
        ax1.set_ylim(lo, hi)
    else:
        ax1.set_ylim(0, 100)

    ax1.set_ylabel("SoC [%]")
    ax1.grid(True, axis='y', linestyle="--", alpha=0.6)
    ax1.set_xlabel("Time")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"[fig] gráfico salvo em: {save_path}")
    plt.close(fig)


def plot_power_balance_stacked(df: pd.DataFrame, save_path="outputs/operation_balance.png"):
    """Figure 2 — Side-by-side stacked bars (Supply vs Demand)."""
    if df.empty:
        print("Nada a plotar: DataFrame vazio.")
        return

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # width & offsets
    bar_width_days, offset_days = _bar_width_and_offsets(df.index)
    idx_supply = df.index - pd.to_timedelta(offset_days, unit="D")
    idx_demand = df.index + pd.to_timedelta(offset_days, unit="D")

    pv_used   = df["PV_used_kw"].clip(lower=0.0)
    bess_dis  = df["P_bess_discharge_kw"].clip(lower=0.0)
    bess_chg_neg = -df["P_bess_charge_mag_kw"].clip(lower=0.0) if "P_bess_charge_mag_kw" in df.columns else None

    load_srv  = df["Load_served_kw"].clip(lower=0.0)
    shed      = df["Shedding_kw"].clip(lower=0.0) if "Shedding_kw" in df.columns else None

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)

    # Supply positive stack
    ax.bar(idx_supply, pv_used, width=bar_width_days, alpha=0.6, label="Supply: PV used")
    ax.bar(idx_supply, bess_dis, bottom=pv_used, width=bar_width_days, alpha=0.6, label="Supply: BESS (discharge)")
    # Supply negative (BESS charging)
    if bess_chg_neg is not None and (bess_chg_neg != 0).any():
        ax.bar(idx_supply, bess_chg_neg, width=bar_width_days, alpha=0.4, label="Supply: BESS (charge) [neg]")

    # Demand stacked
    ax.bar(idx_demand, load_srv, width=bar_width_days, alpha=0.6, label="Demand: Load served")
    if shed is not None and shed.sum() > 0:
        ax.bar(idx_demand, shed, bottom=load_srv, width=bar_width_days, alpha=0.6, label="Demand: Shedding")

    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel("Power [kW]")
    ax.set_title("Power balance — Supply vs Demand (stacked bars)")
    ax.grid(True, axis='y', linestyle="--", alpha=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.legend(ncol=3, fontsize=9, frameon=False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"[fig] gráfico salvo em: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    df = run_and_log()
    save_operation_csv(df, path="outputs/operation_real.csv")
    plot_identity_and_soc(df, save_path="outputs/operation_plot.png")
    plot_power_balance_stacked(df, save_path="outputs/operation_balance.png")


if __name__ == "__main__":
    main()
