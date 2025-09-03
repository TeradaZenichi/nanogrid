import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def _bar_width_and_offsets(index):
    if len(index) > 1:
        delta = (index[1] - index[0]).total_seconds()
    else:
        delta = 300.0  # fallback: 5 min
    bar_width_days = 0.8 * (delta / 86400.0)
    offset_days = 0.5 * bar_width_days
    return bar_width_days, offset_days


def plot_identity_and_soc(
    df: pd.DataFrame,
    save_path: str = "outputs/operation_plot.png",
    e_nom_kwh = None,
):
    """Figure 1 — Supply (bars) vs Target (line) + SoC."""
    if df.empty:
        print("Nothing to plot: DataFrame is empty.")
        return

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    z = pd.Series(0.0, index=df.index)
    pv_used   = df.get("PV_used_kw", z).clip(lower=0.0)
    bess_dis  = df.get("P_bess_discharge_kw", z).clip(lower=0.0)
    bess_chg  = df.get("P_bess_charge_mag_kw", z).clip(lower=0.0)
    grid_in   = df.get("P_grid_in_kw", z).clip(lower=0.0)
    grid_out  = df.get("P_grid_out_kw", z).clip(lower=0.0)
    load_srv  = df.get("Load_served_kw", z).clip(lower=0.0)

    target_kw = load_srv + bess_chg
    supply_net = pv_used + bess_dis + grid_in - grid_out
    residual = (supply_net - target_kw).abs().max()

    bar_width_days, _ = _bar_width_and_offsets(df.index)

    fig, (ax0, ax1) = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.2], "hspace": 0.15}
    )

    h_pv   = ax0.bar(df.index, pv_used,  width=bar_width_days, label="PV·(1−X_PV)", alpha=0.65)
    h_bdis = ax0.bar(df.index, bess_dis, bottom=pv_used, width=bar_width_days, label="BESS (discharge)", alpha=0.65)
    bottom_pos = pv_used + bess_dis
    h_gin  = ax0.bar(df.index, grid_in,  bottom=bottom_pos, width=bar_width_days, label="Grid import", alpha=0.65)
    h_gout = ax0.bar(df.index, -grid_out, width=bar_width_days, label="Grid export [neg]", alpha=0.45)

    h_line = ax0.plot(df.index, target_kw, linewidth=2.0, label="LOAD·(1−X_L) + BESS (charge)", zorder=3)

    ax0.axhline(0.0, linewidth=1.0, alpha=0.7)
    ax0.set_ylabel("Power [kW]")
    ax0.set_title(f"Supply (bars) vs Target (line) — max |residual| = {float(residual):.4f} kW")
    ax0.grid(True, axis='y', linestyle="--", alpha=0.6)

    handles = [h_pv, h_bdis, h_gin, h_gout, h_line]
    labels  = [h.get_label() if hasattr(h, 'get_label') else 'LOAD·(1−X_L) + BESS (charge)' for h in handles]
    ax0.legend(handles, labels, ncol=2, fontsize=9, frameon=False)

    if "SoC_pct" in df.columns:
        soc_series = pd.to_numeric(df["SoC_pct"], errors="coerce")
    elif "E_kwh" in df.columns and e_nom_kwh:
        soc_series = 100.0 * (pd.to_numeric(df["E_kwh"], errors="coerce") / max(e_nom_kwh, 1e-9))
    else:
        soc_series = pd.Series(index=df.index, dtype=float)

    ax1.plot(df.index, soc_series, linewidth=2.0, label="SoC", zorder=3)
    ax1.set_ylabel("SoC [%]")
    ax1.grid(True, axis='y', linestyle="--", alpha=0.6)
    ax1.set_xlabel("Time")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"[fig] Plot saved at: {save_path}")
    plt.close(fig)


def plot_power_balance_stacked(df: pd.DataFrame, save_path="outputs/operation_balance.png"):
    """Figure 2 — Supply vs Demand in side-by-side stacked bars (includes grid)."""
    if df.empty:
        print("Nothing to plot: DataFrame is empty.")
        return

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    bar_width_days, offset_days = _bar_width_and_offsets(df.index)
    idx_supply = df.index - pd.to_timedelta(offset_days, unit="D")
    idx_demand = df.index + pd.to_timedelta(offset_days, unit="D")

    z = pd.Series(0.0, index=df.index)
    pv_used      = df.get("PV_used_kw", z).clip(lower=0.0)
    bess_dis     = df.get("P_bess_discharge_kw", z).clip(lower=0.0)
    bess_chg_neg = -df.get("P_bess_charge_mag_kw", z).clip(lower=0.0)
    grid_imp     = df.get("P_grid_in_kw", z).clip(lower=0.0)
    grid_exp_neg = -df.get("P_grid_out_kw", z).clip(lower=0.0)

    load_srv  = df.get("Load_served_kw", z).clip(lower=0.0)
    shed      = df.get("Shedding_kw", z).clip(lower=0.0)

    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)

    h_sup1 = ax.bar(idx_supply, pv_used, width=bar_width_days, alpha=0.6, label="Supply: PV used")
    h_sup2 = ax.bar(idx_supply, bess_dis, bottom=pv_used, width=bar_width_days, alpha=0.6, label="Supply: BESS (discharge)")
    h_sup3 = ax.bar(idx_supply, grid_imp, bottom=(pv_used + bess_dis), width=bar_width_days, alpha=0.6, label="Supply: Grid import")

    h_sup4 = ax.bar(idx_supply, bess_chg_neg, width=bar_width_days, alpha=0.4, label="Supply: BESS (charge) [neg]")
    h_sup5 = ax.bar(idx_supply, grid_exp_neg, width=bar_width_days, alpha=0.4, label="Supply: Grid export [neg]")

    h_dem1 = ax.bar(idx_demand, load_srv, width=bar_width_days, alpha=0.6, label="Demand: Load served")
    h_dem2 = ax.bar(idx_demand, shed, bottom=load_srv, width=bar_width_days, alpha=0.6, label="Demand: Shedding")

    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel("Power [kW]")
    ax.set_title("Power balance — Supply vs Demand (stacked bars, incl. grid)")
    ax.grid(True, axis='y', linestyle="--", alpha=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))

    handles = [h_sup1, h_sup2, h_sup3, h_sup4, h_sup5, h_dem1, h_dem2]
    labels  = [h.get_label() for h in handles]
    ax.legend(handles, labels, ncol=3, fontsize=9, frameon=False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"[fig] Plot saved at: {save_path}")
    plt.close(fig)
