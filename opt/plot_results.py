from typing import Optional, Tuple 
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# def _bar_width_and_offsets(index):
#     """Calculates bar width and offsets for plotting based on the time index."""
#     if len(index) > 1:
#         if not isinstance(index, pd.DatetimeIndex):
#             index = pd.to_datetime(index)
#         delta = (index[1] - index[0]).total_seconds()
#     else:
#         delta = 300.0  # fallback: 5 min
#     bar_width_days = 0.8 * (delta / 86400.0)
#     offset_days = 0.5 * bar_width_days
#     return bar_width_days, offset_days


def plot_identity_and_soc(
    df: pd.DataFrame,
    save_path: str = "outputs/operation_plot.png",
    e_nom_kwh=None,
):
    """Figure 1 — Supply (bars) vs Target (line) + SoC."""
    if df.empty:
        print("Nothing to plot: DataFrame is empty.")
        return

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    z = pd.Series(0.0, index=df.index)
    pv_used  = df.get("PV_used_kw", z).clip(lower=0.0)
    bess_dis  = df.get("P_bess_discharge_kw", z).clip(lower=0.0)
    bess_chg  = df.get("P_bess_charge_mag_kw", z).clip(lower=0.0)
    grid_in  = df.get("P_grid_in_kw", z).clip(lower=0.0)
    grid_out  = df.get("P_grid_out_kw", z).clip(lower=0.0)
    load_srv  = df.get("Load_served_kw", z).clip(lower=0.0)

    target_kw = load_srv + bess_chg
    
    bar_width_days, _ = _bar_width_and_offsets(df.index)

    fig, (ax0, ax1) = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.2], "hspace": 0.15}
    )

    h_pv   = ax0.bar(df.index, pv_used,  width=bar_width_days, label="PV Used", alpha=0.65)
    h_bdis = ax0.bar(df.index, bess_dis, bottom=pv_used, width=bar_width_days, label="BESS Discharge", alpha=0.65)
    bottom_pos = pv_used + bess_dis
    h_gin  = ax0.bar(df.index, grid_in,  bottom=bottom_pos, width=bar_width_days, label="Grid Import", alpha=0.65)
    h_gout = ax0.bar(df.index, -grid_out, width=bar_width_days, label="Grid Export [neg]", alpha=0.45)
    h_line = ax0.plot(df.index, target_kw, linewidth=2.0, label="Target (Load Served + BESS Charge)", zorder=3)[0]

    ax0.axhline(0.0, linewidth=1.0, alpha=0.7)
    ax0.set_ylabel("Power [kW]")
    ax0.set_title("Supply (bars) vs Target (line)")
    ax0.grid(True, axis='y', linestyle="--", alpha=0.6)

    handles = [h_pv, h_bdis, h_gin, h_gout, h_line]
    labels  = [h.get_label() for h in handles]
    ax0.legend(handles, labels, ncol=2, fontsize=9, frameon=False)

    if "SoC_pct" in df.columns:
        soc_series = pd.to_numeric(df["SoC_pct"], errors="coerce")
    elif "E_kwh" in df.columns and e_nom_kwh:
        soc_series = 100.0 * (pd.to_numeric(df["E_kwh"], errors="coerce") / max(e_nom_kwh, 1e-9))
    else:
        soc_series = pd.Series(index=df.index, dtype=float)

    ax1.plot(df.index, soc_series, linewidth=2.0, label="SoC", zorder=3, color="teal")
    ax1.set_ylabel("SoC [%]")
    ax1.grid(True, axis='y', linestyle="--", alpha=0.6)
    ax1.set_xlabel("Time")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax1.set_ylim(0, 105)

    # plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"[fig] Plot saved at: {save_path}")
    plt.close(fig)


def plot_power_balance_stacked(
    df: pd.DataFrame,
    save_path="outputs/operation_balance.png",
    e_nom_kwh=None,
):
    """Figure 2 — Supply vs Demand in side-by-side stacked bars (includes grid) + SoC."""
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
    load_srv     = df.get("Load_served_kw", z).clip(lower=0.0)
    shed         = df.get("Shedding_kw", z).clip(lower=0.0)

    fig, (ax0, ax1) = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.2], "hspace": 0.15}
    )

    h_sup1 = ax0.bar(idx_supply, pv_used, width=bar_width_days, alpha=0.6, label="Supply: PV Used")
    h_sup2 = ax0.bar(idx_supply, bess_dis, bottom=pv_used, width=bar_width_days, alpha=0.6, label="Supply: BESS Discharge")
    h_sup3 = ax0.bar(idx_supply, grid_imp, bottom=(pv_used + bess_dis), width=bar_width_days, alpha=0.6, label="Supply: Grid Import")
    h_sup4 = ax0.bar(idx_supply, bess_chg_neg, width=bar_width_days, alpha=0.4, label="Demand: BESS Charge [neg]")
    h_sup5 = ax0.bar(idx_supply, grid_exp_neg, width=bar_width_days, alpha=0.4, label="Demand: Grid Export [neg]")

    h_dem1 = ax0.bar(idx_demand, load_srv, width=bar_width_days, alpha=0.6, label="Demand: Load Served")
    h_dem2 = ax0.bar(idx_demand, shed, bottom=load_srv, width=bar_width_days, alpha=0.6, label="Demand: Load Shedding")

    ax0.axhline(0.0, linewidth=1.0)
    ax0.set_ylabel("Power [kW]")
    ax0.set_title("Power Balance — Supply vs Demand (stacked bars)")
    ax0.grid(True, axis='y', linestyle="--", alpha=0.6)

    handles = [h_sup1, h_sup2, h_sup3, h_sup4, h_sup5, h_dem1, h_dem2]
    labels  = [h.get_label() for h in handles]
    ax0.legend(handles, labels, ncol=3, fontsize=9, frameon=False)

    if "SoC_pct" in df.columns:
        soc_series = pd.to_numeric(df["SoC_pct"], errors="coerce")
    elif "E_kwh" in df.columns and e_nom_kwh:
        soc_series = 100.0 * (pd.to_numeric(df["E_kwh"], errors="coerce") / max(e_nom_kwh, 1e-9))
    else:
        soc_series = pd.Series(index=df.index, dtype=float)

    ax1.plot(df.index, soc_series, linewidth=2.0, label="SoC", zorder=3, color="teal")
    ax1.set_ylabel("SoC [%]")
    ax1.grid(True, axis='y', linestyle="--", alpha=0.6)
    ax1.set_xlabel("Time")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax1.set_ylim(0, 105)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"[fig] Plot saved at: {save_path}")
    plt.close(fig)


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from typing import Optional, Tuple
import math

# Função auxiliar para o código funcionar
def _bar_width_and_offsets(index: pd.DatetimeIndex) -> Tuple[float, dict]:
    """Calcula a largura da barra com base na frequência do índice."""
    if len(index) < 2:
        return (5 / (24 * 60)), {}
    freq = pd.infer_freq(index)
    if freq:
        delta = pd.to_timedelta(freq)
        width_days = delta.total_seconds() / (24 * 3600)
    else:
        width_days = (index[1] - index[0]).total_seconds() / (24 * 3600)
    return width_days * 0.9, {}

def plot_custom_dispatch_with_soc(
    df: pd.DataFrame,
    save_path="outputs/custom_dispatch_with_soc.pdf",
    e_nom_kwh: Optional[float] = None,
    start_dt: Optional[str] = None,
    end_dt: Optional[str] = None
):
    """
    Plots a custom dispatch view with an SoC subplot,
    highlighting off-grid periods with legends in English.
    """
    if df.empty:
        print("Nothing to plot: DataFrame is empty.")
        return

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Filtra o DataFrame pelo intervalo de datetime fornecido
    if start_dt or end_dt:
        try:
            start_ts = pd.to_datetime(start_dt) if start_dt else None
            end_ts = pd.to_datetime(end_dt) if end_dt else None
            print(f"Filtering data between {start_ts or 'the beginning'} and {end_ts or 'the end'}...")
            df = df.loc[start_ts:end_ts]
        except Exception as e:
            print(f"Error converting the provided dates: {e}")
            return
        if df.empty:
            print("Nothing to plot: DataFrame is empty after applying the date/time filter.")
            return

    # Preparação dos dados
    z = pd.Series(0.0, index=df.index)
    p_bess = -df.get("P_bess_kw", z)
    p_grid = df.get("P_grid_in_kw", z) - df.get("P_grid_out_kw", z)
    p_load_total = df.get("Load_kw", z)
    p_pv_total = df.get("PV_kw", z)
    p_load_served = df.get("Load_served_kw", z)
    p_pv_used = df.get("PV_used_kw", z)
    shedding = df.get("Shedding_kw", z)
    curtailment = df.get("Curtailment_kw", z)

    # Criação da figura
    fig, (ax0, ax1) = plt.subplots(
        nrows=2, ncols=1, figsize=(16, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1}
    )

    bar_width_days, _ = _bar_width_and_offsets(df.index)

    # --- Gráfico Superior: Despacho de Potência (ax0) ---
    ax0.bar(df.index, p_bess, width=bar_width_days, color="green", alpha=0.6, label="P_BESS (charge > 0)")
    ax0.plot(df.index, p_grid, color="royalblue", linewidth=2, label="P_Grid (import > 0)")
    ax0.plot(df.index, p_load_total, color="black", linestyle='--', linewidth=1.5, label="P_Load (potential)")
    ax0.plot(df.index, p_pv_total, color="gold", linestyle='--', linewidth=1.5, label="P_PV (potential)")
    ax0.plot(df.index, p_load_served, color="black", marker='.', markersize=4, linestyle='-', label="P_Load (served)")
    ax0.plot(df.index, p_pv_used, color="gold", linewidth=2.5, label="P_PV (used)")
    
    shed_points = df[shedding > 0.001]
    if not shed_points.empty:
        ax0.vlines(shed_points.index, shed_points["Load_served_kw"], shed_points["Load_kw"],
                   color='black', linestyle='-', linewidth=4, label='Load Shedding', alpha=0.5)

    curt_points = df[curtailment > 0.001]
    if not curt_points.empty:
        ax0.vlines(curt_points.index, curt_points["PV_used_kw"], curt_points["PV_kw"],
                   color='orange', linestyle='-', linewidth=4, label='PV Curtailment', alpha=0.6)

    # Adiciona linhas verticais para eventos off-grid
    if 'mode' in df.columns:
        df['prev_mode'] = df['mode'].shift(1)
        df['prev_mode'].fillna(df['mode'], inplace=True)
        
        offgrid_starts = df.index[(df['mode'] == 'offgrid') & (df['prev_mode'] == 'ongrid')]
        offgrid_ends = df.index[(df['mode'] == 'ongrid') & (df['prev_mode'] == 'offgrid')]

        for i, ts in enumerate(offgrid_starts):
            label = 'Off-Grid Event' if i == 0 else None
            ax0.axvline(ts, color='red', linestyle='--', linewidth=1.5, label=label)
            ax1.axvline(ts, color='red', linestyle='--', linewidth=1.5)
        
        for ts in offgrid_ends:
            ax0.axvline(ts, color='red', linestyle='--', linewidth=1.5)
            ax1.axvline(ts, color='red', linestyle='--', linewidth=1.5)

    ax0.axhline(0.0, linewidth=1.0, color='k')
    ax0.set_ylabel("Power [kW]")
    ax0.grid(True, axis='y', linestyle="--", alpha=0.6)
    
    # Legenda em duas linhas
    handles, labels = ax0.get_legend_handles_labels()
    num_cols = math.ceil(len(labels) / 2)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95),
               ncol=num_cols, frameon=False, fontsize=11)
    
    # --- Gráfico Inferior: SoC (ax1) ---
    if "SoC_pct" in df.columns:
        soc_series = pd.to_numeric(df["SoC_pct"], errors="coerce")
    elif "E_kwh" in df.columns and e_nom_kwh is not None:
        soc_series = 100.0 * (pd.to_numeric(df["E_kwh"], errors="coerce") / max(e_nom_kwh, 1e-9))
    else:
        soc_series = pd.Series(index=df.index, dtype=float)

    ax1.plot(df.index, soc_series, linewidth=2.5, label="SoC", zorder=3, color="teal")
    ax1.set_ylabel("SoC [%]")
    ax1.grid(True, axis='y', linestyle="--", alpha=0.6)
    ax1.set_xlabel("Time")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax1.set_ylim(-5, 105)
    
    # Ajusta o layout para dar espaço para a legenda
    fig.tight_layout(rect=[0, 0, 0.5, 0.0])

    # --- Salvando a figura ---
    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved at: {save_path}")
    plt.close(fig)