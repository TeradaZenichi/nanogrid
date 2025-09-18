# -*- coding: utf-8 -*-
"""
main.py

Runs the MPC inside GridEnv (OFF-GRID or ON-GRID), saves realized operation CSV,
and produces the two plots:

  (Figure 1)  Abordagem A (híbrida):
    - Barras (positivas): PV_used + BESS(discharge) + Grid import
    - Barra  (negativa):  Grid export
    - Linha:  Load_served + BESS(charge)
    - Bottom: SoC [%]

  (Figure 2)
    - Side-by-side stacked bars por timestep:
        * Supply: PV_used + BESS_discharge + Grid_import
                  (barras negativas para BESS_charge e Grid_export)
        * Demand: Load_served + Shedding
"""

import os
import json
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from opt.offgrid import OffGridMPC
from opt.ongrid import OnGridMPC   # <- usa a versão estocástica simplificada
from env.grid_env import GridEnv

# --- data paths ---
LOAD_CSV = "data/load_5min_test.csv"
PV_CSV   = "data/pv_5min_test.csv"
PARAMS_JSON = "data/parameters.json"

# Choose mode manually: "offgrid" or "ongrid"
MODE = "ongrid"   # mude para "offgrid" para operação ilhada

# REQUIRED simulation start (GridEnv alinha para o próximo timestamp disponível se necessário)
START_TS = pd.Timestamp("2009-04-26 00:00:00")
NUMBER_ITERS = 288


# ---------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------

def _try_extract_first_step(mpc, times=None):
    """Extrai o primeiro passo do MPC (dict) tentando assinaturas com e sem 'times'."""
    try:
        if times is not None:
            return mpc.extract_first_step(times)
    except Exception:
        pass
    try:
        return mpc.extract_first_step()
    except Exception as e:
        print("[WARN] MPC infeasible or extract_first_step failed:", str(e))
        return None


def run_and_log() -> pd.DataFrame:
    """Executa horizonte rolante com GridEnv + MPC; retorna a operação realizada (primeiro passo de cada janela)."""

    # Instancia o MPC conforme o modo escolhido
    if MODE.lower() == "ongrid":
        mpc = OnGridMPC(PARAMS_JSON)
    elif MODE.lower() == "offgrid":
        mpc = OffGridMPC(PARAMS_JSON)
    else:
        raise ValueError("MODE deve ser 'offgrid' ou 'ongrid'.")

    # >>> Passe o JSON BRUTO para o GridEnv (para ele ler costs/EDS aninhados corretamente)
    with open(PARAMS_JSON, "r", encoding="utf-8") as f:
        raw_params = json.load(f)

    env = GridEnv(
        params=raw_params,          # <-- JSON bruto (não use mpc.params aqui)
        load_csv=LOAD_CSV,
        pv_csv=PV_CSV,
        start_dt0=START_TS,
        n_iters=NUMBER_ITERS,
        mode=MODE,
        debug=True,                 # defina False para suprimir logs por passo do env
        clamp_soc_pct=True,
    )

    # Loop MPC (horizonte rolante)
    while not env.done():
        times, forecasts = env.get_window()
        if times is None:
            print("End of data — stopping.")
            break

        # Constrói e resolve o MPC usando o estado medido do ambiente
        if MODE.lower() == "ongrid":
            # OnGridMPC recebe o datetime inicial e monta a malha/contingências internamente
            mpc.build(start_dt=env.start_dt0, forecasts=forecasts, E_hat_kwh=env.E_meas)
        else:  # offgrid inalterado
            mpc.build(times, forecasts, E_hat_kwh=env.E_meas)

        mpc.solve("gurobi", tee=False)

        # Extrai decisões do primeiro passo (robusto a inviabilidade)
        step0 = _try_extract_first_step(mpc, times=times if MODE.lower()=="offgrid" else None)

        # Defaults
        P_bess = 0.0
        X_L = None
        X_PV = None
        obj = None
        P_grid_in_mpc = None
        P_grid_out_mpc = None

        if step0 is None:
            print("[FALLBACK] Applying safe step: P_bess=0, X_L/X_PV=None; GridEnv will close the balance.")
        else:
            P_bess = float(step0.get("P_bess_kw", 0.0))
            X_L    = float(step0.get("X_L", 0.0))
            X_PV   = float(step0.get("X_PV", 0.0))
            obj    = float(step0.get("obj", 0.0))
            # Caso OnGridMPC tenha incluído rede no extrator
            if "P_grid_in_kw" in step0:
                P_grid_in_mpc  = float(step0.get("P_grid_in_kw", 0.0))
            if "P_grid_out_kw" in step0:
                P_grid_out_mpc = float(step0.get("P_grid_out_kw", 0.0))

        # Aplica no ambiente (o GridEnv assegura limites JSON e fecha balanço/import/export/custos)
        row, done = env.step(P_bess_kw=P_bess, X_L=X_L, X_PV=X_PV, obj=obj)

        # ---------- Impressão amigável de REDE (MPC vs aplicado) ----------
        ts = pd.Timestamp(row["timestamp"])
        gin_applied  = float(row.get("P_grid_in_kw", 0.0))
        gout_applied = float(row.get("P_grid_out_kw", 0.0))
        if MODE.lower() == "ongrid":
            if P_grid_in_mpc is not None or P_grid_out_mpc is not None:
                print(
                    f"[{ts}] Grid (MPC → ENV): "
                    f"in {P_grid_in_mpc if P_grid_in_mpc is not None else 0.0:.3f} → {gin_applied:.3f} kW, "
                    f"out {P_grid_out_mpc if P_grid_out_mpc is not None else 0.0:.3f} → {gout_applied:.3f} kW"
                )
            else:
                print(f"[{ts}] Grid (ENV applied): in {gin_applied:.3f} kW, out {gout_applied:.3f} kW")

        if done:
            break

    # Operação realizada final
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

def _bar_width_and_offsets(index) -> Tuple[float, float]:
    """Compute bar width (in days) and lateral offset for side-by-side bars."""
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
    e_nom_kwh: Optional[float] = None,
):
    """
    Figure 1 — Supply (barras) vs Target (linha) + SoC.
    Barras positivas (empilhadas): PV·(1−X_PV), BESS(discharge), Grid import
    Barra negativa:                Grid export [neg]
    Linha alvo:                    LOAD·(1−X_L) + BESS(charge)
    """
    if df.empty:
        print("Nada a plotar: DataFrame vazio.")
        return

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Séries sempre presentes (mesmo zeradas) -> legenda estável
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

    # ---------------- TOP: supply (bars) vs target (line) ----------------
    h_pv   = ax0.bar(df.index, pv_used,  width=bar_width_days, label="PV·(1−X_PV)", alpha=0.65)
    h_bdis = ax0.bar(df.index, bess_dis, bottom=pv_used, width=bar_width_days, label="BESS (discharge)", alpha=0.65)
    bottom_pos = pv_used + bess_dis
    h_gin  = ax0.bar(df.index, grid_in,  bottom=bottom_pos, width=bar_width_days, label="Grid import", alpha=0.65)

    # exportação como barra negativa
    h_gout = ax0.bar(df.index, -grid_out, width=bar_width_days, label="Grid export [neg]", alpha=0.45)

    # linha alvo
    (h_line,) = ax0.plot(df.index, target_kw, linewidth=2.0, label="LOAD·(1−X_L) + BESS (charge)", zorder=3)

    ax0.axhline(0.0, linewidth=1.0, alpha=0.7)
    ax0.set_ylabel("Power [kW]")
    ax0.set_title(f"Supply (bars) vs Target (line) — max |residual| = {float(residual):.4f} kW")
    ax0.grid(True, axis='y', linestyle="--", alpha=0.6)

    handles = [h_pv, h_bdis, h_gin, h_gout, h_line]
    labels  = [h.get_label() if hasattr(h, "get_label") else "?" for h in handles]
    ax0.legend(handles, labels, ncol=2, fontsize=9, frameon=False)

    # ---------------- BOTTOM: SoC [%] ----------------
    if "SoC_pct" in df.columns:
        soc_series = pd.to_numeric(df["SoC_pct"], errors="coerce")
        if soc_series.max(skipna=True) is not None and soc_series.max(skipna=True) > 1000:
            soc_series = soc_series / 100.0
    elif "E_kwh" in df.columns and e_nom_kwh:
        soc_series = 100.0 * (pd.to_numeric(df["E_kwh"], errors="coerce") / max(e_nom_kwh, 1e-9))
        print("[plot_identity_and_soc] Using SoC derived from E_kwh/E_nom_kwh.")
    else:
        soc_series = pd.Series(index=df.index, dtype=float)

    ax1.plot(df.index, soc_series, linewidth=2.0, label="SoC", zorder=3)
    smin = soc_series.min(skipna=True); smax = soc_series.max(skipna=True)
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
    """Figure 2 — Supply vs Demand em barras lado a lado (inclui rede)."""
    if df.empty:
        print("Nada a plotar: DataFrame vazio.")
        return

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Largura e offsets
    bar_width_days, offset_days = _bar_width_and_offsets(df.index)
    idx_supply = df.index - pd.to_timedelta(offset_days, unit="D")
    idx_demand = df.index + pd.to_timedelta(offset_days, unit="D")

    # Supply (sempre presentes para legenda estável)
    z = pd.Series(0.0, index=df.index)
    pv_used      = df.get("PV_used_kw", z).clip(lower=0.0)
    bess_dis     = df.get("P_bess_discharge_kw", z).clip(lower=0.0)
    bess_chg_neg = -df.get("P_bess_charge_mag_kw", z).clip(lower=0.0)
    grid_imp     = df.get("P_grid_in_kw", z).clip(lower=0.0)
    grid_exp_neg = -df.get("P_grid_out_kw", z).clip(lower=0.0)

    # Demand
    load_srv  = df.get("Load_served_kw", z).clip(lower=0.0)
    shed      = df.get("Shedding_kw", z).clip(lower=0.0)

    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)

    # Supply empilhado positivo
    h_sup1 = ax.bar(idx_supply, pv_used, width=bar_width_days, alpha=0.6, label="Supply: PV used")
    h_sup2 = ax.bar(idx_supply, bess_dis, bottom=pv_used, width=bar_width_days, alpha=0.6, label="Supply: BESS (discharge)")
    h_sup3 = ax.bar(idx_supply, grid_imp, bottom=(pv_used + bess_dis), width=bar_width_days, alpha=0.6, label="Supply: Grid import")

    # Supply negativo (cargas/saídas)
    h_sup4 = ax.bar(idx_supply, bess_chg_neg, width=bar_width_days, alpha=0.4, label="Supply: BESS (charge) [neg]")
    h_sup5 = ax.bar(idx_supply, grid_exp_neg, width=bar_width_days, alpha=0.4, label="Supply: Grid export [neg]")

    # Demand empilhado
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
