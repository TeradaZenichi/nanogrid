# -*- coding: utf-8 -*-
"""
main_offgrid.py
Executa o MPC Off-Grid, salva CSV da operação real e plota:
  (Figura 1)
    - (Topo)  PV·(1−X_PV) + BESS (descarreg.) (+ EDS) = Load·(1−X_L)  [barras empilhadas vs linha]
    - (Abaixo) Estado de carga (SoC) em %
  (Figura 2)
    - Balanço de potência com duas barras empilhadas lado a lado por timestep:
        * Supply: PV_used + BESS_discharge + (EDS)  e barra negativa BESS_charge
        * Demand: Load_served + Shedding

Requisitos:
- opt/offgrid.py (classe OffGridMPC)
- opt/utils.py   (funções utilitárias)
- matplotlib, pandas
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from opt.offgrid import OffGridMPC
from opt.utils import (
    build_dt_vector, build_time_grid,
    load_series_scaled, slice_forecasts,
)

LOAD_CSV = "data/load_5min_test.csv"
PV_CSV   = "data/pv_5min_test.csv"


def run_and_log():
    """Roda o MPC Off-Grid e retorna DataFrame com a OPERAÇÃO REAL aplicada a cada passo."""
    mpc = OffGridMPC("data/parameters.json")
    p = mpc.params

    # grade temporal base do horizonte do MPC
    dt_min = build_dt_vector(
        p["horizon_hours"], p["fine_hours"],
        p["timestep_1_min"], p["timestep_2_min"]
    )

    # séries (kW) já escaladas
    load_kw_s, pv_kw_s = load_series_scaled(p, LOAD_CSV, PV_CSV)

    # início comum
    common_index = load_kw_s.index.intersection(pv_kw_s.index)
    if len(common_index) == 0:
        raise RuntimeError("Séries de carga e PV não têm interseção de timestamps.")
    start_dt0 = pd.Timestamp(common_index[0]).to_pydatetime()

    # SoC medido inicial
    E_meas = p["E_nom_kwh"] * max(0.01, p["soc_min_frac"])

    # acumula operação real (apenas o 1º passo aplicado a cada iteração)
    rows = []

    # número de iterações do rolling horizon (ex.: 24h em passos de 5min → 288)
    n_iters = 288
    for k in range(n_iters):
        times = build_time_grid(start_dt0, dt_min)

        # garante disponibilidade nas séries
        if any(t not in load_kw_s.index for t in times) or any(t not in pv_kw_s.index for t in times):
            print("Fim dos dados ou timestamps fora das séries — encerrando.")
            break

        forecasts = slice_forecasts(times, load_kw_s, pv_kw_s)

        # constroi e resolve
        mpc.build(times, forecasts, E_hat_kwh=E_meas)
        mpc.solve("gurobi", tee=False)

        # aplica apenas o primeiro passo (progressão real)
        t0 = times[0]
        step0 = mpc.extract_first_step(times)

        # valores de base nesse passo
        load0 = float(forecasts['load_kw'][t0])
        pv0   = float(forecasts['pv_kw'][t0])

        # decisões/variáveis do primeiro passo
        XL   = float(step0["X_L"])
        XPV  = float(step0["X_PV"])
        Pb   = float(step0["P_bess_kw"])  # P_bess = P_dis - P_ch
        Pch  = float(step0["P_ch_kw"])
        Pdis = float(step0["P_dis_kw"])
        E0   = float(step0["E_kwh"])
        obj0 = float(step0["obj"])
        Peds = float(step0.get("P_EDS_kw", 0.0))  # opcional (gerador local)

        # grandezas para a identidade:
        served_load = load0 * (1.0 - XL)      # Load(1 - X_L)
        pv_used     = pv0   * (1.0 - XPV)     # PV(1 - X_PV)
        bess_discharge = max(Pb, 0.0)         # kW >= 0
        bess_charge_mag = max(-Pb, 0.0)       # kW >= 0

        # perdas/recusas
        shedding = load0 - served_load        # = load0 * XL
        curtail  = pv0   - pv_used            # = pv0 * XPV

        # checagem / residual
        total_supply = pv_used + bess_discharge + Peds
        residual = served_load - total_supply  # deve ser ~0

        # SoC em %
        soc_pct = 100.0 * (E0 / p["E_nom_kwh"]) if p["E_nom_kwh"] > 0 else 0.0

        rows.append({
            "timestamp": pd.Timestamp(t0),
            # básicos
            "Load_kw": load0,
            "PV_kw": pv0,
            # identidade (lado direito)
            "Load_served_kw": served_load,
            "Shedding_kw": shedding,
            # identidade (lado esquerdo - contribuições)
            "PV_used_kw": pv_used,
            "Curtailment_kw": curtail,
            "P_bess_discharge_kw": bess_discharge,
            "P_bess_charge_mag_kw": bess_charge_mag,
            "P_EDS_kw": Peds,
            "Total_supply_kw": total_supply,
            "Residual_kw": residual,
            # demais úteis
            "P_bess_kw": Pb,
            "P_ch_kw": Pch,
            "P_dis_kw": Pdis,
            "E_kwh": E0,
            "SoC_pct": soc_pct,
            "obj": obj0
        })

        # evolui SoC medido e desliza o horizonte
        dt_h0 = (times[1] - times[0]).total_seconds() / 3600.0
        E_meas = E0 + dt_h0 * (p["eta_c"] * Pch - (1.0 / p["eta_d"]) * Pdis)
        start_dt0 = times[1]

    # DataFrame final da operação real
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df


def save_operation_csv(df: pd.DataFrame, path="outputs/operation_real.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)
    print(f"[csv] operação real salva em: {path}")


def _bar_width_and_offsets(index):
    """Calcula largura da barra (em dias) e offset lateral para barras lado a lado."""
    if len(index) > 1:
        delta = (index[1] - index[0]).total_seconds()
    else:
        delta = 300.0  # fallback: 5 min
    bar_width_days = 0.8 * (delta / 86400.0)
    offset_days = 0.5 * bar_width_days
    return bar_width_days, offset_days


def plot_identity_and_soc(df: pd.DataFrame, save_path="outputs/operation_plot.png"):
    """Figura 1 — Identidade PV+BESS(+EDS)=Load_served + SoC[%]."""
    if df.empty:
        print("Nada a plotar: DataFrame vazio.")
        return

    bar_width_days, _ = _bar_width_and_offsets(df.index)

    fig, (ax0, ax1) = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.2], "hspace": 0.15}
    )

    # --- Subplot superior: Identidade de potência ---
    pv_used = df["PV_used_kw"].clip(lower=0.0)
    bess_dis = df["P_bess_discharge_kw"].clip(lower=0.0)
    eds = df["P_EDS_kw"].clip(lower=0.0) if "P_EDS_kw" in df.columns else None

    ax0.bar(df.index, pv_used, width=bar_width_days, label="PV·(1−X_PV)", alpha=0.6)
    ax0.bar(df.index, bess_dis, bottom=pv_used, width=bar_width_days, label="BESS (descarreg.)", alpha=0.6)
    if eds is not None and eds.sum() > 0:
        ax0.bar(df.index, eds, bottom=(pv_used + bess_dis), width=bar_width_days, label="EDS", alpha=0.6)

    ax0.plot(df.index, df["Load_served_kw"], linewidth=1.8, label="Load·(1−X_L) (linha-alvo)")
    if "Residual_kw" in df.columns:
        ax0.plot(df.index, df["Residual_kw"], linewidth=1.0, alpha=0.5, label="Residual (alvo − soma)")

    ax0.set_ylabel("Potência [kW]")
    ax0.grid(True, axis='y', linestyle="--", alpha=0.6)
    ax0.legend(ncol=3, fontsize=9, frameon=False)

    # --- Subplot inferior: SoC [%] ---
    ax1.plot(df.index, df["SoC_pct"], linewidth=1.8, label="SoC")
    ax1.set_ylabel("SoC [%]")
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis='y', linestyle="--", alpha=0.6)

    # eixo x
    ax1.set_xlabel("Tempo")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"[fig] gráfico salvo em: {save_path}")
    plt.close(fig)


def plot_power_balance_stacked(df: pd.DataFrame, save_path="outputs/operation_balance.png"):
    """Figura 2 — Balanço de potência com duas barras empilhadas lado a lado (Supply vs Demand)."""
    if df.empty:
        print("Nada a plotar: DataFrame vazio.")
        return

    # garante índice datetime
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # largura e deslocamento lateral
    if len(df.index) > 1:
        delta = (df.index[1] - df.index[0]).total_seconds()
    else:
        delta = 300.0  # fallback 5 min
    bar_width_days = 0.8 * (delta / 86400.0)
    offset_days = 0.5 * bar_width_days

    # índices deslocados (lado a lado)
    idx_supply = df.index - pd.to_timedelta(offset_days, unit="D")
    idx_demand = df.index + pd.to_timedelta(offset_days, unit="D")  # <— corrigido

    # SUPPLY stacks (acima de zero): PV_used + BESS_discharge + EDS
    pv_used   = df["PV_used_kw"].clip(lower=0.0)
    bess_dis  = df["P_bess_discharge_kw"].clip(lower=0.0)
    eds       = df["P_EDS_kw"].clip(lower=0.0) if "P_EDS_kw" in df.columns else None
    # SUPPLY negativo (abaixo de zero): BESS carregando (magnitude positiva → barra negativa)
    bess_chg_neg = -df["P_bess_charge_mag_kw"].clip(lower=0.0) if "P_bess_charge_mag_kw" in df.columns else None

    # DEMAND stacks: Load_served + Shedding
    load_srv  = df["Load_served_kw"].clip(lower=0.0)
    shed      = df["Shedding_kw"].clip(lower=0.0) if "Shedding_kw" in df.columns else None

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)

    # Supply positivo empilhado
    ax.bar(idx_supply, pv_used, width=bar_width_days, alpha=0.6, label="Supply: PV usado")
    ax.bar(idx_supply, bess_dis, bottom=pv_used, width=bar_width_days, alpha=0.6, label="Supply: BESS (descarreg.)")
    if eds is not None and eds.sum() > 0:
        ax.bar(idx_supply, eds, bottom=(pv_used + bess_dis), width=bar_width_days, alpha=0.6, label="Supply: EDS")

    # Supply negativo (BESS carregando)
    if bess_chg_neg is not None and (bess_chg_neg != 0).any():
        ax.bar(idx_supply, bess_chg_neg, width=bar_width_days, alpha=0.4, label="Supply: BESS (carreg.) [neg]")

    # Demand empilhado
    ax.bar(idx_demand, load_srv, width=bar_width_days, alpha=0.6, label="Demand: Load atendida")
    if shed is not None and shed.sum() > 0:
        ax.bar(idx_demand, shed, bottom=load_srv, width=bar_width_days, alpha=0.6, label="Demand: Shedding")

    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel("Potência [kW]")
    ax.set_title("Balanço de potência — Supply vs Demand (barras empilhadas)")
    ax.grid(True, axis='y', linestyle="--", alpha=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.legend(ncol=3, fontsize=9, frameon=False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"[fig] gráfico salvo em: {save_path}")
    plt.close(fig)



def main():
    df = run_and_log()
    save_operation_csv(df, path="outputs/operation_real.csv")
    plot_identity_and_soc(df, save_path="outputs/operation_plot.png")
    plot_power_balance_stacked(df, save_path="outputs/operation_balance.png")


if __name__ == "__main__":
    main()
