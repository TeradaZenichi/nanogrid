# -*- coding: utf-8 -*-
"""
Gera gráficos dos cenários MPC lendo APENAS os dados do JSON do plano.
Para cada timestamp, o JSON já fornece:
- P_load_kw (carga)
- P_pv_kw   (PV)
- P_bess_kw (sinal: charge > 0)
- P_grid_in_kw, P_grid_out_kw
- X_L, X_PV (frações de shed/curtail, se existirem)
- E_kwh (energia instantânea da BESS; usada para SoC)

Saída:
- PNG por cenário em outputs/mpc_plots/
- 1 PDF contendo todas as figuras (eixo Y sincronizado entre cenários)

Obs.: Não há mais leitura de CSV de contexto.
"""

import os
import json
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

# Garante que o diretório de saída exista
os.makedirs("outputs/mpc_plots", exist_ok=True)


def plot_mpc_scenario(
    plan_df: pd.DataFrame,
    scenario_name: str,
    decision_time: pd.Timestamp,
    save_path_png: str,
    y_lims: Optional[Tuple[float, float]] = None,
):
    """
    Plota um único cenário de um plano MPC usando APENAS as séries do próprio JSON.

    Espera-se que plan_df tenha índice datetime e colunas:
    - 'P_load_kw', 'P_pv_kw', 'P_bess_kw', 'P_grid_in_kw', 'P_grid_out_kw'
    - Opcional: 'X_L', 'X_PV', 'E_kwh'
    """
    df = plan_df.copy()

    # Larguras das barras em dias (para ax.bar com timestamps irregulares)
    time_deltas = df.index.to_series().diff().bfill()
    bar_widths_days = time_deltas.dt.total_seconds() / (24 * 3600)

    # Série "zero" alinhada ao índice para fallbacks
    z = pd.Series(0.0, index=df.index)

    # Séries principais a partir do JSON
    p_bess = df.get("P_bess_kw", z)
    p_grid = df.get("P_grid_in_kw", z) - df.get("P_grid_out_kw", z)

    p_load_total = df.get("P_load_kw", z)  # Carga total do plano
    p_pv_total = df.get("P_pv_kw", z)      # PV total do plano

    # Aplicação de shed/curtail se vierem no JSON (do contrário considera 0)
    p_load_served = p_load_total * (1 - df.get("X_L", z))
    p_pv_used = p_pv_total * (1 - df.get("X_PV", z))

    # Figura: potências (em cima) e SoC (embaixo)
    fig, (ax0, ax1) = plt.subplots(
        nrows=2, ncols=1, figsize=(16, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1}
    )

    # BESS (convenção charge>0). No gráfico usamos -p_bess para desenhar barras para baixo.
    ax0.bar(df.index, -p_bess, width=bar_widths_days, color="green", alpha=0.6,
            label="P_BESS (charge > 0)", align='edge')

    # Grid líquido (importação > 0, exportação < 0)
    ax0.plot(df.index, p_grid, color="royalblue", linewidth=2,
             label="P_Grid (import > 0)", drawstyle='steps-post')

    # Séries do plano (totais) e efetivamente servidas/usadas
    ax0.plot(df.index, p_load_total, color="black", linestyle='--', linewidth=1.5,
             label="P_Load (plan)")
    ax0.plot(df.index, p_pv_total, color="gold", linestyle='--', linewidth=1.5,
             label="P_PV (plan)")
    ax0.plot(df.index, p_load_served, color="black", marker='.', markersize=4,
             linestyle='-', label="P_Load (served)")
    ax0.plot(df.index, p_pv_used, color="gold", linewidth=2.5,
             label="P_PV (used)")

    ax0.axhline(0.0, linewidth=1.0, color='k')
    ax0.set_ylabel("Power [kW]")
    ax0.set_title(f"MPC Plan Scenario: '{scenario_name}'\nDecision Time: {decision_time.strftime('%Y-%m-%d %H:%M')}")
    ax0.grid(True, axis='y', linestyle="--", alpha=0.6)
    ax0.legend(loc='best')

    # Aplica limites Y globais se fornecidos (sincroniza entre cenários)
    if y_lims:
        ax0.set_ylim(y_lims)

    # SoC: usa E_kwh do próprio plano; a capacidade nominal é o máximo de E_kwh no horizonte
    if "E_kwh" in df.columns:
        e_nom_kwh = max(df["E_kwh"].max(), 1e-9)
        soc_series = 100.0 * df["E_kwh"] / e_nom_kwh
    else:
        soc_series = pd.Series(0.0, index=df.index)  # fallback visível

    ax1.plot(df.index, soc_series, linewidth=2.5, label="SoC",
             zorder=3, color="teal", drawstyle='steps-post')
    ax1.set_ylabel("SoC [%]")
    ax1.grid(True, axis='y', linestyle="--", alpha=0.6)
    ax1.set_xlabel("Time")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.set_ylim(-5, 105)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Salva o PNG individual
    fig.savefig(save_path_png, dpi=150, bbox_inches='tight')
    print(f"✅ Gráfico do cenário '{scenario_name}' salvo em: {save_path_png}")

    return fig


def generate_plots_for_timestamp(target_dt_str: str, mpc_plans_dir: str):
    """
    Carrega o plano MPC (JSON) para o timestamp alvo e gera:
    - PNG por cenário
    - PDF agregando todos os cenários

    Agora NÃO lê mais CSV de contexto. Todas as séries (PV, LOAD, etc.)
    vêm do próprio JSON de cada cenário.
    """
    try:
        target_dt = pd.to_datetime(target_dt_str)

        # Nome do arquivo JSON do plano MPC
        fname = f"mpc_plan_{target_dt.strftime('%Y%m%dT%H%M%S')}.json"
        json_path = os.path.join(mpc_plans_dir, fname)

        if not os.path.exists(json_path):
            print(f"❌ Erro: Arquivo de plano não encontrado em: {json_path}")
            return

        print(f"📄 Carregando plano de: {json_path}")
        with open(json_path, 'r') as f:
            mpc_data = json.load(f)

        scenarios = mpc_data.get('scenarios', {})
        if not scenarios:
            print("⚠️ Nenhum cenário encontrado no arquivo JSON.")
            return

        # ------------------ Pré-processamento para limites Y globais ------------------
        print("Preprocessando cenários para sincronizar eixos Y...")
        global_y_min, global_y_max = float('inf'), float('-inf')

        # Função auxiliar para garantir Series alinhada ao índice
        def get_series(df_local: pd.DataFrame, col: str) -> pd.Series:
            return df_local[col] if col in df_local.columns else pd.Series(0.0, index=df_local.index)

        for scenario_name, scenario_plan in scenarios.items():
            plan_df = pd.DataFrame(scenario_plan)
            if 'timestamp' not in plan_df.columns:
                # Pula cenários malformados
                continue
            plan_df.index = pd.to_datetime(plan_df['timestamp'])

            z = pd.Series(0.0, index=plan_df.index)

            p_bess = get_series(plan_df, "P_bess_kw")
            p_grid = get_series(plan_df, "P_grid_in_kw") - get_series(plan_df, "P_grid_out_kw")
            p_load = get_series(plan_df, "P_load_kw")
            p_pv = get_series(plan_df, "P_pv_kw")

            all_power_series = [
                -p_bess,   # barras (charge > 0 para baixo)
                p_grid,    # grid líquido
                p_load,    # carga total
                p_pv       # PV total
            ]
            for s in all_power_series:
                if len(s) == 0:
                    continue
                global_y_min = min(global_y_min, float(s.min()))
                global_y_max = max(global_y_max, float(s.max()))

        # Se nada foi encontrado (cenários vazios), evita erro
        if not (global_y_min < global_y_max):
            global_y_min, global_y_max = -1.0, 1.0

        # Margem de 5% para melhor visualização (com salvaguarda)
        span = max(global_y_max - global_y_min, 1e-6)
        y_buffer = 0.05 * span
        y_lims = (global_y_min - y_buffer, global_y_max + y_buffer)
        print(f"Limites Y globais definidos para: ({y_lims[0]:.2f}, {y_lims[1]:.2f})")

        # ------------------ Geração do PDF com todas as figuras ------------------
        pdf_fname = f"all_scenarios_{target_dt.strftime('%Y%m%dT%H%M%S')}.pdf"
        pdf_path = os.path.join("outputs/mpc_plots", pdf_fname)
        with PdfPages(pdf_path) as pdf:
            for scenario_name, scenario_plan in scenarios.items():
                plan_df = pd.DataFrame(scenario_plan)
                if 'timestamp' not in plan_df.columns:
                    continue
                plan_df.index = pd.to_datetime(plan_df['timestamp'])

                safe_scenario_name = scenario_name.replace(":", "-").replace(" ", "_")
                plot_fname_png = f"plot_{target_dt.strftime('%Y%m%dT%H%M%S')}_{safe_scenario_name}.png"
                save_path_png = os.path.join("outputs/mpc_plots", plot_fname_png)

                fig = plot_mpc_scenario(
                    plan_df=plan_df,
                    scenario_name=scenario_name,
                    decision_time=target_dt,
                    save_path_png=save_path_png,
                    y_lims=y_lims
                )

                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        print(f"\n📄 PDF com todos os cenários salvo em: {pdf_path}")

    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado: {e}")


# --- Exemplo de Uso ---
TARGET_DATETIME = "2009-05-01 12:00:00"
MPC_SOLUTIONS_DIR = "outputs/mpc_solutions"

generate_plots_for_timestamp(TARGET_DATETIME, MPC_SOLUTIONS_DIR)
