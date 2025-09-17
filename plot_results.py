from opt.plot_results import plot_custom_dispatch_with_soc


import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
import re

def create_mpc_evolution_3d_plots(mpc_plans_dir: str, output_dir: str):
    """
    Lê todos os planos MPC, extrai o cenário base e gera gráficos 3D
    da evolução das decisões do MPC (Bateria, Rede, SoC, Cortes).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_plans = []
    
    # 1. Encontrar e processar todos os arquivos JSON
    print(f"🔎 Varrendo o diretório: {mpc_plans_dir}")
    json_files = sorted([f for f in os.listdir(mpc_plans_dir) if f.startswith("mpc_plan_") and f.endswith(".json")])

    if not json_files:
        print("❌ Erro: Nenhum arquivo de plano do MPC encontrado no diretório.")
        return

    fname_regex = re.compile(r"mpc_plan_(\d{8}T\d{6})\.json")

    for fname in json_files:
        match = fname_regex.match(fname)
        if not match: continue
            
        decision_time = pd.to_datetime(match.group(1), format='%Y%m%dT%H%M%S')
        
        try:
            with open(os.path.join(mpc_plans_dir, fname), 'r') as f:
                data = json.load(f)
            
            base_case = data.get("scenarios", {}).get("c0_base_case")
            if base_case:
                df = pd.DataFrame(base_case)
                df['decision_time'] = decision_time
                df['plan_timestamp'] = pd.to_datetime(df['timestamp'])
                all_plans.append(df)
        except Exception as e:
            print(f"⚠️ Aviso: Falha ao processar o arquivo {fname}. Erro: {e}")

    if not all_plans:
        print("❌ Erro: Nenhum cenário 'c0_base_case' válido foi encontrado nos arquivos.")
        return

    full_df = pd.concat(all_plans, ignore_index=True).fillna(0)

    # 2. Preparar todas as colunas necessárias para os plots, usando apenas dados do JSON
    full_df['horizon_hours'] = (full_df['plan_timestamp'] - full_df['decision_time']).dt.total_seconds() / 3600
    e_nom_kwh = full_df['E_kwh'].max()
    full_df['SoC_pct'] = 100 * full_df['E_kwh'] / max(e_nom_kwh, 1e-9)
    full_df['P_grid_net_kw'] = full_df['P_grid_in_kw'] - full_df['P_grid_out_kw']

    # Dicionário para configurar os plots
    plots_to_generate = {
        'P_bess_kw': {'cmap': 'viridis', 'title': 'MPC Planned Battery Power Evolution', 'zlabel': 'Battery Power [kW]'},
        'SoC_pct': {'cmap': 'plasma', 'title': 'MPC Planned SoC Evolution', 'zlabel': 'SoC [%]'},
        'P_grid_net_kw': {'cmap': 'coolwarm', 'title': 'MPC Planned Net Grid Power Evolution', 'zlabel': 'Net Grid Power [kW]'},
        'X_L': {'cmap': 'magma', 'title': 'MPC Planned Load Shedding Fraction Evolution', 'zlabel': 'Load Shedding Fraction'},
        'X_PV': {'cmap': 'cividis', 'title': 'MPC Planned PV Curtailment Fraction Evolution', 'zlabel': 'PV Curtailment Fraction'},
    }

    for var_name, config in plots_to_generate.items():
        print(f"⚙️ Gerando gráfico 3D para: {var_name}")
        
        grid = full_df.pivot_table(index='decision_time', columns='horizon_hours', values=var_name).interpolate(axis=1)
        
        if grid.empty:
            print(f"⚠️ Aviso: Não foi possível gerar a grade para '{var_name}'. Pulando.")
            continue

        X, Y = np.meshgrid(grid.columns, grid.index)
        Z = grid.values

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, mdates.date2num(Y), Z, cmap=config['cmap'], edgecolor='none', rstride=1, cstride=1)

        ax.set_title(config['title'], fontsize=16)
        ax.set_xlabel('Prediction Horizon [hours]', labelpad=10)
        ax.set_ylabel('Decision Time', labelpad=10)
        ax.set_zlabel(config['zlabel'], labelpad=10)
        ax.yaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1).set_label(config['zlabel'])
        ax.view_init(elev=20, azim=-120)
        ax.invert_yaxis()
        
        if 'SoC' in var_name or 'X_' in var_name: # Limita o eixo Z para SoC e frações
            ax.set_zlim(0, 100 if 'SoC' in var_name else 1.0)

        save_path = os.path.join(output_dir, f"3d_plot_{var_name}_evolution.png")
        plt.savefig(save_path, dpi=150)
        print(f"✅ Gráfico salvo em: {save_path}")
        plt.close(fig)

# --- Exemplo de Uso ---
def create_dummy_plans(base_file, output_dir, num_files=5, time_step_min=15):
    if not os.path.exists(base_file): return
    with open(base_file, 'r') as f:
        base_data = json.load(f)
    
    match = re.search(r"mpc_plan_(\d{8}T\d{6})\.json", os.path.basename(base_file))
    if not match: return
    base_dt = pd.to_datetime(match.group(1), format='%Y%m%dT%H%M%S')

    for i in range(num_files):
        new_dt = base_dt + pd.Timedelta(minutes=i * time_step_min)
        new_data = base_data.copy()
        
        for j, step in enumerate(new_data['scenarios']['c0_base_case']):
            step['P_bess_kw'] *= (1 - i*0.02 + j*0.001)
            step['E_kwh'] *= (1 - i*0.01)
            step['X_L'] += i*0.005
            step['X_PV'] += i*0.005

        fname = f"mpc_plan_{new_dt.strftime('%Y%m%dT%H%M%S')}.json"
        with open(os.path.join(output_dir, fname), 'w') as f:
            json.dump(new_data, f, indent=4)


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





if __name__ == "__main__":

    # 3D plot
    MPC_SOLUTIONS_DIR = "outputs/mpc_solutions"
    OUTPUT_DIR = "outputs/3d_plots"

    # Plot all scenarios of a single timestamp
    TARGET_DATETIME = "2009-05-01 04:15:00"
    MPC_SOLUTIONS_DIR = "outputs/mpc_solutions"

    # generate_plots_for_timestamp(TARGET_DATETIME, MPC_SOLUTIONS_DIR)
    # create_mpc_evolution_3d_plots(MPC_SOLUTIONS_DIR, OUTPUT_DIR)

    FROM = pd.Timestamp("2009-05-01 00:00:00")
    TO = pd.Timestamp("2009-05-03 00:00:00")
    df = pd.read_csv("outputs/operation_real.csv", index_col=0, parse_dates=True)
    plot_custom_dispatch_with_soc(df, start_dt=FROM, end_dt=TO)