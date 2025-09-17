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

MPC_SOLUTIONS_DIR = "outputs/mpc_solutions"
OUTPUT_DIR = "outputs/3d_plots"

# Cria arquivos de teste para a visualização.
# Comente ou remova esta chamada se você já tiver múltiplos arquivos de plano.
create_dummy_plans(
    os.path.join(MPC_SOLUTIONS_DIR, "mpc_plan_20090426T000000.json"),
    MPC_SOLUTIONS_DIR
)

# Chama a função principal que agora gera os 5 gráficos baseados apenas nos JSONs
create_mpc_evolution_3d_plots(MPC_SOLUTIONS_DIR, OUTPUT_DIR)