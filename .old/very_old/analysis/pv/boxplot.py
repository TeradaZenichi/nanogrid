import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregar os dados
df_pv_5min = pd.read_csv("data/pv_5min.csv", sep=',', parse_dates=["datetime"])
df_pv_5min["datetime"] = pd.to_datetime(df_pv_5min["datetime"], errors="coerce")

# Criar um índice de tempo de 5 em 5 minutos para garantir consistência
df_pv_5min["time_index"] = (df_pv_5min["datetime"].dt.hour * 60 + df_pv_5min["datetime"].dt.minute) // 5

# Criar a figura
plt.figure(figsize=(12, 6))

# Criar boxplot agrupando os dados pelo índice de tempo (intervalos de 5 minutos)
df_pv_5min.boxplot(column="Ppower", by="time_index", grid=False, showfliers=False, patch_artist=True)

# Ajustar rótulos e layout
plt.xlabel("Timestamp")
plt.ylabel("Normalized Power")
plt.title("")
plt.suptitle("")

# Definir rótulos do eixo X corretamente, cobrindo de 00:00 até 23:55
tick_positions = np.arange(0, 288, 24)  # Mostrar a cada 2 horas (24 amostras * 5 min = 120 min)
tick_labels = [f"{h:02d}:00" for h in range(0, 24, 2)]  # Gera '00:00', '02:00', '04:00', ..., '22:00'
plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45)

plt.grid(True, linestyle="--", alpha=0.6)

# Configurar fonte no estilo LaTeX sem precisar instalar LaTeX
plt.rcParams['font.family'] = 'serif'

# Salvar o gráfico
plt.savefig("analysis/pv/pv_5min_boxplot.pdf", dpi=300)
plt.show()
