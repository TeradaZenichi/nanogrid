import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_load_5min = pd.read_csv("data/load_5min.csv", sep=";", parse_dates=["datetime"])

# Criar um boxplot ajustado para garantir a distribuição correta ao longo de 24 horas
plt.figure(figsize=(12, 6))

# Criar um boxplot agrupando os dados por hora e minuto do dia
df_load_5min["hour_minute"] = df_load_5min["datetime"].dt.strftime("%H:%M")
df_load_5min.boxplot(column="P", by="hour_minute", grid=False, showfliers=False, patch_artist=True)

# Ajustar rótulos e layout
plt.xlabel("Timestamp")
plt.ylabel("Normalized power")
plt.title("")
plt.suptitle("")

# Definir ticks para mostrar apenas algumas marcações ao longo do tempo (a cada 2 horas)
tick_positions = np.linspace(0, len(df_load_5min["hour_minute"].unique()) - 1, 13, dtype=int)
tick_labels = [df_load_5min["hour_minute"].unique()[i] for i in tick_positions]
plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45)

plt.grid(True, linestyle="--", alpha=0.6)

#use latex font without having latex installed another option instead of serif and sans-serif is monospace
plt.rcParams['font.family'] = 'serif'

plt.savefig("analysis/load/load_5min_boxplot.pdf", dpi=300)
