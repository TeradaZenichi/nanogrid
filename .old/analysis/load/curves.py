
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_load_5min = pd.read_csv("data/load_5min.csv", sep=";", parse_dates=["datetime"])

# Converter datetime para apenas horas e minutos como string para plotagem mais eficiente
df_load_5min["time_str"] = df_load_5min["datetime"].dt.strftime("%H:%M")

# Reduzir a quantidade de dias para evitar sobrecarga no gráfico (exemplo: 10 dias aleatórios)
unique_dates = df_load_5min["datetime"].dt.date.unique()
# sampled_dates = np.random.choice(unique_dates, min(10, len(unique_dates)), replace=False)

# Criar um gráfico sobrepondo os dias
plt.figure(figsize=(12, 6))
for date in unique_dates:
    daily_data = df_load_5min[df_load_5min["datetime"].dt.date == date]
    plt.plot(daily_data["datetime"].dt.hour + daily_data["datetime"].dt.minute / 60, daily_data["P"], alpha=0.2, label=str(date), color="grey", linewidth=0.5)

plt.xlabel("Timestamp")
plt.ylabel("Normalized Power")
# plt.title("Perfil de Carga - Sobreposição de Dias")
plt.xticks(ticks=np.arange(0, 25, 3), labels=[f"{i}:00" for i in range(0, 25, 3)])
plt.xlim(0, 24)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("analysis/load/load_5min_curves.pdf", dpi=300)

