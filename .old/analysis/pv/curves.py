import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregar os dados
df_pv_5min = pd.read_csv("data/pv_5min.csv", parse_dates=["datetime"])
df_pv_5min["datetime"] = pd.to_datetime(df_pv_5min["datetime"], errors="coerce")


# Converter datetime para apenas horas e minutos como string para plotagem mais eficiente
df_pv_5min["time"] = df_pv_5min["datetime"].dt.hour + df_pv_5min["datetime"].dt.minute / 60

# Obter dias únicos para a análise
unique_dates = df_pv_5min["datetime"].dt.date.unique()

# Criar um gráfico sobrepondo os dias
plt.figure(figsize=(12, 6))
for date in unique_dates:
    daily_data = df_pv_5min[df_pv_5min["datetime"].dt.date == date]
    plt.plot(daily_data["time"], daily_data["Ppower"], alpha=0.2, label=str(date), color="grey", linewidth=0.5)

plt.xlabel("Timestamp")
plt.ylabel("Normalized Power")
plt.xticks(ticks=np.arange(0, 25, 3), labels=[f"{i}:00" for i in range(0, 25, 3)])
plt.xlim(0, 24)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("analysis/pv/pv_5min_curves.pdf", dpi=300)