import pandas as pd
import numpy as np

# Carregar os dados
file_path = "data/pv_5min.csv"  # Ajuste o caminho conforme necessário
df = pd.read_csv(file_path)

# Converter a coluna datetime para formato de data-hora, tratando erros como NaT
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

# Criar coluna de data para análise por dia
df["date"] = df["datetime"].dt.date

# Identificar dias a serem removidos
days_to_remove = []

for date, group in df.groupby("date"):
    unique_values = group["Ppower"].nunique()
    all_zero = (group["Ppower"] == 0).all()
    constant_values = unique_values == 1

    if all_zero or constant_values:
        days_to_remove.append(date)

# Filtrar os dados, removendo os dias indesejados
df_filtered = df[~df["date"].isin(days_to_remove)].copy()

# Garantir que os dados finais tenham apenas as colunas necessárias
df_filtered = df_filtered[["datetime", "Ppower"]]

# Salvar o novo CSV
output_file = "data/pv_5min_cleaned.csv"
df_filtered.to_csv(output_file, index=False)

print(f"Arquivo limpo salvo em: {output_file}")
