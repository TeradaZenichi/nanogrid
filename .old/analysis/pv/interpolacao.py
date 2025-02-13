import pandas as pd

# Carregar o arquivo CSV de PV de 5 minutos
file_pv_5min_path = "data/pv_5min.csv"  # Ajuste o caminho conforme necessário
df_pv = pd.read_csv(file_pv_5min_path, sep=",", parse_dates=["datetime"])

# Remover duplicatas mantendo a média dos valores repetidos
df_pv = df_pv.groupby("datetime", as_index=False).mean()

# Definir o índice como DateTime para facilitar a reamostragem
df_pv.set_index("datetime", inplace=True)

# Ordenar os dados para garantir interpolação correta
df_pv = df_pv.sort_index()

# Aplicar interpolação para 1 minuto
df_pv_1min = df_pv.resample("1T").interpolate(method="linear")

# Resetar o índice para salvar corretamente
df_pv_1min = df_pv_1min.reset_index()

# Caminho para salvar o novo arquivo interpolado
pv_1min_path = "data/PV_1min_interpolated.csv"

# Salvar o arquivo CSV interpolado
df_pv_1min.to_csv(pv_1min_path, sep=";", index=False)

# Mensagem de confirmação
print(f"Arquivo gerado: {pv_1min_path}")
