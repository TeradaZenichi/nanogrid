import pandas as pd

# Caminhos dos arquivos CSV de carga
file_load_1min_path = "data/load_1min_old.csv"  # Ajuste o caminho conforme necessário
file_load_5min_path = "data/load_5min_old.csv"  # Ajuste o caminho conforme necessário

# Carregar os datasets
df_load_1min = pd.read_csv(file_load_1min_path, sep=";", parse_dates=["datetime"])
df_load_5min = pd.read_csv(file_load_5min_path, sep=";", parse_dates=["datetime"])

# Identificar o maior valor absoluto entre P e Q nos dois datasets
max_value =df_load_1min[["P"]].max()

# Normalizar os dados dividindo pelo maior valor
df_load_1min[["P", "Q"]] /= max_value
df_load_5min[["P", "Q"]] /= max_value

# Caminhos para salvar os novos arquivos normalizados
load_1min_norm_path = "data/load_1min_normalized.csv"
load_5min_norm_path = "data/load_5min_normalized.csv"

# Salvar os arquivos CSV normalizados
df_load_1min.to_csv(load_1min_norm_path, sep=";", index=False)
df_load_5min.to_csv(load_5min_norm_path, sep=";", index=False)

# Mensagem de confirmação
print(f"Arquivos normalizados gerados:\n{load_1min_norm_path}\n{load_5min_norm_path}")
