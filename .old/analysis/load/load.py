import pandas as pd

# Carregar o arquivo CSV original
file_path = "data/household_power_consumption_5min.csv"
df = pd.read_csv(file_path, sep=";", parse_dates=["datetime"])

# Criar o primeiro arquivo (load_5min.csv) com apenas P e Q
df_load_5min = df[["datetime", "Global_active_power", "Global_reactive_power"]].rename(
    columns={"Global_active_power": "P", "Global_reactive_power": "Q"}
)

# Salvar o arquivo
load_5min_path = "data/load_5min.csv"
df_load_5min.to_csv(load_5min_path, sep=";", index=False)

# Criar o segundo arquivo (load_1min.csv) com interpolação para 1 minuto aplique um round para 4 casas decimais
df_load_5min.set_index("datetime", inplace=True)
df_load_1min = df_load_5min.resample("1T").interpolate()

# Salvar o arquivo interpolado
load_1min_path = "data/load_1min.csv"
df_load_1min.to_csv(load_1min_path, sep=";")


