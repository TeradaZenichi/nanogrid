import pandas as pd
import matplotlib.pyplot as plt

# Carregar o CSV
file_path = 'data/load.csv'  # Atualize o caminho conforme necessário
data = pd.read_csv(file_path, sep=';')

# Converter 'datetime' para datetime e criar coluna 'hour'
data['datetime'] = pd.to_datetime(data['datetime'])
data['hour'] = data['datetime'].dt.hour

# Estatísticas descritivas para 'Global_active_power'
stats = data['Global_active_power'].describe()
print("Estatísticas descritivas de Global_active_power:")
print(stats)

# Gráfico de caixa (Boxplot) por hora
plt.figure(figsize=(12, 6))
data.boxplot(column='Global_active_power', by='hour', grid=False, showfliers=True)
plt.title("Boxplot de Global_active_power por Hora")
plt.suptitle("")  # Remove o título automático do pandas
plt.xlabel("Hora do Dia")
plt.ylabel("Potência Ativa Global (Global_active_power)")
plt.xticks(range(0, 24))
plt.show()

print('Fim do script.')