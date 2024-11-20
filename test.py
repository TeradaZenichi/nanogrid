import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import json

# Classe de Configuração
class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        """Carrega as configurações do arquivo JSON e define como atributos."""
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        for key, value in self.config.items():
            setattr(self, key, value)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Caminhos de arquivos
config_path = 'data/load.json'
results_dir = "Results/load"
model_path = os.path.join(results_dir, "model_sliding.pth")

# Carregar Configurações
config = Config(config_path)

# Carregar dados normalizados
file_path = 'data/load.csv'
data = pd.read_csv(file_path, sep=';')

data['power'] = pd.to_numeric(data['power'], errors='coerce')
data = data.dropna()

scaler = MinMaxScaler()
data['power'] = scaler.fit_transform(data[['power']])
series = data['power'].values

# Criar sequências para entrada de teste
def create_test_sequences(data, n_input):
    X = []
    for i in range(len(data) - n_input):
        X.append(data[i:i + n_input])
    return np.array(X)

X_test_seq = create_test_sequences(series, config.N_input)
X_test_seq = torch.tensor(X_test_seq, dtype=torch.float32).to(config.device)

# Modelo LSTM
class SlidingWindowLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(SlidingWindowLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Carregar modelo
model = SlidingWindowLSTM(
    input_dim=config.input_dim,
    hidden_dim=config.hidden_dim,
    output_dim=1,
    num_layers=config.num_layers
).to(config.device)

model.load_state_dict(torch.load(model_path))
model.eval()

# Função para previsão com janela deslizante
def forecast_with_sliding_window(model, initial_window, n_forecast, device):
    predictions = []
    current_window = initial_window.copy()

    for _ in range(n_forecast):
        input_tensor = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        next_step = model(input_tensor).item()
        predictions.append(next_step)
        current_window = np.roll(current_window, -1)
        current_window[-1] = next_step

    return predictions

# Tamanho da janela de previsão
n_forecast = 30

# Usar os últimos N_input timesteps como janela inicial
initial_window = series[-config.N_input:]

# Fazer previsões
predictions = forecast_with_sliding_window(model, initial_window, n_forecast, config.device)

# Reverter normalização
predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Visualizar resultados
plt.figure(figsize=(10, 6))
plt.plot(range(len(series)), scaler.inverse_transform(series.reshape(-1, 1)), label="Valores Reais")
plt.plot(range(len(series), len(series) + n_forecast), predictions_rescaled.flatten(), label="Previsões", color='orange')
plt.legend()
plt.title(f"Previsão com Janela Deslizante ({n_forecast} Timesteps)")
plt.xlabel("Timesteps")
plt.ylabel("Potência")
plt.grid()

# Salvar figura
forecast_plot_path = os.path.join(results_dir, f"forecast_plot_{n_forecast}_steps.png")
plt.savefig(forecast_plot_path)
plt.show()

print(f"Figura de previsão salva em: {forecast_plot_path}")
