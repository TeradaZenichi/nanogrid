import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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
        # Define cada chave do JSON como atributo da instância
        for key, value in self.config.items():
            setattr(self, key, value)
        # Adicionar atributo device dinamicamente
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def save(self):
        """Salva as configurações no arquivo JSON."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def update(self, key, value):
        """Atualiza o valor de uma configuração e redefine o atributo."""
        self.config[key] = value
        setattr(self, key, value)

    def __getitem__(self, key):
        """Permite acessar configurações como um dicionário."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Permite definir configurações como um dicionário."""
        self.update(key, value)

# Criar pasta para salvar resultados
results_dir = "Results/load"
os.makedirs(results_dir, exist_ok=True)

# Carregar Configurações
config = Config('data/load.json')

# Carregar Dados
file_path = 'data/load.csv'
data = pd.read_csv(file_path, sep=';')

# Usar a coluna power
data['power'] = pd.to_numeric(data['power'], errors='coerce')
data = data.dropna()

# Normalizar os dados
scaler = MinMaxScaler()
data['power'] = scaler.fit_transform(data[['power']])

# Criar sequências para prever 1 timestep
def create_sequences(data, n_input):
    X, y = [], []
    for i in range(len(data) - n_input):
        X.append(data[i:i + n_input])  # Janela deslizante de entrada
        y.append(data[i + n_input])   # Próximo timestep como saída
    return np.array(X), np.array(y)

series = data['power'].values
X, y = create_sequences(series, config.N_input)

# Dividir em treino e teste
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Ajustar dimensões para saída de um timestep
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Converter para tensores do PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32).to(config.device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(config.device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(config.device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(config.device)

# Criar DataLoader
train_data = torch.utils.data.TensorDataset(X_train, y_train)
test_data = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

# Modelo LSTM
class SlidingWindowLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(SlidingWindowLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layer para saída
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Última saída da sequência
        out = self.fc(out)   # Previsão do próximo timestep
        return out

# Instanciar o modelo
model = SlidingWindowLSTM(
    input_dim=config.input_dim,
    hidden_dim=config.hidden_dim,
    output_dim=1,  # Apenas 1 timestep na saída
    num_layers=config.num_layers
).to(config.device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Treinamento
for epoch in range(config.epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.unsqueeze(-1)  # Adicionar dimensão de entrada
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch+1}/{config.epochs}, Loss: {train_loss/len(train_loader):.4f}")

# Avaliação
model.eval()
test_loss = 0.0
predictions = []
actuals = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.unsqueeze(-1)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item()
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(y_batch.cpu().numpy())

print(f"Teste Loss: {test_loss/len(test_loader):.4f}")

# Reverter normalização para visualização
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))

# Salvar modelo
model_path = os.path.join(results_dir, "model_sliding.pth")
torch.save(model.state_dict(), model_path)
print(f"Modelo salvo em: {model_path}")

# Visualizar e salvar resultados
plt.figure(figsize=(10, 6))
plt.plot(range(len(actuals)), actuals, label='Valores Reais')
plt.plot(range(len(predictions)), predictions, label='Previsões')
plt.legend()
plt.title("Previsão com Janela Deslizante")
plt.xlabel("Timesteps")
plt.ylabel("Potência")

# Salvar a figura
figure_path = os.path.join(results_dir, "prediction_plot_sliding.png")
plt.savefig(figure_path)
print(f"Figura salva em: {figure_path}")
plt.show()
