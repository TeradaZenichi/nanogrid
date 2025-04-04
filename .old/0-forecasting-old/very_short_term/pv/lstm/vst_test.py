import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from forecasting.pv.lstm.vst_model import LSTMModel

# =======================
# 1. CARREGAR OS DADOS
# =======================
file_path = "data/pv_5min.csv"
df = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')

# NORMALIZAR OS DADOS ENTRE 0 E 1
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['Ppower']])

# =======================
# 2. CRIAÇÃO DE SEQUÊNCIAS PARA TESTE
# =======================
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = int(24*60/5)  # 24 passos (~2h)
X, y = create_sequences(df_scaled, seq_length)

# CONVERTER PARA TENSORES DO PYTORCH
X_tensor = torch.tensor(X, dtype=torch.float32).clone().detach()
y_tensor = torch.tensor(y, dtype=torch.float32).clone().detach()

# =======================
# 3. SPLIT DOS DADOS (70% treino, 15% validação, 15% teste)
# =======================
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
test_size = len(X) - train_size - val_size

X_test, y_test = X_tensor[train_size+val_size:], y_tensor[train_size+val_size:]

print(f"Tamanho do Conjunto de Teste: {len(X_test)}")

# =======================
# 4. DEFINIR O BATCH SIZE E CRIAR DATASET
# =======================
batch_size = 16  # Reduzido para evitar OOM na GPU

class PVForecastDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.clone().detach()
        self.y = y.clone().detach()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

test_dataset = PVForecastDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =======================
# 5. CARREGAR O MODELO TREINADO
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel().to(device)

# Carregar pesos treinados
model.load_state_dict(torch.load("lstm_model.pth", map_location=device))
model.eval()

# =======================
# 6. TESTAR O MODELO (PROCESSANDO EM LOTES)
# =======================
y_pred_list = []

with torch.no_grad():
    for X_test_batch, _ in tqdm(test_loader, desc="Testando LSTM"):
        X_test_batch = X_test_batch.to(device)  # Enviar batch para GPU/CPU
        y_pred_batch = model(X_test_batch)  # Fazer previsão

        y_pred_batch = y_pred_batch.squeeze(-1).cpu()  # Remover dimensões extras e mover para CPU
        y_pred_list.append(y_pred_batch)

# Concatenar todas as previsões
y_pred = torch.cat(y_pred_list, dim=0)

# =======================
# 7. CONVERTER PARA FORMATO ORIGINAL
# =======================
y_test_np = scaler.inverse_transform(y_test.cpu().numpy())
y_pred_np = scaler.inverse_transform(y_pred.numpy().reshape(-1, 1))

# =======================
# 8. CÁLCULO DAS MÉTRICAS DE ERRO
# =======================
mae = mean_absolute_error(y_test_np, y_pred_np)
rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_np))

# Exibir métricas
print(f"📌 MAE (Erro Médio Absoluto): {mae:.4f}")
print(f"📌 RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f}")

# =======================
# 9. SALVAR RESULTADOS
# =======================
np.savetxt("predictions.csv", np.column_stack([y_test_np, y_pred_np]), delimiter=",", header="Real,Previsto", comments="")
print("Previsões salvas em 'predictions.csv'")

# =======================
# 10. PLOTAR RESULTADOS COM MÉTRICAS
# =======================
plt.figure(figsize=(12,6))
plt.plot(y_test_np[:288], label="Real")  # Exibe apenas 24 horas
plt.plot(y_pred_np[:288], label="Previsto", linestyle="dashed")
plt.legend()
plt.title(f"Previsão para Very Short-Term\nMAE: {mae:.4f}, RMSE: {rmse:.4f}")
plt.savefig("forecast_very_short_term.pdf")
plt.show()
