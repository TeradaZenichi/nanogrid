import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =======================
# 1. CARREGAR OS DADOS
# =======================
file_path = "data/pv_5min.csv"
df = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')

# NORMALIZAR OS DADOS ENTRE 0 E 1
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['Ppower']])

# =======================
# 2. CRIAÇÃO DE SEQUÊNCIAS PARA LSTM
# =======================
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 24  # Cada amostra contém 24 passos (~2h)
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

X_train, y_train = X_tensor[:train_size], y_tensor[:train_size]
X_val, y_val = X_tensor[train_size:train_size+val_size], y_tensor[train_size:train_size+val_size]
X_test, y_test = X_tensor[train_size+val_size:], y_tensor[train_size+val_size:]

print(f"Tamanho do Conjunto de Treinamento: {len(X_train)}")
print(f"Tamanho do Conjunto de Validação: {len(X_val)}")
print(f"Tamanho do Conjunto de Teste: {len(X_test)}")

# =======================
# 4. DEFINIR O BATCH SIZE COMO PARÂMETRO
# =======================
batch_size = 64  # Pode ser ajustado para otimizar desempenho

# =======================
# 5. CRIAR DATASET E DATALOADER
# =======================
class PVForecastDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.clone().detach()
        self.y = y.clone().detach()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = PVForecastDataset(X_train, y_train)
val_dataset = PVForecastDataset(X_val, y_val)
test_dataset = PVForecastDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =======================
# 6. DEFINIR O MODELO LSTM
# =======================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Pegamos a saída do último timestamp corretamente

# =======================
# 7. TREINAMENTO DO MODELO
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
train_losses = []
val_losses = []

with tqdm(total=num_epochs, desc="Treinando LSTM") as pbar:
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            y_batch = y_batch.unsqueeze(-1)  # Garantir que `y_batch` tem shape (batch_size, 1)

            loss = criterion(outputs, y_batch.squeeze(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validação
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_batch = y_val_batch.unsqueeze(-1)

                val_outputs = model(X_val_batch)
                val_loss = criterion(val_outputs, y_val_batch.squeeze(-1))
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        pbar.set_postfix({"Train Loss": f"{avg_train_loss:.6f}", "Val Loss": f"{avg_val_loss:.6f}"})
        pbar.update(1)

# =======================
# 8. SALVAR O MODELO TREINADO
# =======================
torch.save(model.state_dict(), "lstm_model.pth")
print("Modelo salvo como 'lstm_model.pth'")

# =======================
# 9. TESTAR O MODELO
# =======================
model.eval()
X_test, y_test = X_test.to(device), y_test.to(device)
with torch.no_grad():
    y_pred = model(X_test)

# CONVERTER PARA FORMATO ORIGINAL
y_test_np = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))
y_pred_np = scaler.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))

# SALVAR PREVISÕES
np.savetxt("predictions.csv", np.column_stack([y_test_np, y_pred_np]), delimiter=",", header="Real,Previsto", comments="")

# =======================
# 10. PLOTAR RESULTADOS
# =======================
plt.figure(figsize=(10,5))
plt.plot(y_test_np, label="Real")
plt.plot(y_pred_np, label="Previsto", linestyle="dashed")
plt.legend()
plt.title("Previsão da Geração PV com LSTM")
plt.show()

# =======================
# 11. PLOTAR A PERDA DURANTE O TREINAMENTO E VALIDAÇÃO
# =======================
plt.figure(figsize=(8,4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Época")
plt.ylabel("MSE Loss")
plt.title("Evolução da Perda no Treinamento e Validação")
plt.legend()
plt.show()