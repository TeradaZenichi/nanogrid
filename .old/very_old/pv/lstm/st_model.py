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
# 2. CRIAÇÃO DE SEQUÊNCIAS PARA PREVISÃO DAY-AHEAD (36H)
# =======================
def create_sequences(data, seq_length, future_steps):
    X, y = [], []
    for i in range(len(data) - seq_length - future_steps):
        X.append(data[i:i+seq_length])  # Últimos `seq_length` pontos como entrada
        y.append(data[i+seq_length : i+seq_length+future_steps])  # Próximos `future_steps` pontos como saída
    return np.array(X), np.array(y)

seq_length = 288  # Usamos 1 dia de histórico (~288 pontos)
future_steps = 432  # Previsão para 36 horas (~432 pontos de 5 min)

X, y = create_sequences(df_scaled, seq_length, future_steps)

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
batch_size = 16  # Reduzido para evitar OOM na GPU

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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =======================
# 6. DEFINIR O MODELO LSTM (AGORA PREVÊ VÁRIOS PASSOS)
# =======================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=future_steps):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Agora prevê `future_steps` valores

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

            # Corrigir dimensões antes da perda
            y_batch = y_batch.squeeze(-1)
            outputs = outputs.squeeze(-1)

            loss = criterion(outputs, y_batch)  # Agora `outputs` e `y_batch` têm o mesmo shape
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

                val_outputs = model(X_val_batch)

                y_val_batch = y_val_batch.squeeze(-1)
                val_outputs = val_outputs.squeeze(-1)

                val_loss = criterion(val_outputs, y_val_batch)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        pbar.set_postfix({"Train Loss": f"{avg_train_loss:.6f}", "Val Loss": f"{avg_val_loss:.6f}"})
        pbar.update(1)

# =======================
# 8. TESTAR O MODELO SEM ESTOURAR A MEMÓRIA
# =======================
model.eval()
y_pred_list = []

with torch.no_grad():
    for X_test_batch, _ in test_loader:  # Prevendo em lotes
        X_test_batch = X_test_batch.to(device)
        y_pred_batch = model(X_test_batch)
        
        # Corrigir dimensões antes de salvar na CPU
        y_pred_batch = y_pred_batch.squeeze(-1)
        
        y_pred_list.append(y_pred_batch.cpu())  # Salvar na CPU para economizar memória da GPU

# Concatenar previsões
y_pred = torch.cat(y_pred_list, dim=0)

# =======================
# 9. CONVERTER PARA FORMATO ORIGINAL
# =======================
y_test_np = scaler.inverse_transform(y_test.cpu().numpy())
y_pred_np = scaler.inverse_transform(y_pred.numpy())  # Já está na CPU

# =======================
# 10. PLOTAR RESULTADOS PARA 36H
# =======================
plt.figure(figsize=(12,6))
plt.plot(y_test_np[0], label="Real")
plt.plot(y_pred_np[0], label="Previsto", linestyle="dashed")
plt.legend()
plt.title("Previsão da Geração PV para as Próximas 36h")
plt.savefig("forecast_36h.pdf")
plt.close("all")

# =======================
# 11. SALVAR O MODELO TREINADO
# =======================
torch.save(model.state_dict(), "forecasting/pv/lstm/lstm_short_term.pth")
print("Modelo salvo como 'forecasting/pv/lstm/lstm_short_term.pth'")
