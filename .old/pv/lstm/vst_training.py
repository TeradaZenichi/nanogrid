import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import optuna
import os

# =======================
# 1. LOAD DATA
# =======================
file_path = "data/pv_5min.csv"
df = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')

# Normalize data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['Ppower']])

# =======================
# 2. CREATE SEQUENCES FOR 36H PREDICTION
# =======================
def create_sequences(data, seq_length, future_steps):
    X, y = [], []
    for i in range(len(data) - seq_length - future_steps):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length : i+seq_length+future_steps])
    return np.array(X), np.array(y)

seq_length = 288  # 1 day of history (~288 points)
future_steps = 432  # 36-hour forecast (~432 points)

X, y = create_sequences(df_scaled, seq_length, future_steps)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).clone().detach()
y_tensor = torch.tensor(y, dtype=torch.float32).clone().detach()

# =======================
# 3. SPLIT DATA (70% train, 15% validation, 15% test)
# =======================
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
test_size = len(X) - train_size - val_size

X_train, y_train = X_tensor[:train_size], y_tensor[:train_size]
X_val, y_val = X_tensor[train_size:train_size+val_size], y_tensor[train_size:train_size+val_size]
X_test, y_test = X_tensor[train_size+val_size:], y_tensor[train_size+val_size:]

print(f"Training Set: {len(X_train)} samples")
print(f"Validation Set: {len(X_val)} samples")
print(f"Test Set: {len(X_test)} samples")

# =======================
# 4. CREATE DATASET AND DATALOADER
# =======================
batch_size = 16  # Prevents GPU OOM errors

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
# 5. DEFINE LSTM MODEL
# =======================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=future_steps):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Output only the last timestep

# =======================
# 6. LOAD PROGRESS IF AVAILABLE
# =======================
progress_file = "training_progress.json"
if os.path.exists(progress_file):
    with open(progress_file, "r") as f:
        progress = json.load(f)
    best_val_loss = progress["best_val_loss"]
    train_losses = progress["train_losses"]
    val_losses = progress["val_losses"]
else:
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

# =======================
# 7. TRAINING WITH EARLY STOPPING
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
patience = 5
epochs_no_improve = 0

with tqdm(total=num_epochs, desc="Training LSTM") as pbar:
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            y_batch = y_batch.squeeze(-1)
            outputs = outputs.squeeze(-1)

            loss = criterion(outputs, y_batch)  
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
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

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_lstm_model.pth")

        # Save Progress
        progress = {
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses
        }
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=4)

        pbar.set_postfix({"Train Loss": f"{avg_train_loss:.6f}", "Val Loss": f"{avg_val_loss:.6f}"})
        pbar.update(1)

# =======================
# 8. SAVE OPTUNA STUDY
# =======================
study = optuna.create_study(direction="minimize", storage="sqlite:///optuna_study.db", load_if_exists=True)
study.optimize(lambda trial: best_val_loss, n_trials=1)  # Save only best trial result

# =======================
# 9. PLOT TRAINING LOSS CURVE
# =======================
plt.figure(figsize=(8,4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("loss_curve.pdf")
plt.close()

print("🔥 Training progress and best model saved!")