# forecasting/very_short_term/pv/LSTM/model.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class PVDataset(Dataset):
    def __init__(self, csv_file, window_size, transform=None):
        self.df = pd.read_csv(csv_file)
        if "Ppower" not in self.df.columns:
            raise ValueError("Coluna 'Ppower' não encontrada no CSV.")
        self.data = self.df["Ppower"].values.astype(np.float32)
        self.window_size = window_size
        self.transform = transform
        self.samples = []
        # Cria amostras usando uma janela deslizante
        for i in range(len(self.data) - window_size):
            x = self.data[i : i + window_size]
            y = self.data[i + window_size]
            self.samples.append((x, y))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        # Converte para tensor e adiciona dimensão de feature (univariado)
        x = torch.tensor(x).unsqueeze(-1)  # shape: (window_size, 1)
        y = torch.tensor(y)
        if self.transform:
            x = self.transform(x)
        return x, y

class PVModel(nn.Module):
    def __init__(self, window_size, hidden_size, num_layers=1):
        super(PVModel, self).__init__()
        # Camada LSTM univariada
        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        # Camada linear para gerar a predição final
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        output, _ = self.lstm(x)
        # Utiliza o último estado da sequência
        last_output = output[:, -1, :]
        y_pred = self.fc(last_output)
        return y_pred.squeeze()

class HParams:
    def __init__(self):
        # Hiperparâmetros do dataset
        self.window_size = 10
        
        # Hiperparâmetros do modelo
        self.hidden_size = 64
        self.num_layers = 1
        
        # Hiperparâmetros de treinamento
        self.learning_rate = 1e-3
        self.epochs = 50
        self.batch_size = 32
