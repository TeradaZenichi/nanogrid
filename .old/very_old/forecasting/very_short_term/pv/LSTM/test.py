# forecasting/very_short_term/pv/LSTM/test.py

import os
import torch
import torch.nn as nn
import joblib
from torch.utils.data import DataLoader, Subset
import numpy as np

from model import HParams, PVDataset, PVModel

def get_time_series_splits(dataset, train_pct=0.7, val_pct=0.2):
    """
    Divide o dataset de forma sequencial em índices para treino, validação e teste.
    """
    total = len(dataset)
    train_end = int(total * train_pct)
    val_end = int(total * (train_pct + val_pct))
    train_idx = list(range(0, train_end))
    val_idx = list(range(train_end, val_end))
    test_idx = list(range(val_end, total))
    return train_idx, val_idx, test_idx

def test_model(model, test_loader, criterion, device):
    """
    Avalia o modelo no conjunto de teste, retornando a loss, as predições e os valores reais.
    """
    model.eval()
    total_loss = 0.0
    predictions = []
    actuals = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y.cpu().numpy())
    test_loss = total_loss / len(test_loader.dataset)
    return test_loss, predictions, actuals

def main():
    # Caminhos para o modelo, dados e estudo Optuna
    best_model_path = 'C:/Users/Lucas/Code/nanogrid/forecasting/very_short_term/pv/LSTM/best_model.pth'
    csv_file = 'C:/Users/Lucas/Code/nanogrid/data/pv_5min.csv'
    study_path = 'C:/Users/Lucas/Code/nanogrid/forecasting/very_short_term/pv/LSTM/optuna_study.pkl'
    
    # Carrega o estudo salvo para recuperar os melhores hiperparâmetros
    study = joblib.load(study_path)
    best_params = study.best_params
    print("Best Hyperparameters from HPO:", best_params)
    
    # Cria uma instância de HParams e atualiza com os melhores valores
    hparams = HParams()
    hparams.window_size = best_params["window_size"]
    hparams.hidden_size = best_params["hidden_size"]
    hparams.num_layers = best_params["num_layers"]
    hparams.learning_rate = best_params["learning_rate"]
    hparams.epochs = best_params["epochs"]  # embora não seja usado no teste
    hparams.batch_size = best_params["batch_size"]
    
    # Carrega o dataset e realiza a divisão sequencial (70% treino, 20% validação, 10% teste)
    dataset = PVDataset(csv_file, window_size=hparams.window_size)
    _, _, test_idx = get_time_series_splits(dataset, train_pct=0.7, val_pct=0.2)
    test_dataset = Subset(dataset, test_idx)
    
    # DataLoader sem shuffle para preservar a ordem temporal
    test_loader = DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instancia o modelo com os hiperparâmetros otimizados
    model = PVModel(window_size=hparams.window_size, 
                    hidden_size=hparams.hidden_size, 
                    num_layers=hparams.num_layers).to(device)
    
    # Carrega os pesos do melhor modelo salvo
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    
    criterion = nn.MSELoss()
    test_loss, predictions, actuals = test_model(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    
    # Opcional: plotar 5 dias aleatórios
    import random
    import matplotlib.pyplot as plt

    # Número de pontos por dia (24 horas * 12 intervalos de 5 min = 288)
    points_per_day = 288

    if len(actuals) < points_per_day:
        print("Número insuficiente de amostras no conjunto de teste para plotar um dia completo.")
    else:
        # Seleciona 5 índices de início aleatórios que permitam um dia completo
        max_start = len(actuals) - points_per_day
        random_starts = random.sample(range(max_start), 5)
        for i, start in enumerate(random_starts):
            end = start + points_per_day
            plt.figure(figsize=(12,6))
            plt.plot(actuals[start:end], label='Real')
            plt.plot(predictions[start:end], label='Predicted')
            plt.legend()
            plt.title(f"Predição vs Real - Dia aleatório {i+1}")
            plt.xlabel("Intervalos de 5 min")
            plt.ylabel("Ppower")
            plt.show()


if __name__ == '__main__':
    main()
