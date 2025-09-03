# forecasting/very_short_term/pv/LSTM/training.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import optuna
from optuna.exceptions import TrialPruned
import joblib

from model import HParams, PVDataset, PVModel

def get_time_series_splits(dataset, train_pct=0.7, val_pct=0.2):
    """
    Divide o dataset de forma sequencial em treinamento, validação e teste.
    Retorna os índices para cada conjunto.
    """
    total = len(dataset)
    train_end = int(total * train_pct)
    val_end = int(total * (train_pct + val_pct))
    train_idx = list(range(0, train_end))
    val_idx = list(range(train_end, val_end))
    test_idx = list(range(val_end, total))
    return train_idx, val_idx, test_idx

def run_epoch(model, data_loader, criterion, optimizer, device, train=True):
    """
    Executa uma época de treinamento ou validação.
    """
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(data_loader.dataset)

def run_hpo(hparams, n_trials=20):
    """
    Executa a otimização de hiperparâmetros usando Optuna com early stopping tradicional,
    atualizando o objeto hparams com os melhores valores encontrados.
    Salva os resultados para análises futuras.
    """
    csv_file = 'C:/Users/Lucas/Code/nanogrid/data/pv_5min.csv'

    def objective(trial):
        # Sugere os hiperparâmetros
        window_size = trial.suggest_categorical("window_size", [5, 30, 288, 576])
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        epochs = trial.suggest_categorical("epochs", [5, 10, 15])
        
        # Fixa as sementes para reprodutibilidade
        torch.manual_seed(0)
        np.random.seed(0)
        
        # Prepara o dataset e realiza o split sequencial
        dataset = PVDataset(csv_file, window_size=window_size)
        total = len(dataset)
        train_idx, val_idx, _ = get_time_series_splits(dataset, train_pct=0.7, val_pct=0.2)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        
        # DataLoaders sem shuffle para manter a ordem temporal
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PVModel(window_size=window_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Early stopping tradicional
        best_val_loss = float('inf')
        patience = 5       # número de épocas sem melhora permitidas
        min_delta = 1e-6   # melhoria mínima exigida
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            
            # Verifica se houve melhoria
            if best_val_loss - val_loss > min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Reporta a perda para o Optuna e verifica o pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise TrialPruned()
            
            print(f"Trial {trial.number} | Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"Trial {trial.number}: Early stopping triggered at epoch {epoch+1}")
                break
        
        return best_val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    print("Melhores hiperparâmetros encontrados:")
    print(study.best_params)
    print("Melhor Val Loss:", study.best_value)
    
    # Salva o estudo para análises futuras
    joblib.dump(study, "C:/Users/Lucas/Code/nanogrid/forecasting/very_short_term/pv/LSTM/optuna_study.pkl")
    study.trials_dataframe().to_csv("C:/Users/Lucas/Code/nanogrid/forecasting/very_short_term/pv/LSTM/optuna_trials.csv", index=False)
    
    # Atualiza os hiperparâmetros com os melhores valores
    best_params = study.best_params
    hparams.window_size = best_params["window_size"]
    hparams.hidden_size = best_params["hidden_size"]
    hparams.num_layers = best_params["num_layers"]
    hparams.learning_rate = best_params["learning_rate"]
    hparams.epochs = best_params["epochs"]
    hparams.batch_size = best_params["batch_size"]
    
    return study

def train():
    # Instancia os hiperparâmetros com valores default
    hparams = HParams()
    
    # Executa a HPO para atualizar os hiperparâmetros
    print("Iniciando a HPO...")
    study = run_hpo(hparams, n_trials=20)
    print("HPO concluída. Hiperparâmetros atualizados:")
    print(vars(hparams))
    
    # Prepara o dataset para o treinamento final, mantendo a ordem temporal
    csv_file = 'C:/Users/Lucas/Code/nanogrid/data/pv_5min.csv'
    dataset = PVDataset(csv_file, window_size=hparams.window_size)
    total = len(dataset)
    train_idx, val_idx, test_idx = get_time_series_splits(dataset, train_pct=0.7, val_pct=0.2)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)  # Para avaliação futura
    
    # DataLoaders sem shuffle
    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=hparams.batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PVModel(window_size=hparams.window_size, hidden_size=hparams.hidden_size, num_layers=hparams.num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
    
    best_val_loss = float('inf')
    best_model_path = 'C:/Users/Lucas/Code/nanogrid/forecasting/very_short_term/pv/LSTMbest_model.pth'
    
    for epoch in range(hparams.epochs):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"Época {epoch+1}/{hparams.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
    
    print("Treinamento final concluído. Melhor Val Loss:", best_val_loss)

if __name__ == '__main__':
    train()
