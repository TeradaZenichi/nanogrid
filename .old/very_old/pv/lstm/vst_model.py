from torch.utils.data import Dataset
import torch.nn as nn
import json


class Config:
    def __init__(self, config_path="forecasting/pv/lstm/vst_config.json"):
        """
        Loads configurations from the JSON file.
        """
        with open(config_path, "r") as config_file:
            self.data = json.load(config_file)

        # Global configurations
        self.DATA_FILE = self.data["data_file"]
        self.TRAIN_RATIO = self.data["train_ratio"]
        self.VAL_RATIO = self.data["val_ratio"]
        self.TEST_RATIO = self.data["test_ratio"]
        self.NUM_TRIALS = self.data["num_trials"]

        # Hyperparameter ranges
        self.SEQ_LENGTH_VALUES = self.data["seq_length_values"]
        self.HIDDEN_SIZE_RANGE = self.data["hidden_size_range"]
        self.NUM_LAYERS_RANGE = self.data["num_layers_range"]
        self.DROPOUT_RANGE = self.data["dropout_range"]
        self.LEARNING_RATE_RANGE = self.data["learning_rate_range"]
        self.BATCH_SIZES = self.data["batch_sizes"]
        self.MAX_EPOCHS = self.data["max_epochs"]
        self.PATIENCE = self.data["patience"]


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, output_size)
 
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Take the last time step output



class PVForecastDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.clone().detach()
        self.y = y.clone().detach()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]