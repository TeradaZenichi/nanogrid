# -*- coding: utf-8 -*-
"""
LSTM Very-Short-Term Load Forecast (5-min data → next hour = 12 steps)

- Input CSV: load_5min_train.csv with columns: timestamp,p_norm
- End-aligned windows: X[i-LB+1:i+1], y[i+1:i+1+H]
- Time-aware split (no horizon leakage)
- Metrics (MSE, MAE, R2), training curve
- Saves best model (include_optimizer=False)

Author: You
"""

import os, json, math, time
from dataclasses import dataclass, asdict
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# -------------------------
# Config (edit here)
# -------------------------
CSV_PATH = "data/load_5min_train_smoothed_normalized.csv"

HORIZON_MINUTES = 60*4          # forecast horizon (next hour) -> 12 steps @5min
LOOKBACK_HOURS  = 8.0         # hours of history in input
TRAIN_RATIO     = 0.75        # temporal split

# Model hyperparameters
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 128
PATIENCE = 10
SEED = 42

# Output dirs
MODELS_DIR = "models"
FIGURES_DIR = "figures"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# Utils
# -------------------------
def infer_step_minutes(index: pd.DatetimeIndex) -> int:
    """Infer the data resolution in minutes from the time index."""
    diffs = index.to_series().diff().dropna()
    step = diffs.mode().iloc[0]
    step_min = int(step.total_seconds() // 60)
    if step_min <= 0:
        raise ValueError("Could not infer a positive step size from timestamps.")
    return step_min

def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical features: day-of-week and time-of-day (sin/cos)."""
    idx = df.index
    dow = idx.dayofweek.values.astype(float)
    dow_angle = 2.0 * np.pi * dow / 7.0
    df["dow_sin"] = np.sin(dow_angle)
    df["dow_cos"] = np.cos(dow_angle)

    tod_min = (idx.hour * 60 + idx.minute).values.astype(float)  # 0..1435
    tod_angle = 2.0 * np.pi * tod_min / (24.0 * 60.0)
    df["tod_sin"] = np.sin(tod_angle)
    df["tod_cos"] = np.cos(tod_angle)
    return df

def build_end_aligned_windows(
    X: np.ndarray,
    y: np.ndarray,
    lookback_steps: int,
    horizon_steps: int,
    end_indices: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each end index i (inclusive):
      X_window = X[i - lookback_steps + 1 : i + 1, :]
      y_window = y[i + 1 : i + 1 + horizon_steps]
    """
    n_feat = X.shape[1]
    X_seq = np.zeros((len(end_indices), lookback_steps, n_feat), dtype=np.float32)
    Y_seq = np.zeros((len(end_indices), horizon_steps), dtype=np.float32)
    for k, i in enumerate(end_indices):
        xs, xe = i - lookback_steps + 1, i + 1
        ys, ye = i + 1, i + 1 + horizon_steps
        X_seq[k] = X[xs:xe, :]
        Y_seq[k] = y[ys:ye]
    return X_seq, Y_seq

def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = y_true.reshape(-1), y_pred.reshape(-1)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot

@tf.function
def r2_keras(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    mean_y = tf.reduce_mean(y_true)
    ss_tot = tf.reduce_sum(tf.square(y_true - mean_y))
    return 1.0 - ss_res / (ss_tot + tf.keras.backend.epsilon())

@dataclass
class RunConfig:
    csv_path: str
    horizon_minutes: int
    lookback_hours: float
    train_ratio: float
    step_minutes: int
    lookback_steps: int
    horizon_steps: int
    features: list
    seed: int
    model_name: str

# -------------------------
# Load data & features
# -------------------------
df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
if "p_norm" not in df.columns:
    raise ValueError("CSV must contain column 'p_norm'.")

STEP_MIN = infer_step_minutes(df.index)  # expect 5
if HORIZON_MINUTES % STEP_MIN != 0:
    raise ValueError(f"HORIZON_MINUTES ({HORIZON_MINUTES}) must be multiple of step ({STEP_MIN}).")

H = HORIZON_MINUTES // STEP_MIN
LB = int(round(LOOKBACK_HOURS * 60.0 / STEP_MIN))
if LB < 1:
    raise ValueError("LOOKBACK_STEPS must be >= 1.")

df = add_cyclical_time_features(df)
feature_cols = ["p_norm", "dow_sin", "dow_cos", "tod_sin", "tod_cos"]
X_all = df[feature_cols].values.astype(np.float32)
y_all = df["p_norm"].values.astype(np.float32)
n_points = len(df)

# -------------------------
# Valid end indices & split (time-aware, no horizon leakage)
# -------------------------
i_min = LB - 1
i_max = n_points - 1 - H
if i_max < i_min:
    raise ValueError("Not enough 5-min data for the chosen lookback/horizon.")

boundary_idx = max(int(math.floor(n_points * TRAIN_RATIO)) - 1, 0)

# Train: ensure i + H <= boundary_idx  ⇒  i <= boundary_idx - H
train_i_min = i_min
train_i_max = min(i_max, boundary_idx - H)
train_end_indices = list(range(train_i_min, train_i_max + 1)) if train_i_max >= train_i_min else []

# Val  : start at boundary (targets cross boundary), until i_max
val_i_min = max(i_min, boundary_idx)
val_i_max = i_max
val_end_indices = list(range(val_i_min, val_i_max + 1)) if val_i_max >= val_i_min else []

if (len(train_end_indices) == 0) or (len(val_end_indices) == 0):
    raise ValueError(
        f"Empty split. Increase TRAIN_RATIO or reduce HORIZON_MINUTES. "
        f"(train_i_max={train_i_max}, val_i_min={val_i_min}, i_min={i_min}, i_max={i_max})"
    )

# Build tensors
X_train, y_train = build_end_aligned_windows(X_all, y_all, LB, H, train_end_indices)
X_val,   y_val   = build_end_aligned_windows(X_all, y_all, LB, H, val_end_indices)

print(f"Step: {STEP_MIN} min | Lookback steps: {LB} | Horizon steps: {H}")
print(f"Points: {n_points} | Boundary idx: {boundary_idx} (time={df.index[boundary_idx]})")
print(f"Train samples: {len(train_end_indices)} | Val samples: {len(val_end_indices)}")

# -------------------------
# Build & train model
# -------------------------
input_shape = (LB, X_train.shape[2])
model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.LSTM(LSTM_UNITS_1, return_sequences=True),
    layers.LSTM(LSTM_UNITS_2, return_sequences=False),
    layers.Dropout(DROPOUT_RATE),
    layers.Dense(H, activation="linear")
])
opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=opt, loss="mse",
              metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                       tf.keras.metrics.MeanSquaredError(name="mse"),
                       r2_keras])

ts = time.strftime("%Y%m%d-%H%M%S")
model_name = f"llstm_vstf_load_{STEP_MIN}min_{HORIZON_MINUTES}min_{LB}"
ckpt_path  = os.path.join(MODELS_DIR, model_name + ".keras")

cbs = [
    callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
    callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True)
]

hist = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cbs,
    verbose=1
)

# -------------------------
# Evaluate
# -------------------------
y_val_pred = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
mae = float(np.mean(np.abs(y_val - y_val_pred)))
mse = float(np.mean((y_val - y_val_pred) ** 2))
r2  = float(r2_score_np(y_val, y_val_pred))
print("\nValidation metrics (flattened over horizon):")
print(f"MAE: {mae:.6f} | MSE: {mse:.6f} | R2: {r2:.6f}")

# -------------------------
# Plot: 3 samples spaced by horizon (visual sanity check)
# -------------------------
i = 12*7  # first sample in validation
y_val_concat = np.concatenate([y_val[i], y_val[i + 12], y_val[i + 24]])
y_val_pred_concat = np.concatenate([y_val_pred[i], y_val_pred[i + 12], y_val_pred[i + 24]])
plt.figure(figsize=(10, 5))
plt.plot(y_val_concat, label="True", marker='o')
plt.plot(y_val_pred_concat, label="Pred", marker='x')
plt.title(f"Validation Sample {i} (first +12h +24h steps)")
plt.xlabel("Hour")
plt.ylabel("Normalized Load")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(FIGURES_DIR, model_name + "_val_sample_VSTF.png"), dpi=150)


# -------------------------
# Save config, plot, and model (no optimizer)
# -------------------------
@dataclass
class RunCfg:
    csv_path: str
    horizon_minutes: int
    lookback_hours: float
    train_ratio: float
    step_minutes: int
    lookback_steps: int
    horizon_steps: int
    features: list
    seed: int
    model_name: str

run_cfg = RunCfg(
    csv_path=CSV_PATH,
    horizon_minutes=HORIZON_MINUTES,
    lookback_hours=LOOKBACK_HOURS,
    train_ratio=TRAIN_RATIO,
    step_minutes=STEP_MIN,
    lookback_steps=LB,
    horizon_steps=H,
    features=["p_norm","dow_sin","dow_cos","tod_sin","tod_cos"],
    seed=SEED,
    model_name=model_name
)
with open(os.path.join(MODELS_DIR, model_name + "_config.json"), "w") as f:
    json.dump({"config": asdict(run_cfg),
               "val_metrics": {"MAE": mae, "MSE": mse, "R2": r2}}, f, indent=2)

# Save model WITHOUT optimizer (safer to load with compile=False)
model.save(ckpt_path, include_optimizer=False)
print(f"\nSaved best model to: {ckpt_path}")
print(f"Saved config to:     {os.path.join(MODELS_DIR, model_name + '_config.json')}")
print(f"Saved training plot: {os.path.join(FIGURES_DIR, model_name + '_val_3xH.png')}")
