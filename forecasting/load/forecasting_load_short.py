# -*- coding: utf-8 -*-
"""
Hourly Load Forecast with LSTM (multi-step: next H hours, default 36)

Key fixes vs previous version:
- End-aligned windows: input uses [i-LOOKBACK+1 ... i] (inclusive), targets are [i+1 ... i+H].
- Time-aware split: no horizon leakage from train into val.
- Save model without optimizer to simplify inference loading.

Author: You
"""

import os
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# -------------------------
# Configuration (edit here)
# -------------------------
CSV_PATH = "data/load_5min_train.csv"

# Resampling
RESAMPLE_RULE = "H"       # hourly
RESAMPLE_AGG = "mean"     # mean | median | sum (prefer mean for normalized data)

# Forecast setup
LOOKBACK_HOURS = 8        # hours of history in input window
HORIZON_HOURS  = 36       # hours ahead to predict
TRAIN_RATIO    = 0.75     # temporal split on the hourly index

# Model hyperparameters
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 64
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
EPOCHS = 80
BATCH_SIZE = 128
PATIENCE = 10

SEED = 42

# Output dirs
MODELS_DIR = "models"
FIGURES_DIR = "figures"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# -------------------------
# Reproducibility
# -------------------------
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# Utils
# -------------------------
def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical features for day-of-week and hour-of-day."""
    idx = df.index
    # Day of week [0..6]
    dow = idx.dayofweek.values.astype(float)
    dow_angle = 2.0 * np.pi * dow / 7.0
    df["dow_sin"] = np.sin(dow_angle)
    df["dow_cos"] = np.cos(dow_angle)
    # Hour of day [0..23]
    hod = idx.hour.values.astype(float)
    hod_angle = 2.0 * np.pi * hod / 24.0
    df["hod_sin"] = np.sin(hod_angle)
    df["hod_cos"] = np.cos(hod_angle)
    return df

def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R^2 over flattened arrays (multi-step)."""
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot

@tf.function
def r2_keras(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    mean_y = tf.reduce_mean(y_true)
    ss_tot = tf.reduce_sum(tf.square(y_true - mean_y))
    return 1.0 - ss_res / (ss_tot + tf.keras.backend.epsilon())

def build_end_aligned_windows(
    X: np.ndarray,
    y: np.ndarray,
    lookback_steps: int,
    horizon_steps: int,
    end_indices: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build windows using end-aligned semantics:
      For each end index i (inclusive):
        X_window = X[i - lookback_steps + 1 : i + 1, :]
        y_window = y[i + 1 : i + 1 + horizon_steps]
    end_indices must satisfy:
      i >= lookback_steps - 1  and  i <= len(X) - 1 - horizon_steps
    """
    n_feat = X.shape[1]
    n_samples = len(end_indices)
    X_seq = np.zeros((n_samples, lookback_steps, n_feat), dtype=np.float32)
    Y_seq = np.zeros((n_samples, horizon_steps), dtype=np.float32)
    for k, i in enumerate(end_indices):
        xs = i - lookback_steps + 1
        xe = i + 1
        ys = i + 1
        ye = i + 1 + horizon_steps
        X_seq[k] = X[xs:xe, :]
        Y_seq[k] = y[ys:ye]
    return X_seq, Y_seq

# -------------------------
# Load and resample
# -------------------------
df5 = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
df5 = df5.sort_values("timestamp").set_index("timestamp")

if "p_norm" not in df5.columns:
    raise ValueError("CSV must contain column 'p_norm'.")

# Hourly resampling (consistent with training normalization)
if RESAMPLE_AGG == "mean":
    dfH = df5.resample(RESAMPLE_RULE).mean(numeric_only=True)
elif RESAMPLE_AGG == "median":
    dfH = df5.resample(RESAMPLE_RULE).median(numeric_only=True)
elif RESAMPLE_AGG == "sum":
    dfH = df5.resample(RESAMPLE_RULE).sum(numeric_only=True)
else:
    raise ValueError("Unsupported RESAMPLE_AGG. Use 'mean' | 'median' | 'sum'.")

dfH = dfH.dropna()
dfH = add_cyclical_time_features(dfH)

# Build features: past p_norm + cyclical predictors
feature_cols = ["p_norm", "dow_sin", "dow_cos", "hod_sin", "hod_cos"]
X_all = dfH[feature_cols].values.astype(np.float32)
y_all = dfH["p_norm"].values.astype(np.float32)
n_points = len(dfH)

# Steps are hourly so 60 min per step
LOOKBACK_STEPS = int(LOOKBACK_HOURS)    # 1 step = 1 hour
HORIZON_STEPS  = int(HORIZON_HOURS)

# Valid end indices i must have enough history and future:
#   i >= LOOKBACK_STEPS - 1
#   i <= n_points - 1 - HORIZON_STEPS
i_min = LOOKBACK_STEPS - 1
i_max = n_points - 1 - HORIZON_STEPS
if i_max < i_min:
    raise ValueError("Not enough hourly data for the chosen lookback/horizon.")

# -------------------------
# Time-aware split (no horizon leakage)
# -------------------------
# Choose a split boundary on the hourly timeline
boundary_idx = int(math.floor(n_points * TRAIN_RATIO)) - 1
boundary_idx = max(boundary_idx, 0)

# Training end indices i such that targets [i+1 ... i+H] lie strictly before the boundary:
# i + HORIZON_STEPS <= boundary_idx  ->  i <= boundary_idx - HORIZON_STEPS
train_i_max = min(i_max, boundary_idx - HORIZON_STEPS)
train_i_min = i_min
train_end_indices = list(range(train_i_min, train_i_max + 1)) if train_i_max >= train_i_min else []

# Validation starts at the boundary (first val target crosses boundary):
# i >= boundary_idx  and  i <= i_max
val_i_min = max(i_min, boundary_idx)
val_i_max = i_max
val_end_indices = list(range(val_i_min, val_i_max + 1)) if val_i_max >= val_i_min else []

if (len(train_end_indices) == 0) or (len(val_end_indices) == 0):
    raise ValueError(
        f"Split produced empty sets. "
        f"Try reducing HORIZON_HOURS or increasing TRAIN_RATIO. "
        f"(train_i_max={train_i_max}, val_i_min={val_i_min}, i_min={i_min}, i_max={i_max})"
    )

# Build supervised tensors (end-aligned)
X_train, y_train = build_end_aligned_windows(X_all, y_all, LOOKBACK_STEPS, HORIZON_STEPS, train_end_indices)
X_val,   y_val   = build_end_aligned_windows(X_all, y_all, LOOKBACK_STEPS, HORIZON_STEPS, val_end_indices)

print(f"Hourly points: {n_points}  | Split boundary idx: {boundary_idx} (time={dfH.index[boundary_idx]})")
print(f"Lookback steps: {LOOKBACK_STEPS} | Horizon steps: {HORIZON_STEPS}")
print(f"Train samples: {len(train_end_indices)} (i∈[{train_end_indices[0]}, {train_end_indices[-1]}])")
print(f"Val   samples: {len(val_end_indices)} (i∈[{val_end_indices[0]}, {val_end_indices[-1]}])")

# -------------------------
# Build model
# -------------------------
input_shape = (LOOKBACK_STEPS, X_train.shape[2])
model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.LSTM(LSTM_UNITS_1, return_sequences=True),
    layers.LSTM(LSTM_UNITS_2, return_sequences=False),
    layers.Dropout(DROPOUT_RATE),
    layers.Dense(HORIZON_STEPS, activation="linear")
])

opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=opt,
    loss="mse",
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.MeanSquaredError(name="mse"),
        r2_keras
    ]
)
model.summary()

# -------------------------
# Train
# -------------------------
ts = time.strftime("%Y%m%d-%H%M%S")
model_name = f"lstm_hourly_load_{HORIZON_STEPS}h_{LOOKBACK_STEPS}"
ckpt_path = os.path.join(MODELS_DIR, model_name + ".keras")

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
    verbose=1,
)

# -------------------------
# Evaluate on validation
# -------------------------
y_val_pred = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
mae = float(np.mean(np.abs(y_val - y_val_pred)))
mse = float(np.mean(np.square(y_val - y_val_pred)))
r2  = float(r2_score_np(y_val, y_val_pred))

print("\nValidation metrics (flattened across the full horizon):")
print(f"MAE: {mae:.6f}")
print(f"MSE: {mse:.6f}")
print(f"R2 : {r2:.6f}")

# Visual sanity check: plot 1 validation sample (full horizon)
# create a plot for the plot y_val[i], y_val[i+24], y_val[i+48] and y_val_pred[i], y_val_pred[i+24], y_val_pred[i+48]
# first concatenate y_val and y_val_pred for easier indexing
i = 14  # first sample in validation
y_val_concat = np.concatenate([y_val[i], y_val[i + 36], y_val[i + 72]])
y_val_pred_concat = np.concatenate([y_val_pred[i], y_val_pred[i + 36], y_val_pred[i + 72]])
plt.figure(figsize=(10, 5))
plt.plot(y_val_concat, label="True", marker='o')
plt.plot(y_val_pred_concat, label="Pred", marker='x')
plt.title(f"Validation Sample {i} (first +36h +72h steps)")
plt.xlabel("Hour")
plt.ylabel("Normalized Load")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(FIGURES_DIR, model_name + "_val_sample.png"), dpi=150)


# -------------------------
# Save config & metrics
# -------------------------
@dataclass
class RunConfig:
    csv_path: str
    resample_rule: str
    resample_agg: str
    lookback_hours: int
    horizon_hours: int
    train_ratio: float
    lookback_steps: int
    horizon_steps: int
    features: list
    seed: int
    model_name: str

run_cfg = RunConfig(
    csv_path=CSV_PATH,
    resample_rule=RESAMPLE_RULE,
    resample_agg=RESAMPLE_AGG,
    lookback_hours=LOOKBACK_HOURS,
    horizon_hours=HORIZON_HOURS,
    train_ratio=TRAIN_RATIO,
    lookback_steps=LOOKBACK_STEPS,
    horizon_steps=HORIZON_STEPS,
    features=feature_cols,
    seed=SEED,
    model_name=model_name
)
meta = {"config": asdict(run_cfg), "val_metrics": {"MAE": mae, "MSE": mse, "R2": r2}}
with open(os.path.join(MODELS_DIR, model_name + "_config.json"), "w") as f:
    json.dump(meta, f, indent=2)

# -------------------------
# Plot training curve
# -------------------------
plt.figure(figsize=(8, 4))
plt.plot(hist.history["loss"], label="train_loss")
plt.plot(hist.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("LSTM Training Curve (Hourly, Multi-step)")
plt.legend()
plt.grid(True, alpha=0.3)
fig_path = os.path.join(FIGURES_DIR, model_name + "_training_curve.png")
plt.tight_layout()
plt.savefig(fig_path, dpi=150)

# Save model WITHOUT optimizer to avoid deserialization issues at inference time
model.save(ckpt_path, include_optimizer=False)

print(f"\nSaved best model to: {ckpt_path}")
print(f"Saved config to:     {os.path.join(MODELS_DIR, model_name + '_config.json')}")
print(f"Saved training plot: {fig_path}")
