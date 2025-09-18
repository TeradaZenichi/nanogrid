# -*- coding: utf-8 -*-
"""
Hourly PV Forecast: LSTM (multi-step) and XGBoost (direct multi-step)

What this script provides
-------------------------
- Reads PV data from 'data/pv_5min_train.csv' with columns: [timestamp, p_norm]
- Resamples to hourly and builds an end-aligned supervised set
- Time-aware split (no horizon leakage)
- Two forecasting backends:
    * "lstm": sequence-to-multi-horizon (one LSTM, output H steps)
    * "xgb_direct": H independent XGBoost models, one per horizon step
- Saves:
    * Trained models under ./models
    * Training plots under ./figures
    * Config + validation metrics JSON
    * Latest horizon forecast CSV for both backends

Author: You (PV adaptation)
"""

import os
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Try to import xgboost lazily (only used if backend == "xgb_direct")
try:
    import xgboost as xgb
    from sklearn.multioutput import MultiOutputRegressor  # not used, direct strategy chosen
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# -------------------------
# Configuration (edit here)
# -------------------------
CSV_PATH = "data/pv_5min_train.csv"

# Resampling
RESAMPLE_RULE = "H"       # hourly
RESAMPLE_AGG = "mean"     # mean | median | sum

# Forecast setup
LOOKBACK_HOURS = 8        # history window length
HORIZON_HOURS  = 36       # hours ahead to predict
TRAIN_RATIO    = 0.75     # temporal split

# Backend: "lstm" or "xgb_direct"
BACKEND = "lstm"          # change to "xgb_direct" for XGBoost direct multi-step

# LSTM hyperparameters
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 64
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
EPOCHS = 40
BATCH_SIZE = 128
PATIENCE = 10

# XGBoost hyperparameters (used only if BACKEND == "xgb_direct")
XGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=0,  # use all cores
)

SEED = 42

# Output dirs
MODELS_DIR = "models"
FIGURES_DIR = "figures"
PRED_DIR = "predictions"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

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
    dow = idx.dayofweek.values.astype(float)
    dow_angle = 2.0 * np.pi * dow / 7.0
    df["dow_sin"] = np.sin(dow_angle)
    df["dow_cos"] = np.cos(dow_angle)
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
    End-aligned windows:
      For end index i (inclusive):
        X_window = X[i - lookback_steps + 1 : i + 1, :]
        y_window = y[i + 1 : i + 1 + horizon_steps]
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

def make_lag_matrix_from_series(y_all: np.ndarray, lookback_steps: int, end_indices: List[int]) -> np.ndarray:
    """
    Build a 2D lag matrix for tree models:
      For end index i, features are [y[i-L+1], ..., y[i]]  (L=lookback_steps).
    """
    n_samples = len(end_indices)
    X_lag = np.zeros((n_samples, lookback_steps), dtype=np.float32)
    for k, i in enumerate(end_indices):
        X_lag[k] = y_all[i - lookback_steps + 1: i + 1]
    return X_lag

def time_feats_for_targets(idx: pd.DatetimeIndex, end_indices: List[int], horizon_steps: int) -> List[np.ndarray]:
    """
    For each horizon h=1..H, build a (n_samples, 4) matrix with [dow_sin, dow_cos, hod_sin, hod_cos]
    for the target timestamp idx[i+h].
    Returns: list of length H with arrays shaped (n_samples, 4).
    """
    n_samples = len(end_indices)
    feats_per_h = []
    for h in range(1, horizon_steps + 1):
        dow_s = np.zeros((n_samples,), dtype=np.float32)
        dow_c = np.zeros((n_samples,), dtype=np.float32)
        hod_s = np.zeros((n_samples,), dtype=np.float32)
        hod_c = np.zeros((n_samples,), dtype=np.float32)
        for k, i in enumerate(end_indices):
            t = idx[i + h]
            dow = float(t.dayofweek)
            hod = float(t.hour)
            dow_ang = 2.0 * np.pi * dow / 7.0
            hod_ang = 2.0 * np.pi * hod / 24.0
            dow_s[k] = np.sin(dow_ang)
            dow_c[k] = np.cos(dow_ang)
            hod_s[k] = np.sin(hod_ang)
            hod_c[k] = np.cos(hod_ang)
        feats_per_h.append(np.stack([dow_s, dow_c, hod_s, hod_c], axis=1))
    return feats_per_h

# -------------------------
# Load and resample PV
# -------------------------
df5 = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
df5 = df5.sort_values("timestamp").set_index("timestamp")

if "p_norm" not in df5.columns:
    raise ValueError("CSV must contain column 'p_norm'.")

# Hourly resampling
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

# Features for LSTM branch (past p_norm + cyc features at each input step)
feature_cols = ["p_norm", "dow_sin", "dow_cos", "hod_sin", "hod_cos"]
X_all = dfH[feature_cols].values.astype(np.float32)
y_all = dfH["p_norm"].values.astype(np.float32)
n_points = len(dfH)

LOOKBACK_STEPS = int(LOOKBACK_HOURS)
HORIZON_STEPS  = int(HORIZON_HOURS)

# Valid end indices
i_min = LOOKBACK_STEPS - 1
i_max = n_points - 1 - HORIZON_STEPS
if i_max < i_min:
    raise ValueError("Not enough hourly data for the chosen lookback/horizon.")

# -------------------------
# Time-aware split (no horizon leakage)
# -------------------------
boundary_idx = int(math.floor(n_points * TRAIN_RATIO)) - 1
boundary_idx = max(boundary_idx, 0)

train_i_max = min(i_max, boundary_idx - HORIZON_STEPS)
train_i_min = i_min
train_end_indices = list(range(train_i_min, train_i_max + 1)) if train_i_max >= train_i_min else []

val_i_min = max(i_min, boundary_idx)
val_i_max = i_max
val_end_indices = list(range(val_i_min, val_i_max + 1)) if val_i_max >= val_i_min else []

if (len(train_end_indices) == 0) or (len(val_end_indices) == 0):
    raise ValueError(
        f"Split produced empty sets. "
        f"Try reducing HORIZON_HOURS or increasing TRAIN_RATIO. "
        f"(train_i_max={train_i_max}, val_i_min={val_i_min}, i_min={i_min}, i_max={i_max})"
    )

# -------------------------
# LSTM branch
# -------------------------
def run_lstm_branch():
    X_train, y_train = build_end_aligned_windows(X_all, y_all, LOOKBACK_STEPS, HORIZON_STEPS, train_end_indices)
    X_val,   y_val   = build_end_aligned_windows(X_all, y_all, LOOKBACK_STEPS, HORIZON_STEPS, val_end_indices)

    print(f"Hourly points: {n_points}  | Split boundary idx: {boundary_idx} (time={dfH.index[boundary_idx]})")
    print(f"LSTM Lookback steps: {LOOKBACK_STEPS} | Horizon steps: {HORIZON_STEPS}")
    print(f"Train samples: {len(train_end_indices)}  Val samples: {len(val_end_indices)}")

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

    model_name = f"lstm_hourly_pv_{HORIZON_STEPS}h_{LOOKBACK_STEPS}"
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

    # Evaluate
    y_val_pred = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    mae = float(np.mean(np.abs(y_val - y_val_pred)))
    mse = float(np.mean(np.square(y_val - y_val_pred)))
    r2  = float(r2_score_np(y_val, y_val_pred))

    print("\n[LSTM] Validation metrics (flattened across horizon):")
    print(f"MAE: {mae:.6f}  MSE: {mse:.6f}  R2: {r2:.6f}")

    # Plot one sanity-check sample (concatenating 3 consecutive windows, like your load script)
    i = min(14, len(y_val) - 73) if len(y_val) >= 73 else 0
    y_val_concat = np.concatenate([y_val[i], y_val[i + 36], y_val[i + 72]]) if len(y_val) > i + 72 else y_val[i]
    y_pred_concat = np.concatenate([y_val_pred[i], y_val_pred[i + 36], y_val_pred[i + 72]]) if len(y_val_pred) > i + 72 else y_val_pred[i]
    plt.figure(figsize=(10, 5))
    plt.plot(y_val_concat, label="True", marker='o')
    plt.plot(y_pred_concat, label="Pred", marker='x')
    plt.title(f"[LSTM] Validation Sample {i} (first +36h +72h)")
    plt.xlabel("Hour step")
    plt.ylabel("Normalized PV")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_sample = os.path.join(FIGURES_DIR, model_name + "_val_sample.png")
    plt.savefig(fig_sample, dpi=150)
    plt.close()

    # Training curve
    plt.figure(figsize=(8, 4))
    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("[LSTM] Training Curve (Hourly, Multi-step)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_curve = os.path.join(FIGURES_DIR, model_name + "_training_curve.png")
    plt.tight_layout()
    plt.savefig(fig_curve, dpi=150)
    plt.close()

    # Save model WITHOUT optimizer (clean inference)
    model.save(ckpt_path, include_optimizer=False)

    # Save config & metrics
    @dataclass
    class RunConfig:
        csv_path: str
        backend: str
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
        backend="lstm",
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

    # Export latest horizon forecast (last available end index on full data)
    i_last = n_points - 1 - HORIZON_STEPS
    X_last, _ = build_end_aligned_windows(X_all, y_all, LOOKBACK_STEPS, HORIZON_STEPS, [i_last])
    y_last_pred = model.predict(X_last, verbose=0)[0]
    horizon_idx = dfH.index[i_last + 1: i_last + 1 + HORIZON_STEPS]
    df_out = pd.DataFrame({"timestamp": horizon_idx, "y_pred_lstm": y_last_pred})
    out_csv = os.path.join(PRED_DIR, f"{model_name}_latest_forecast.csv")
    df_out.to_csv(out_csv, index=False)

    print(f"\n[LSTM] Saved best model to: {ckpt_path}")
    print(f"[LSTM] Saved config to:     {os.path.join(MODELS_DIR, model_name + '_config.json')}")
    print(f"[LSTM] Saved training plot: {fig_curve}")
    print(f"[LSTM] Saved sample plot:   {fig_sample}")
    print(f"[LSTM] Saved latest forecast CSV: {out_csv}")

# -------------------------
# XGBoost (direct multi-step) branch
# -------------------------
def run_xgb_direct_branch():
    if not XGB_AVAILABLE:
        raise ImportError("xgboost not available. Install with: pip install xgboost")

    # Build lag features from the PV target series (p_norm)
    X_lag_tr = make_lag_matrix_from_series(y_all, LOOKBACK_STEPS, train_end_indices)
    X_lag_va = make_lag_matrix_from_series(y_all, LOOKBACK_STEPS, val_end_indices)

    # Time features per horizon (for target timestamps)
    feats_tr_per_h = time_feats_for_targets(dfH.index, train_end_indices, HORIZON_STEPS)
    feats_va_per_h = time_feats_for_targets(dfH.index, val_end_indices, HORIZON_STEPS)

    # Prepare per-horizon training sets and train H models
    models_h = []
    val_preds = np.zeros((len(val_end_indices), HORIZON_STEPS), dtype=np.float32)
    train_preds = np.zeros((len(train_end_indices), HORIZON_STEPS), dtype=np.float32)

    print(f"[XGB] Training {HORIZON_STEPS} models (direct strategy)...")

    model_base_name = f"xgb_direct_hourly_pv_{HORIZON_STEPS}h_{LOOKBACK_STEPS}"
    step_mae, step_rmse = [], []

    for h in range(1, HORIZON_STEPS + 1):
        # Features: lags + time features at target time (i+h)
        X_tr_h = np.hstack([X_lag_tr, feats_tr_per_h[h - 1]]) if len(X_lag_tr) > 0 else feats_tr_per_h[h - 1]
        X_va_h = np.hstack([X_lag_va, feats_va_per_h[h - 1]]) if len(X_lag_va) > 0 else feats_va_per_h[h - 1]

        # Targets
        y_tr_h = np.array([y_all[i + h] for i in train_end_indices], dtype=np.float32)
        y_va_h = np.array([y_all[i + h] for i in val_end_indices], dtype=np.float32)

        model_h = xgb.XGBRegressor(**XGB_PARAMS)
        # Early stopping per horizon using validation of that horizon
        model_h.fit(
            X_tr_h, y_tr_h,
            eval_set=[(X_va_h, y_va_h)],
            verbose=False,
            early_stopping_rounds=50
        )
        models_h.append(model_h)

        # Predictions
        train_preds[:, h - 1] = model_h.predict(X_tr_h)
        val_preds[:, h - 1] = model_h.predict(X_va_h)

        # Step metrics
        mae_h = float(np.mean(np.abs(y_va_h - val_preds[:, h - 1])))
        rmse_h = float(np.sqrt(np.mean((y_va_h - val_preds[:, h - 1]) ** 2)))
        step_mae.append(mae_h)
        step_rmse.append(rmse_h)

        # Save each horizon model (JSON format)
        model_path = os.path.join(MODELS_DIR, f"{model_base_name}_step{h:02d}.json")
        model_h.get_booster().save_model(model_path)

    # Aggregate validation metrics across horizon
    mae = float(np.mean(np.abs(val_preds - build_end_aligned_windows(
        X_all, y_all, LOOKBACK_STEPS, HORIZON_STEPS, val_end_indices
    )[1])))
    mse = float(np.mean(np.square(val_preds - build_end_aligned_windows(
        X_all, y_all, LOOKBACK_STEPS, HORIZON_STEPS, val_end_indices
    )[1])))
    r2  = float(r2_score_np(build_end_aligned_windows(
        X_all, y_all, LOOKBACK_STEPS, HORIZON_STEPS, val_end_indices
    )[1], val_preds))

    print("\n[XGB] Validation metrics (flattened across horizon):")
    print(f"MAE: {mae:.6f}  MSE: {mse:.6f}  R2: {r2:.6f}")

    # Plot per-horizon MAE curve
    plt.figure(figsize=(9, 4))
    plt.plot(np.arange(1, HORIZON_STEPS + 1), step_mae, marker='o')
    plt.xlabel("Horizon step (h)")
    plt.ylabel("MAE")
    plt.title("[XGB] Validation MAE per step")
    plt.grid(True, alpha=0.3)
    fig_steps = os.path.join(FIGURES_DIR, model_base_name + "_val_mae_per_step.png")
    plt.tight_layout()
    plt.savefig(fig_steps, dpi=150)
    plt.close()

    # Plot a concatenated validation sample (like in your load script)
    i0 = 0
    if len(val_end_indices) > 72:
        y_val_concat = np.concatenate([
            build_end_aligned_windows(X_all, y_all, LOOKBACK_STEPS, HORIZON_STEPS, val_end_indices)[1][i0],
            build_end_aligned_windows(X_all, y_all, LOOKBACK_STEPS, HORIZON_STEPS, val_end_indices)[1][i0 + 36],
            build_end_aligned_windows(X_all, y_all, LOOKBACK_STEPS, HORIZON_STEPS, val_end_indices)[1][i0 + 72],
        ])
        y_pred_concat = np.concatenate([
            val_preds[i0],
            val_preds[i0 + 36],
            val_preds[i0 + 72],
        ])
    else:
        Y_val_full = build_end_aligned_windows(X_all, y_all, LOOKBACK_STEPS, HORIZON_STEPS, val_end_indices)[1]
        y_val_concat = Y_val_full[i0]
        y_pred_concat = val_preds[i0]

    plt.figure(figsize=(10, 5))
    plt.plot(y_val_concat, label="True", marker='o')
    plt.plot(y_pred_concat, label="Pred", marker='x')
    plt.title("[XGB] Validation sample (concat)")
    plt.xlabel("Hour step")
    plt.ylabel("Normalized PV")
    plt.legend(); plt.grid(True, alpha=0.3)
    fig_sample = os.path.join(FIGURES_DIR, model_base_name + "_val_sample.png")
    plt.savefig(fig_sample, dpi=150); plt.close()

    # Save config & metrics
    @dataclass
    class RunConfigXGB:
        csv_path: str
        backend: str
        resample_rule: str
        resample_agg: str
        lookback_hours: int
        horizon_hours: int
        train_ratio: float
        lookback_steps: int
        horizon_steps: int
        lag_feats: int
        time_feats_per_h: int
        seed: int
        model_base_name: str
        xgb_params: dict

    run_cfg = RunConfigXGB(
        csv_path=CSV_PATH,
        backend="xgb_direct",
        resample_rule=RESAMPLE_RULE,
        resample_agg=RESAMPLE_AGG,
        lookback_hours=LOOKBACK_HOURS,
        horizon_hours=HORIZON_HOURS,
        train_ratio=TRAIN_RATIO,
        lookback_steps=LOOKBACK_STEPS,
        horizon_steps=HORIZON_STEPS,
        lag_feats=LOOKBACK_STEPS,
        time_feats_per_h=4,
        seed=SEED,
        model_base_name=model_base_name,
        xgb_params=XGB_PARAMS
    )
    meta = {"config": asdict(run_cfg), "val_metrics": {"MAE": mae, "MSE": mse, "R2": r2}}
    with open(os.path.join(MODELS_DIR, model_base_name + "_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Latest forecast using the most recent end index from full data
    i_last = n_points - 1 - HORIZON_STEPS
    X_lag_last = make_lag_matrix_from_series(y_all, LOOKBACK_STEPS, [i_last])  # shape (1, L)
    feats_last_per_h = time_feats_for_targets(dfH.index, [i_last], HORIZON_STEPS)
    preds_last = []
    for h in range(1, HORIZON_STEPS + 1):
        X_last_h = np.hstack([X_lag_last, feats_last_per_h[h - 1]])
        preds_last.append(models_h[h - 1].predict(X_last_h)[0])
    preds_last = np.array(preds_last, dtype=np.float32)
    horizon_idx = dfH.index[i_last + 1: i_last + 1 + HORIZON_STEPS]
    df_out = pd.DataFrame({"timestamp": horizon_idx, "y_pred_xgb": preds_last})
    out_csv = os.path.join(PRED_DIR, f"{model_base_name}_latest_forecast.csv")
    df_out.to_csv(out_csv, index=False)

    print(f"\n[XGB] Saved {HORIZON_STEPS} models to: models/{model_base_name}_stepXX.json")
    print(f"[XGB] Saved config JSON to: {os.path.join(MODELS_DIR, model_base_name + '_config.json')}")
    print(f"[XGB] Saved per-step MAE plot: {fig_steps}")
    print(f"[XGB] Saved sample plot:       {fig_sample}")
    print(f"[XGB] Saved latest forecast CSV: {out_csv}")

# -------------------------
# Run selected backend
# -------------------------
if __name__ == "__main__":
    print(f">> Backend: {BACKEND}")
    if BACKEND == "lstm":
        run_lstm_branch()
    elif BACKEND == "xgb_direct":
        run_xgb_direct_branch()
    else:
        raise ValueError("Unknown BACKEND. Use 'lstm' or 'xgb_direct'.")
