# -*- coding: utf-8 -*-
"""
VSTF PV Forecast (5-min data → next hour = 12 steps) with LSTM or XGBoost (direct)

What this script does
---------------------
- Reads a 5-min PV CSV with columns: [timestamp, p_norm].
- Builds end-aligned windows: X[i-LB+1:i+1], y[i+1:i+1+H].
- Time-aware split (no horizon leakage).
- Two backends:
    * "lstm": one seq2seq model that outputs H future steps at once.
    * "xgb_direct": H independent XGBoost regressors, one per step h=1..H.
- Adds cyclical time features (DOW and time-of-day).
- Saves:
    * Best model(s) to ./models
    * Training/sample plots to ./figures
    * Config + validation metrics JSON
    * Latest H-step forecast CSV to ./predictions

Author: You
"""

import os, json, math, time
from dataclasses import dataclass, asdict
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TensorFlow (for LSTM)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# XGBoost (optional, for xgb_direct)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# -------------------------
# Config (edit here)
# -------------------------
CSV_PATH = "data/pv_5min_train.csv"  # <-- PV file with [timestamp,p_norm]

BACKEND = "lstm"                # "lstm" or "xgb_direct"

HORIZON_MINUTES = 60            # forecast horizon (next hour) -> 12 steps @5min
LOOKBACK_HOURS  = 4.0           # hours of history in input
TRAIN_RATIO     = 0.75          # temporal split

# LSTM hyperparameters
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 128
PATIENCE = 10

# XGBoost (only used if BACKEND == "xgb_direct")
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
    n_jobs=0,   # use all cores
)
EARLY_STOP_ROUNDS = 50

SEED = 42

# Output dirs
MODELS_DIR = "models"
FIGURES_DIR = "figures"
PRED_DIR = "predictions"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

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

def make_lag_matrix_from_series(y_all: np.ndarray, lookback_steps: int, end_indices: List[int]) -> np.ndarray:
    """Return lag matrix for tree models: row k = [y[i-L+1], ..., y[i]]."""
    n_samples = len(end_indices)
    X_lag = np.zeros((n_samples, lookback_steps), dtype=np.float32)
    for k, i in enumerate(end_indices):
        X_lag[k] = y_all[i - lookback_steps + 1: i + 1]
    return X_lag

def time_feats_for_targets(idx: pd.DatetimeIndex, end_indices: List[int], horizon_steps: int) -> List[np.ndarray]:
    """
    For each horizon h=1..H, build a (n_samples, 4) matrix with [dow_sin, dow_cos, tod_sin, tod_cos]
    for the target timestamp idx[i+h].
    """
    n_samples = len(end_indices)
    feats_per_h = []
    for h in range(1, horizon_steps + 1):
        dow_s = np.zeros((n_samples,), dtype=np.float32)
        dow_c = np.zeros((n_samples,), dtype=np.float32)
        tod_s = np.zeros((n_samples,), dtype=np.float32)
        tod_c = np.zeros((n_samples,), dtype=np.float32)
        for k, i in enumerate(end_indices):
            t = idx[i + h]
            dow = float(t.dayofweek)
            minutes = float(t.hour * 60 + t.minute)
            dow_ang = 2.0 * np.pi * dow / 7.0
            tod_ang = 2.0 * np.pi * minutes / (24.0 * 60.0)
            dow_s[k] = np.sin(dow_ang)
            dow_c[k] = np.cos(dow_ang)
            tod_s[k] = np.sin(tod_ang)
            tod_c[k] = np.cos(tod_ang)
        feats_per_h.append(np.stack([dow_s, dow_c, tod_s, tod_c], axis=1))
    return feats_per_h

def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Flattened R^2 over all steps and samples."""
    yt, yp = y_true.reshape(-1), y_pred.reshape(-1)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot

@tf.function
def r2_keras(y_true, y_pred):
    """Keras R^2 metric for convenience in training logs."""
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    mean_y = tf.reduce_mean(y_true)
    ss_tot = tf.reduce_sum(tf.square(y_true - mean_y))
    return 1.0 - ss_res / (ss_tot + tf.keras.backend.epsilon())

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

# Build tensors for LSTM and Y matrices for metrics/plots
X_train_seq, y_train_seq = build_end_aligned_windows(X_all, y_all, LB, H, train_end_indices)
X_val_seq,   y_val_seq   = build_end_aligned_windows(X_all, y_all, LB, H, val_end_indices)

print(f"Step: {STEP_MIN} min | Lookback steps: {LB} | Horizon steps: {H}")
print(f"Points: {n_points} | Boundary idx: {boundary_idx} (time={df.index[boundary_idx]})")
print(f"Train samples: {len(train_end_indices)} | Val samples: {len(val_end_indices)}")
print(f"Backend: {BACKEND}")

# -------------------------
# LSTM backend
# -------------------------
def run_lstm():
    input_shape = (LB, X_train_seq.shape[2])
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

    model_name = f"lstm_vstf_pv_{STEP_MIN}min_{HORIZON_MINUTES}min_{LB}"
    ckpt_path  = os.path.join(MODELS_DIR, model_name + ".keras")

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True)
    ]

    hist = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        verbose=1
    )

    # Evaluate
    y_val_pred = model.predict(X_val_seq, batch_size=BATCH_SIZE, verbose=0)
    mae = float(np.mean(np.abs(y_val_seq - y_val_pred)))
    mse = float(np.mean((y_val_seq - y_val_pred) ** 2))
    r2  = float(r2_score_np(y_val_seq, y_val_pred))
    print("\n[LSTM] Validation metrics (flattened over horizon):")
    print(f"MAE: {mae:.6f} | MSE: {mse:.6f} | R2: {r2:.6f}")

    # Plot: 3 validation samples spaced by H (safeguard for length)
    i0 = min(12*7, len(y_val_seq) - (2*H + 1)) if len(y_val_seq) > (2*H) else 0
    y_val_concat = np.concatenate([y_val_seq[i0],
                                   y_val_seq[i0 + H] if i0 + H < len(y_val_seq) else y_val_seq[i0],
                                   y_val_seq[i0 + 2*H] if i0 + 2*H < len(y_val_seq) else y_val_seq[i0]])
    y_pred_concat = np.concatenate([y_val_pred[i0],
                                    y_val_pred[i0 + H] if i0 + H < len(y_val_pred) else y_val_pred[i0],
                                    y_val_pred[i0 + 2*H] if i0 + 2*H < len(y_val_pred) else y_val_pred[i0]])

    plt.figure(figsize=(10, 5))
    plt.plot(y_val_concat, label="True", marker='o')
    plt.plot(y_pred_concat, label="Pred", marker='x')
    plt.title(f"[LSTM] Validation sample (first +H +2H windows)")
    plt.xlabel("5-min steps")
    plt.ylabel("Normalized PV")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_sample = os.path.join(FIGURES_DIR, model_name + "_val_sample_VSTF.png")
    plt.savefig(fig_sample, dpi=150); plt.close()

    # Training curve
    plt.figure(figsize=(8, 4))
    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("[LSTM] Training Curve (VSTF)")
    plt.legend(); plt.grid(True, alpha=0.3)
    fig_curve = os.path.join(FIGURES_DIR, model_name + "_training_curve.png")
    plt.tight_layout(); plt.savefig(fig_curve, dpi=150); plt.close()

    # Save model WITHOUT optimizer
    model.save(ckpt_path, include_optimizer=False)

    # Save config & metrics
    @dataclass
    class RunCfg:
        csv_path: str
        backend: str
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
        backend="lstm",
        horizon_minutes=HORIZON_MINUTES,
        lookback_hours=LOOKBACK_HOURS,
        train_ratio=TRAIN_RATIO,
        step_minutes=STEP_MIN,
        lookback_steps=LB,
        horizon_steps=H,
        features=feature_cols,
        seed=SEED,
        model_name=model_name
    )
    with open(os.path.join(MODELS_DIR, model_name + "_config.json"), "w") as f:
        json.dump({"config": asdict(run_cfg),
                   "val_metrics": {"MAE": mae, "MSE": mse, "R2": r2}}, f, indent=2)

    # Latest H-step forecast using the most recent end index
    i_last = n_points - 1 - H
    X_last, _ = build_end_aligned_windows(X_all, y_all, LB, H, [i_last])
    y_last_pred = model.predict(X_last, verbose=0)[0]
    horizon_idx = df.index[i_last + 1: i_last + 1 + H]
    df_out = pd.DataFrame({"timestamp": horizon_idx, "y_pred_lstm": y_last_pred})
    out_csv = os.path.join(PRED_DIR, f"{model_name}_latest_forecast.csv")
    df_out.to_csv(out_csv, index=False)

    print(f"\n[LSTM] Saved best model to: {ckpt_path}")
    print(f"[LSTM] Saved config to:     {os.path.join(MODELS_DIR, model_name + '_config.json')}")
    print(f"[LSTM] Saved plots to:      {fig_curve} and {fig_sample}")
    print(f"[LSTM] Saved latest forecast CSV: {out_csv}")

# -------------------------
# XGBoost (direct multi-step) backend
# -------------------------
def run_xgb_direct():
    if not XGB_AVAILABLE:
        raise ImportError("xgboost not available. Install with: pip install xgboost")

    # Lag features (from the PV target) for train/val
    X_lag_tr = make_lag_matrix_from_series(y_all, LB, train_end_indices)
    X_lag_va = make_lag_matrix_from_series(y_all, LB, val_end_indices)

    # Time features at target timestamps for each horizon
    feats_tr_per_h = time_feats_for_targets(df.index, train_end_indices, H)
    feats_va_per_h = time_feats_for_targets(df.index, val_end_indices, H)

    # Targets (ground truth) for val (full H matrix) for metrics
    Y_val = y_val_seq  # already built
    models_h = []
    val_preds = np.zeros((len(val_end_indices), H), dtype=np.float32)
    train_preds = np.zeros((len(train_end_indices), H), dtype=np.float32)

    model_base_name = f"xgb_vstf_pv_{STEP_MIN}min_{HORIZON_MINUTES}min_{LB}"
    step_mae, step_rmse = [], []

    print(f"[XGB] Training {H} direct models (one per 5-min step)...")

    for h in range(1, H + 1):
        # Features: lags + time feats at target time (i+h)
        X_tr_h = np.hstack([X_lag_tr, feats_tr_per_h[h - 1]]) if len(X_lag_tr) > 0 else feats_tr_per_h[h - 1]
        X_va_h = np.hstack([X_lag_va, feats_va_per_h[h - 1]]) if len(X_lag_va) > 0 else feats_va_per_h[h - 1]

        # Targets for horizon h
        y_tr_h = np.array([y_all[i + h] for i in train_end_indices], dtype=np.float32)
        y_va_h = np.array([y_all[i + h] for i in val_end_indices], dtype=np.float32)

        model_h = xgb.XGBRegressor(**XGB_PARAMS)
        model_h.fit(
            X_tr_h, y_tr_h,
            eval_set=[(X_va_h, y_va_h)],
            verbose=False,
            early_stopping_rounds=EARLY_STOP_ROUNDS
        )
        models_h.append(model_h)

        train_preds[:, h - 1] = model_h.predict(X_tr_h)
        val_preds[:,  h - 1]  = model_h.predict(X_va_h)

        mae_h = float(np.mean(np.abs(y_va_h - val_preds[:, h - 1])))
        rmse_h = float(np.sqrt(np.mean((y_va_h - val_preds[:, h - 1]) ** 2)))
        step_mae.append(mae_h); step_rmse.append(rmse_h)

        # Save each horizon model
        model_path = os.path.join(MODELS_DIR, f"{model_base_name}_step{h:02d}.json")
        model_h.get_booster().save_model(model_path)

    # Aggregate metrics across all horizons
    mae = float(np.mean(np.abs(Y_val - val_preds)))
    mse = float(np.mean((Y_val - val_preds) ** 2))
    r2  = float(r2_score_np(Y_val, val_preds))

    print("\n[XGB] Validation metrics (flattened over horizon):")
    print(f"MAE: {mae:.6f} | MSE: {mse:.6f} | R2: {r2:.6f}")

    # Plot per-step MAE curve
    plt.figure(figsize=(9, 4))
    plt.plot(np.arange(1, H + 1), step_mae, marker='o')
    plt.xlabel("Horizon step (5-min steps)")
    plt.ylabel("MAE")
    plt.title("[XGB] Validation MAE per step")
    plt.grid(True, alpha=0.3)
    fig_steps = os.path.join(FIGURES_DIR, model_base_name + "_val_mae_per_step.png")
    plt.tight_layout(); plt.savefig(fig_steps, dpi=150); plt.close()

    # Plot one concatenated validation sample (first +H +2H windows)
    i0 = 0
    y_val_concat = np.concatenate([Y_val[i0],
                                   Y_val[i0 + H] if i0 + H < len(Y_val) else Y_val[i0],
                                   Y_val[i0 + 2*H] if i0 + 2*H < len(Y_val) else Y_val[i0]])
    y_pred_concat = np.concatenate([val_preds[i0],
                                    val_preds[i0 + H] if i0 + H < len(val_preds) else val_preds[i0],
                                    val_preds[i0 + 2*H] if i0 + 2*H < len(val_preds) else val_preds[i0]])
    plt.figure(figsize=(10, 5))
    plt.plot(y_val_concat, label="True", marker='o')
    plt.plot(y_pred_concat, label="Pred", marker='x')
    plt.title("[XGB] Validation sample (first +H +2H windows)")
    plt.xlabel("5-min steps")
    plt.ylabel("Normalized PV")
    plt.legend(); plt.grid(True, alpha=0.3)
    fig_sample = os.path.join(FIGURES_DIR, model_base_name + "_val_sample_VSTF.png")
    plt.savefig(fig_sample, dpi=150); plt.close()

    # Save config & metrics
    @dataclass
    class RunCfgXGB:
        csv_path: str
        backend: str
        horizon_minutes: int
        lookback_hours: float
        train_ratio: float
        step_minutes: int
        lookback_steps: int
        horizon_steps: int
        lag_feats: int
        time_feats_per_h: int
        seed: int
        model_base_name: str
        xgb_params: dict

    run_cfg = RunCfgXGB(
        csv_path=CSV_PATH,
        backend="xgb_direct",
        horizon_minutes=HORIZON_MINUTES,
        lookback_hours=LOOKBACK_HOURS,
        train_ratio=TRAIN_RATIO,
        step_minutes=STEP_MIN,
        lookback_steps=LB,
        horizon_steps=H,
        lag_feats=LB,
        time_feats_per_h=4,
        seed=SEED,
        model_base_name=model_base_name,
        xgb_params=XGB_PARAMS
    )
    with open(os.path.join(MODELS_DIR, model_base_name + "_config.json"), "w") as f:
        json.dump({"config": asdict(run_cfg),
                   "val_metrics": {"MAE": mae, "MSE": mse, "R2": r2}}, f, indent=2)

    # Latest H-step forecast from the most recent end index
    i_last = n_points - 1 - H
    X_lag_last = make_lag_matrix_from_series(y_all, LB, [i_last])  # (1, LB)
    feats_last_per_h = time_feats_for_targets(df.index, [i_last], H)
    preds_last = []
    for h in range(1, H + 1):
        X_last_h = np.hstack([X_lag_last, feats_last_per_h[h - 1]])
        preds_last.append(models_h[h - 1].predict(X_last_h)[0])
    preds_last = np.array(preds_last, dtype=np.float32)
    horizon_idx = df.index[i_last + 1: i_last + 1 + H]
    df_out = pd.DataFrame({"timestamp": horizon_idx, "y_pred_xgb": preds_last})
    out_csv = os.path.join(PRED_DIR, f"{model_base_name}_latest_forecast.csv")
    df_out.to_csv(out_csv, index=False)

    print(f"\n[XGB] Saved models to: models/{model_base_name}_stepXX.json")
    print(f"[XGB] Saved config to: {os.path.join(MODELS_DIR, model_base_name + '_config.json')}")
    print(f"[XGB] Saved plots to:  {fig_steps} and {fig_sample}")
    print(f"[XGB] Saved latest forecast CSV: {out_csv}")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    if BACKEND == "lstm":
        run_lstm()
    elif BACKEND == "xgb_direct":
        run_xgb_direct()
    else:
        raise ValueError("Unknown BACKEND. Use 'lstm' or 'xgb_direct'.")