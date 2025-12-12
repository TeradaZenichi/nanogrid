# -*- coding: utf-8 -*-
"""
Hourly Load Forecast (day-ahead) with LSTM or XGBoost-direct.

- Input: CSV with columns [timestamp, p_norm] at 5-minute resolution.
- Resamples to hourly, builds end-aligned windows, and uses a time-aware split.
- Backends:
    * "lstm": one sequence-to-multi-horizon LSTM model
    * "xgb_direct": H independent XGBoost regressors (direct multi-step)
- Cyclical time features: day-of-week and hour-of-day (sin/cos).
- Artifacts:
    * ./models/<name>.keras  (LSTM)  | ./models/<name>.joblib/.pkl (XGB)
    * ./models/<name>_config.json
    * ./figures/*.png
    * ./predictions/<name>_latest_forecast.csv  (full H-hour horizon, hourly)
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

# Optional XGBoost backend
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# Optional joblib for packaging XGB models
try:
    import joblib

    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False


# -------------------------
# Default XGB hyperparameters
# -------------------------
DEFAULT_XGB_PARAMS: Dict[str, Any] = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    eval_metric="rmse",
)


# -------------------------
# Config dataclasses
# -------------------------
@dataclass
class Config:
    # Data and split
    csv_path: str = "data/load_5min_train.csv"
    resample_rule: str = "H"          # hourly
    resample_agg: str = "mean"        # "mean" | "median" | "sum"
    lookback_hours: int = 8           # hours of history in input window
    horizon_hours: int = 36           # hours ahead to predict (multi-step)
    train_ratio: float = 0.75         # temporal split on the hourly index
    backend: str = "lstm"             # "lstm" or "xgb_direct"
    model_prefix: str = "load_day_ahead"

    # LSTM hyperparameters
    lstm_units_1: int = 64
    lstm_units_2: int = 64
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 80
    batch_size: int = 128
    patience: int = 10

    # XGBoost hyperparameters
    xgb_params: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_XGB_PARAMS))
    early_stop_rounds: int = 50

    # Misc
    seed: int = 42
    models_dir: str = "models"
    figures_dir: str = "figures"
    pred_dir: str = "predictions"


@dataclass
class PreparedData:
    dfH: pd.DataFrame
    X_all: np.ndarray
    y_all: np.ndarray
    feature_cols: List[str]
    train_end_indices: List[int]
    val_end_indices: List[int]
    boundary_idx: int
    lookback_steps: int
    horizon_steps: int


# -------------------------
# Utils
# -------------------------
def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dirs(cfg: Config) -> None:
    """Create output directories if they do not exist."""
    Path(cfg.models_dir).mkdir(exist_ok=True)
    Path(cfg.figures_dir).mkdir(exist_ok=True)
    Path(cfg.pred_dir).mkdir(exist_ok=True)


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical features for day-of-week and hour-of-day."""
    idx = df.index

    # Day-of-week [0..6]
    dow = idx.dayofweek.values.astype(float)
    dow_angle = 2.0 * np.pi * dow / 7.0
    df["dow_sin"] = np.sin(dow_angle)
    df["dow_cos"] = np.cos(dow_angle)

    # Hour-of-day [0..23]
    hod = idx.hour.values.astype(float)
    hod_angle = 2.0 * np.pi * hod / 24.0
    df["hod_sin"] = np.sin(hod_angle)
    df["hod_cos"] = np.cos(hod_angle)

    return df


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² over flattened arrays (multi-step)."""
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
    end_indices: List[int],
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


def make_lag_matrix_from_series(
    y_all: np.ndarray, lookback_steps: int, end_indices: List[int]
) -> np.ndarray:
    """
    Build a simple lag matrix of shape [n_samples, lookback_steps] using
    the last `lookback_steps` values before each end index.
    """
    n_samples = len(end_indices)
    X_lag = np.zeros((n_samples, lookback_steps), dtype=np.float32)

    for k, i in enumerate(end_indices):
        X_lag[k] = y_all[i - lookback_steps + 1 : i + 1]

    return X_lag


def time_feats_for_targets(
    idx: pd.DatetimeIndex, end_indices: List[int], horizon_steps: int
) -> List[np.ndarray]:
    """
    For each horizon step h, build time features at index[i+h]:
    returns a list of length horizon_steps, each entry is [n_samples, 4]
    with [dow_sin, dow_cos, hod_sin, hod_cos].
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


def plot_concat_sample(
    y_val: np.ndarray,
    y_val_pred: np.ndarray,
    horizon_steps: int,
    figures_dir: str,
    model_name: str,
    backend_tag: str,
) -> str:
    """
    Plot a concatenated sample of up to three validation horizons separated by H.
    This mimics a sequence of day-ahead (or multi-hour) forecasts along the timeline.
    """
    n_val = len(y_val)
    if n_val == 0:
        return ""

    i0 = 0
    segments_true = [y_val[i0]]
    segments_pred = [y_val_pred[i0]]

    # Add more horizons separated by H if available
    if i0 + horizon_steps < n_val:
        segments_true.append(y_val[i0 + horizon_steps])
        segments_pred.append(y_val_pred[i0 + horizon_steps])
    if i0 + 2 * horizon_steps < n_val:
        segments_true.append(y_val[i0 + 2 * horizon_steps])
        segments_pred.append(y_val_pred[i0 + 2 * horizon_steps])

    y_concat = np.concatenate(segments_true)
    y_pred_concat = np.concatenate(segments_pred)

    plt.figure(figsize=(10, 5))
    plt.plot(y_concat, label="True", marker="o")
    plt.plot(y_pred_concat, label="Pred", marker="x")
    plt.title(f"[{backend_tag.upper()}] Validation sample (3xH horizons)")
    plt.xlabel("Hour step in concatenated horizons")
    plt.ylabel("Normalized load")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = os.path.join(figures_dir, model_name + f"_{backend_tag}_val_concat.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return fig_path


# -------------------------
# Data preparation
# -------------------------
def load_and_prepare(cfg: Config) -> PreparedData:
    """
    Load 5-min load data, resample to hourly, add features, and build end indices for
    hourly multi-step forecasting.
    """
    df5 = (
        pd.read_csv(cfg.csv_path, parse_dates=["timestamp"])
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    if "p_norm" not in df5.columns:
        raise ValueError("CSV must contain column 'p_norm'.")

    # Hourly resampling
    if cfg.resample_agg == "mean":
        dfH = df5.resample(cfg.resample_rule).mean(numeric_only=True)
    elif cfg.resample_agg == "median":
        dfH = df5.resample(cfg.resample_rule).median(numeric_only=True)
    elif cfg.resample_agg == "sum":
        dfH = df5.resample(cfg.resample_rule).sum(numeric_only=True)
    else:
        raise ValueError("Unsupported resample_agg. Use 'mean' | 'median' | 'sum'.")

    dfH = dfH.dropna()
    dfH = add_cyclical_time_features(dfH)

    feature_cols = ["p_norm", "dow_sin", "dow_cos", "hod_sin", "hod_cos"]
    X_all = dfH[feature_cols].values.astype(np.float32)
    y_all = dfH["p_norm"].values.astype(np.float32)
    n_points = len(dfH)

    # For hourly data, one step = one hour
    lookback_steps = int(cfg.lookback_hours)
    horizon_steps = int(cfg.horizon_hours)

    i_min = lookback_steps - 1
    i_max = n_points - 1 - horizon_steps
    if i_max < i_min:
        raise ValueError(
            "Not enough hourly data for the chosen lookback/horizon. "
            f"(n_points={n_points}, i_min={i_min}, i_max={i_max})"
        )

    # Time-aware split: define a boundary index on the hourly timeline
    boundary_idx = int(math.floor(n_points * cfg.train_ratio)) - 1
    boundary_idx = max(boundary_idx, i_min)

    # Train: targets [i+1 ... i+H] must lie fully before or at the boundary
    train_i_min = i_min
    train_i_max = min(i_max, boundary_idx - horizon_steps)
    train_end_indices = (
        list(range(train_i_min, train_i_max + 1)) if train_i_max >= train_i_min else []
    )

    # Validation: end indices from boundary onward
    val_i_min = max(i_min, boundary_idx)
    val_i_max = i_max
    val_end_indices = (
        list(range(val_i_min, val_i_max + 1)) if val_i_max >= val_i_min else []
    )

    if (len(train_end_indices) == 0) or (len(val_end_indices) == 0):
        raise ValueError(
            "Split produced empty train/val sets. "
            f"Try reducing horizon_hours or increasing train_ratio. "
            f"(train_i_max={train_i_max}, val_i_min={val_i_min}, i_min={i_min}, i_max={i_max})"
        )

    return PreparedData(
        dfH=dfH,
        X_all=X_all,
        y_all=y_all,
        feature_cols=feature_cols,
        train_end_indices=train_end_indices,
        val_end_indices=val_end_indices,
        boundary_idx=boundary_idx,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
    )


# -------------------------
# LSTM backend
# -------------------------
def run_lstm_branch(prep: PreparedData, cfg: Config, model_name: str) -> Dict[str, Any]:
    """Train an LSTM for hourly day-ahead load forecasting."""
    X_train, y_train = build_end_aligned_windows(
        prep.X_all,
        prep.y_all,
        prep.lookback_steps,
        prep.horizon_steps,
        prep.train_end_indices,
    )
    X_val, y_val = build_end_aligned_windows(
        prep.X_all,
        prep.y_all,
        prep.lookback_steps,
        prep.horizon_steps,
        prep.val_end_indices,
    )

    split_ts = prep.dfH.index[prep.boundary_idx]
    print(
        f">> Backend LSTM | points={len(prep.dfH)} | split@{split_ts} | "
        f"lookback={prep.lookback_steps}h | horizon={prep.horizon_steps}h | "
        f"train_samples={len(prep.train_end_indices)} | val_samples={len(prep.val_end_indices)}"
    )

    input_shape = (prep.lookback_steps, X_train.shape[2])
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(cfg.lstm_units_1, return_sequences=True),
            layers.LSTM(cfg.lstm_units_2, return_sequences=False),
            layers.Dropout(cfg.dropout_rate),
            layers.Dense(prep.horizon_steps, activation="linear"),
        ]
    )

    opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(
        optimizer=opt,
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
            r2_keras,
        ],
    )

    ckpt_path = os.path.join(cfg.models_dir, model_name + ".keras")
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.patience, restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
    ]

    hist = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=cbs,
        verbose=1,
    )

    # Validation metrics (flattened across horizon)
    y_val_pred = model.predict(X_val, batch_size=cfg.batch_size, verbose=0)
    mae = float(np.mean(np.abs(y_val - y_val_pred)))
    mse = float(np.mean(np.square(y_val - y_val_pred)))
    r2 = float(r2_score_np(y_val, y_val_pred))

    print("\n[LSTM] Validation metrics (flattened across horizon):")
    print(f"MAE: {mae:.6f} | MSE: {mse:.6f} | R2: {r2:.6f}")

    # Training curve
    plt.figure(figsize=(8, 4))
    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("[LSTM] Training curve (hourly load, multi-step)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_curve = os.path.join(cfg.figures_dir, model_name + "_training_curve.png")
    plt.tight_layout()
    plt.savefig(fig_curve, dpi=150)
    plt.close()

    # Concatenated validation sample (3 horizons)
    fig_sample = plot_concat_sample(
        y_val=y_val,
        y_val_pred=y_val_pred,
        horizon_steps=prep.horizon_steps,
        figures_dir=cfg.figures_dir,
        model_name=model_name,
        backend_tag="lstm",
    )

    # Save model (without optimizer)
    model.save(ckpt_path, include_optimizer=False)

    # Save config & metrics
    run_cfg = dict(
        csv_path=cfg.csv_path,
        backend="lstm",
        resample_rule=cfg.resample_rule,
        resample_agg=cfg.resample_agg,
        lookback_hours=cfg.lookback_hours,
        horizon_hours=cfg.horizon_hours,
        train_ratio=cfg.train_ratio,
        lookback_steps=prep.lookback_steps,
        horizon_steps=prep.horizon_steps,
        features=prep.feature_cols,
        seed=cfg.seed,
        model_name=model_name,
        split_time=str(split_ts),
        n_train=len(prep.train_end_indices),
        n_val=len(prep.val_end_indices),
    )
    meta = {
        "config": run_cfg,
        "val_metrics": {"MAE": mae, "MSE": mse, "R2": r2},
        "artifacts": {
            "model_path": ckpt_path,
            "training_curve_png": fig_curve,
            "val_concat_png": fig_sample,
        },
    }
    with open(os.path.join(cfg.models_dir, model_name + "_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Latest day-ahead forecast from the last available window
    i_last = len(prep.dfH) - 1 - prep.horizon_steps
    X_last, _ = build_end_aligned_windows(
        prep.X_all, prep.y_all, prep.lookback_steps, prep.horizon_steps, [i_last]
    )
    y_last_pred = model.predict(X_last, verbose=0)[0]
    horizon_idx = prep.dfH.index[i_last + 1 : i_last + 1 + prep.horizon_steps]
    df_out = pd.DataFrame({"timestamp": horizon_idx, "y_pred_lstm": y_last_pred})
    out_csv = os.path.join(cfg.pred_dir, f"{model_name}_latest_forecast.csv")
    df_out.to_csv(out_csv, index=False)

    print(f"\n[LSTM] Saved best model to: {ckpt_path}")
    print(f"[LSTM] Saved config to:     {os.path.join(cfg.models_dir, model_name + '_config.json')}")
    print(f"[LSTM] Saved plots to:      {fig_curve}, {fig_sample}")
    print(f"[LSTM] Latest forecast CSV: {out_csv}")

    return {"MAE": mae, "MSE": mse, "R2": r2, "ckpt_path": ckpt_path}


# -------------------------
# XGBoost (direct multi-step) backend
# -------------------------
def run_xgb_direct_branch(
    prep: PreparedData, cfg: Config, model_name: str
) -> Dict[str, Any]:
    """Train H independent XGBoost models (one per horizon hour)."""
    if not XGB_AVAILABLE:
        raise ImportError("xgboost not available. Install with: pip install xgboost")

    # Lag features (from the load series) for train/val
    X_lag_tr = make_lag_matrix_from_series(
        prep.y_all, prep.lookback_steps, prep.train_end_indices
    )
    X_lag_va = make_lag_matrix_from_series(
        prep.y_all, prep.lookback_steps, prep.val_end_indices
    )

    # Time features at target timestamps for each horizon
    feats_tr_per_h = time_feats_for_targets(
        prep.dfH.index, prep.train_end_indices, prep.horizon_steps
    )
    feats_va_per_h = time_feats_for_targets(
        prep.dfH.index, prep.val_end_indices, prep.horizon_steps
    )

    # Full validation targets for metrics/plots
    _, Y_val_full = build_end_aligned_windows(
        prep.X_all,
        prep.y_all,
        prep.lookback_steps,
        prep.horizon_steps,
        prep.val_end_indices,
    )

    model_ext = ".joblib" if JOBLIB_AVAILABLE else ".pkl"
    ckpt_path = os.path.join(cfg.models_dir, model_name + model_ext)

    xgb_params = dict(cfg.xgb_params)
    xgb_params["verbosity"] = 0

    models_h = []
    val_preds = np.zeros(
        (len(prep.val_end_indices), prep.horizon_steps), dtype=np.float32
    )
    step_mae, step_rmse = [], []

    split_ts = prep.dfH.index[prep.boundary_idx]
    print(
        f">> Backend XGB | points={len(prep.dfH)} | split@{split_ts} | "
        f"lookback={prep.lookback_steps}h | horizon={prep.horizon_steps}h | "
        f"train_samples={len(prep.train_end_indices)} | val_samples={len(prep.val_end_indices)}"
    )
    print(f"   Training {prep.horizon_steps} direct models...")

    progress_every = max(1, prep.horizon_steps // 6)

    for h in range(1, prep.horizon_steps + 1):
        # Feature matrices for this horizon step
        X_tr_h = np.hstack([X_lag_tr, feats_tr_per_h[h - 1]])
        X_va_h = np.hstack([X_lag_va, feats_va_per_h[h - 1]])

        # Targets at horizon h
        y_tr_h = np.array(
            [prep.y_all[i + h] for i in prep.train_end_indices], dtype=np.float32
        )
        y_va_h = np.array(
            [prep.y_all[i + h] for i in prep.val_end_indices], dtype=np.float32
        )

        model_h = xgb.XGBRegressor(**xgb_params)
        model_h.fit(
            X_tr_h,
            y_tr_h,
            eval_set=[(X_va_h, y_va_h)],
            verbose=False,
        )
        models_h.append(model_h)

        # Validation predictions for this step
        val_preds[:, h - 1] = model_h.predict(X_va_h)
        mae_h = float(np.mean(np.abs(y_va_h - val_preds[:, h - 1])))
        rmse_h = float(np.sqrt(np.mean((y_va_h - val_preds[:, h - 1]) ** 2)))
        step_mae.append(mae_h)
        step_rmse.append(rmse_h)

        if (h % progress_every == 0) or (h == prep.horizon_steps):
            print(f"   ... {h}/{prep.horizon_steps} models trained")

    # Aggregate validation metrics
    mae = float(np.mean(np.abs(Y_val_full - val_preds)))
    mse = float(np.mean(np.square(Y_val_full - val_preds)))
    r2 = float(r2_score_np(Y_val_full, val_preds))

    print("\n[XGB] Validation metrics (flattened across horizon):")
    print(f"MAE: {mae:.6f} | MSE: {mse:.6f} | R2: {r2:.6f}")

    # MAE per step
    plt.figure(figsize=(9, 4))
    plt.plot(np.arange(1, prep.horizon_steps + 1), step_mae, marker="o")
    plt.xlabel("Horizon step (hour)")
    plt.ylabel("MAE")
    plt.title("[XGB] Validation MAE per step (hourly load)")
    plt.grid(True, alpha=0.3)
    fig_steps = os.path.join(cfg.figures_dir, model_name + "_val_mae_per_step.png")
    plt.tight_layout()
    plt.savefig(fig_steps, dpi=150)
    plt.close()

    # Concatenated sample (3 horizons)
    fig_sample = plot_concat_sample(
        y_val=Y_val_full,
        y_val_pred=val_preds,
        horizon_steps=prep.horizon_steps,
        figures_dir=cfg.figures_dir,
        model_name=model_name,
        backend_tag="xgb",
    )

    # Save XGB models
    if JOBLIB_AVAILABLE:
        joblib.dump(models_h, ckpt_path)
    else:
        import pickle

        with open(ckpt_path, "wb") as f:
            pickle.dump(models_h, f)

    # Save config & metrics
    run_cfg = dict(
        csv_path=cfg.csv_path,
        backend="xgb_direct",
        resample_rule=cfg.resample_rule,
        resample_agg=cfg.resample_agg,
        lookback_hours=cfg.lookback_hours,
        horizon_hours=cfg.horizon_hours,
        train_ratio=cfg.train_ratio,
        lookback_steps=prep.lookback_steps,
        horizon_steps=prep.horizon_steps,
        lag_feats=prep.lookback_steps,
        time_feats_per_h=4,
        seed=cfg.seed,
        model_name=model_name,
        xgb_params=xgb_params,
        split_sizes=dict(
            n_points=int(len(prep.dfH)),
            n_train=int(len(prep.train_end_indices)),
            n_val=int(len(prep.val_end_indices)),
            boundary_time=str(split_ts),
        ),
    )
    meta = {
        "config": run_cfg,
        "val_metrics": {"MAE": mae, "MSE": mse, "R2": r2},
        "per_step": {"MAE": step_mae, "RMSE": step_rmse},
        "artifacts": {
            "model_path": ckpt_path,
            "val_mae_per_step_png": fig_steps,
            "val_concat_png": fig_sample,
        },
    }
    with open(os.path.join(cfg.models_dir, model_name + "_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Latest day-ahead forecast from the last available window
    i_last = len(prep.dfH) - 1 - prep.horizon_steps
    X_lag_last = make_lag_matrix_from_series(
        prep.y_all, prep.lookback_steps, [i_last]
    )
    feats_last_per_h = time_feats_for_targets(
        prep.dfH.index, [i_last], prep.horizon_steps
    )

    preds_last = []
    for h in range(1, prep.horizon_steps + 1):
        X_last_h = np.hstack([X_lag_last, feats_last_per_h[h - 1]])
        preds_last.append(models_h[h - 1].predict(X_last_h)[0])
    preds_last = np.array(preds_last, dtype=np.float32)

    horizon_idx = prep.dfH.index[i_last + 1 : i_last + 1 + prep.horizon_steps]
    df_out = pd.DataFrame({"timestamp": horizon_idx, "y_pred_xgb": preds_last})
    out_csv = os.path.join(cfg.pred_dir, f"{model_name}_latest_forecast.csv")
    df_out.to_csv(out_csv, index=False)

    print(f"\n[XGB] Saved models to:       {ckpt_path}")
    print(f"[XGB] Saved config to:       {os.path.join(cfg.models_dir, model_name + '_config.json')}")
    print(f"[XGB] Saved MAE-per-step to: {fig_steps}")
    print(f"[XGB] Saved concat plot to:  {fig_sample}")
    print(f"[XGB] Latest forecast CSV:   {out_csv}")

    return {"MAE": mae, "MSE": mse, "R2": r2, "ckpt_path": ckpt_path}


# -------------------------
# CLI helpers
# -------------------------
def build_model_name(cfg: Config) -> str:
    backend_tag = "lstm" if cfg.backend == "lstm" else "xgb"
    return f"{cfg.model_prefix}_{backend_tag}_{cfg.horizon_hours}hH_{cfg.lookback_hours}hLB"


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Hourly day-ahead load forecast (LSTM or XGBoost-direct)."
    )
    parser.add_argument("--backend", choices=["lstm", "xgb_direct"], default=Config.backend)
    parser.add_argument("--csv-path", default=Config.csv_path)
    parser.add_argument("--resample-rule", default=Config.resample_rule)
    parser.add_argument("--resample-agg", default=Config.resample_agg)
    parser.add_argument("--lookback-hours", type=int, default=Config.lookback_hours)
    parser.add_argument("--horizon-hours", type=int, default=Config.horizon_hours)
    parser.add_argument("--train-ratio", type=float, default=Config.train_ratio)
    parser.add_argument("--model-prefix", default=Config.model_prefix)
    parser.add_argument("--seed", type=int, default=Config.seed)

    # LSTM overrides
    parser.add_argument("--lstm-units-1", type=int, default=Config.lstm_units_1)
    parser.add_argument("--lstm-units-2", type=int, default=Config.lstm_units_2)
    parser.add_argument("--dropout-rate", type=float, default=Config.dropout_rate)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--patience", type=int, default=Config.patience)

    # XGB overrides
    parser.add_argument(
        "--xgb-n-estimators", type=int, default=DEFAULT_XGB_PARAMS["n_estimators"]
    )
    parser.add_argument(
        "--xgb-learning-rate", type=float, default=DEFAULT_XGB_PARAMS["learning_rate"]
    )
    parser.add_argument(
        "--xgb-max-depth", type=int, default=DEFAULT_XGB_PARAMS["max_depth"]
    )
    parser.add_argument(
        "--xgb-early-stop-rounds", type=int, default=Config.early_stop_rounds
    )

    args = parser.parse_args()

    cfg = Config(
        csv_path=args.csv_path,
        resample_rule=args.resample_rule,
        resample_agg=args.resample_agg,
        lookback_hours=args.lookback_hours,
        horizon_hours=args.horizon_hours,
        train_ratio=args.train_ratio,
        backend=args.backend,
        model_prefix=args.model_prefix,
        lstm_units_1=args.lstm_units_1,
        lstm_units_2=args.lstm_units_2,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
    )

    # Update XGB params
    cfg.xgb_params.update(
        n_estimators=args.xgb_n_estimators,
        learning_rate=args.xgb_learning_rate,
        max_depth=args.xgb_max_depth,
    )
    cfg.early_stop_rounds = args.xgb_early_stop_rounds

    return cfg


# -------------------------
# Entrypoint
# -------------------------
def main() -> None:
    cfg = parse_args()
    ensure_dirs(cfg)
    set_seeds(cfg.seed)

    prep = load_and_prepare(cfg)
    model_name = build_model_name(cfg)

    print(
        f">> LOAD DAY-AHEAD | hourly points={len(prep.dfH)} | split@{prep.dfH.index[prep.boundary_idx]} | "
        f"lookback={prep.lookback_steps}h | horizon={prep.horizon_steps}h | backend={cfg.backend}"
    )
    print(
        f"                   train_samples={len(prep.train_end_indices)} | "
        f"val_samples={len(prep.val_end_indices)}"
    )

    if cfg.backend == "lstm":
        metrics = run_lstm_branch(prep, cfg, model_name)
    elif cfg.backend == "xgb_direct":
        metrics = run_xgb_direct_branch(prep, cfg, model_name)
    else:
        raise ValueError("Unknown backend. Use 'lstm' or 'xgb_direct'.")

    print(f"\nBest checkpoint: {metrics['ckpt_path']}")
    print(f"Validation metrics: {metrics}")


if __name__ == "__main__":
    main()
