# -*- coding: utf-8 -*-
"""
VSTF PV Forecast (5-min data → next interval, H minutes ahead) with LSTM or XGBoost (direct).

- Reads 5-min PV CSV with [timestamp, p_norm].
- End-aligned windows: X[i-LB+1:i+1], y[i+1:i+1+H_steps].
- Time-aware split (no horizon leakage).
- Backends:
    * "lstm": one seq2seq model outputs H_steps future values.
    * "xgb_direct": H_steps independent XGBoost regressors (direct multi-step).
- Cyclical time features (DOW + time-of-day).
- The parameter `horizon_minutes` controls how many hours ahead the "next interval"
  forecast refers to. For example, horizon_minutes = 120 corresponds to a 2 h-ahead
  next-interval forecast.
- Saves:
    * ./models/<name>.keras  (LSTM)  | ./models/<name>.joblib (XGB; .pkl fallback)
    * ./models/<name>_config.json
    * ./figures/*.png
    * ./predictions/<name>_latest_forecast.csv
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
# Configurable defaults
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
    n_jobs=-1,  # all cores
    verbosity=0,
    eval_metric="rmse",
)


@dataclass
class Config:
    # Data and split
    csv_path: str = "data/pv_5min_train.csv"
    backend: str = "lstm"  # "lstm" or "xgb_direct"
    horizon_minutes: int = 120   # how far ahead the "next interval" lies
    lookback_hours: float = 4.0  # history window (hours)
    train_ratio: float = 0.75    # temporal split
    model_prefix: str = "pv_vstf"

    # LSTM hyperparameters
    lstm_units_1: int = 64
    lstm_units_2: int = 32
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 10
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
    plot_hours: float = 24.0  # horizon over which we show next-interval predictions


@dataclass
class PreparedData:
    df: pd.DataFrame
    X_all: np.ndarray
    y_all: np.ndarray
    feature_cols: List[str]
    train_end_indices: List[int]
    val_end_indices: List[int]
    boundary_idx: int
    lookback_steps: int
    horizon_steps: int
    step_minutes: int


# -------------------------
# Utils
# -------------------------
def set_seeds(seed: int) -> None:
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dirs(cfg: Config) -> None:
    """Ensure output directories exist."""
    Path(cfg.models_dir).mkdir(exist_ok=True)
    Path(cfg.figures_dir).mkdir(exist_ok=True)
    Path(cfg.pred_dir).mkdir(exist_ok=True)


def infer_step_minutes(index: pd.DatetimeIndex) -> int:
    """Infer the data resolution in minutes from the time index."""
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        raise ValueError("Cannot infer step minutes from an empty or single-point index.")
    step = diffs.mode().iloc[0]
    step_min = int(step.total_seconds() // 60)
    if step_min <= 0:
        raise ValueError("Could not infer a positive step size from timestamps.")
    return step_min


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical features: day-of-week and time-of-day (sin/cos)."""
    idx = df.index
    # Day-of-week
    dow = idx.dayofweek.values.astype(float)
    dow_angle = 2.0 * np.pi * dow / 7.0
    df["dow_sin"] = np.sin(dow_angle)
    df["dow_cos"] = np.cos(dow_angle)

    # Time-of-day in minutes (0..1439)
    tod_min = (idx.hour * 60 + idx.minute).values.astype(float)
    tod_angle = 2.0 * np.pi * tod_min / (24.0 * 60.0)
    df["tod_sin"] = np.sin(tod_angle)
    df["tod_cos"] = np.cos(tod_angle)
    return df


def build_end_aligned_windows(
    X: np.ndarray,
    y: np.ndarray,
    lookback_steps: int,
    horizon_steps: int,
    end_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build end-aligned windows for sequence-to-sequence learning.

    For each end index i in `end_indices`:
      - Input:   X[i-lookback_steps+1 : i+1, :]
      - Targets: y[i+1 : i+1+horizon_steps]
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
    the last `lookback_steps` values of the series before each end index.
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
    Build time-of-week features (dow/time-of-day sine/cosine) for each horizon step.
    Returns a list with length = horizon_steps, where each element is a
    matrix [n_samples, 4].
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
    """Flattened R² over all steps and samples."""
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


def plot_next_interval_over_24h(
    df: pd.DataFrame,
    val_end_indices: List[int],
    y_true_seq: np.ndarray,
    y_pred_seq: np.ndarray,
    horizon_steps: int,
    step_minutes: int,
    figures_dir: str,
    model_name: str,
    backend_tag: str,
    plot_hours: float = 24.0,
) -> str:
    """
    Plot next-interval forecasts (for a fixed horizon in minutes) over ~24 h.

    For a given horizon H (in steps), we consider y[..., H-1] as the next-interval
    forecast at +H*step_minutes minutes ahead. We then sample windows along the
    validation set every H end-indices, so that consecutive forecast timestamps
    are spaced by approximately H*step_minutes minutes.

    The x-axis shows the actual target timestamps, and the y-axis compares
    true vs predicted PV for that horizon.
    """
    n_val = len(val_end_indices)
    if n_val == 0:
        return ""

    ahead_hours = horizon_steps * step_minutes / 60.0
    if ahead_hours <= 0:
        return ""

    # Number of points to cover roughly `plot_hours` with spacing = ahead_hours
    max_points = int(plot_hours / ahead_hours)
    if max_points < 1:
        max_points = 1

    stride = horizon_steps  # one point every "ahead_hours"
    positions = [k * stride for k in range(max_points) if k * stride < n_val]
    if not positions:
        positions = [0]

    times = []
    true_vals = []
    pred_vals = []

    for pos in positions:
        end_idx = val_end_indices[pos]
        # Target timestamp for horizon step H-1 (i + H)
        target_idx = end_idx + horizon_steps
        if target_idx >= len(df.index):
            # Safety check, should not happen given how indices are built
            continue
        t = df.index[target_idx]
        times.append(t)
        true_vals.append(y_true_seq[pos, horizon_steps - 1])
        pred_vals.append(y_pred_seq[pos, horizon_steps - 1])

    if not times:
        return ""

    plt.figure(figsize=(10, 4))
    plt.plot(times, true_vals, label="True", marker="o", linestyle="-")
    plt.plot(times, pred_vals, label="Pred", marker="x", linestyle="--")
    plt.title(
        f"[{backend_tag.upper()}] Next-interval forecast over "
        f"{plot_hours:.0f} h (ahead = {ahead_hours:.1f} h)"
    )
    plt.xlabel("Timestamp")
    plt.ylabel("Normalized PV")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()

    fname = os.path.join(
        figures_dir,
        f"{model_name}_{backend_tag}_next_interval_{int(ahead_hours)}h_over_{int(plot_hours)}h.png",
    )
    plt.savefig(fname, dpi=150)
    plt.close()

    return fname


# -------------------------
# Data preparation
# -------------------------
def load_and_prepare(cfg: Config) -> PreparedData:
    """
    Load 5-min PV data, add time features, and build end indices for VSTF.

    The series is not resampled; instead, the native resolution is inferred
    from the timestamp differences (expected: 5 minutes).
    """
    df = (
        pd.read_csv(cfg.csv_path, parse_dates=["timestamp"])
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    if "p_norm" not in df.columns:
        raise ValueError("CSV must contain column 'p_norm'.")

    step_minutes = infer_step_minutes(df.index)
    if cfg.horizon_minutes % step_minutes != 0:
        raise ValueError(
            f"horizon_minutes ({cfg.horizon_minutes}) must be a multiple of the "
            f"time step ({step_minutes})."
        )

    horizon_steps = cfg.horizon_minutes // step_minutes
    lookback_steps = int(round(cfg.lookback_hours * 60.0 / step_minutes))
    if lookback_steps < 1:
        raise ValueError("lookback_hours must yield at least one lookback step.")

    df = add_cyclical_time_features(df)
    feature_cols = ["p_norm", "dow_sin", "dow_cos", "tod_sin", "tod_cos"]
    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df["p_norm"].values.astype(np.float32)
    n_points = len(df)

    # Valid end indices
    i_min = lookback_steps - 1
    i_max = n_points - 1 - horizon_steps
    if i_max < i_min:
        raise ValueError(
            "Not enough data for the chosen lookback/horizon (VSTF 5-min series)."
        )

    # Temporal split boundary (on raw points)
    boundary_idx = int(math.floor(n_points * cfg.train_ratio)) - 1
    boundary_idx = max(boundary_idx, i_min)

    # Training indices: horizon entirely before or at boundary_idx
    train_i_min = i_min
    train_i_max = min(i_max, boundary_idx - horizon_steps)
    train_end_indices = (
        list(range(train_i_min, train_i_max + 1)) if train_i_max >= train_i_min else []
    )

    # Validation indices: starting at boundary_idx
    val_i_min = max(i_min, boundary_idx)
    val_i_max = i_max
    val_end_indices = (
        list(range(val_i_min, val_i_max + 1)) if val_i_max >= val_i_min else []
    )

    if (len(train_end_indices) == 0) or (len(val_end_indices) == 0):
        raise ValueError(
            "Split produced empty train/val sets. Increase train_ratio, reduce "
            "horizon_minutes, or use a longer time series."
        )

    return PreparedData(
        df=df,
        X_all=X_all,
        y_all=y_all,
        feature_cols=feature_cols,
        train_end_indices=train_end_indices,
        val_end_indices=val_end_indices,
        boundary_idx=boundary_idx,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        step_minutes=step_minutes,
    )


# -------------------------
# Models
# -------------------------
def run_lstm_branch(prep: PreparedData, cfg: Config, model_name: str) -> Dict[str, Any]:
    """
    Train an LSTM to map a sequence of past 5-min values to a multi-step
    forecast for the next interval (H minutes ahead).
    """
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

    split_ts = prep.df.index[prep.boundary_idx]
    ahead_hours = prep.horizon_steps * prep.step_minutes / 60.0
    print(
        f">> Backend LSTM | step={prep.step_minutes} min | lookback_steps={prep.lookback_steps} | "
        f"horizon_steps={prep.horizon_steps} (ahead={ahead_hours:.1f} h)"
    )
    print(
        f"   points={len(prep.df)} | split@{split_ts} | "
        f"train_samples={len(prep.train_end_indices)} | val_samples={len(prep.val_end_indices)}"
    )

    # Model definition
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

    # Validation metrics
    y_val_pred = model.predict(X_val, batch_size=cfg.batch_size, verbose=0)
    mae = float(np.mean(np.abs(y_val - y_val_pred)))
    mse = float(np.mean(np.square(y_val - y_val_pred)))
    r2 = float(r2_score_np(y_val, y_val_pred))

    # Training curve
    plt.figure(figsize=(8, 4))
    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("[LSTM] Training curve (VSTF 5-min)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_curve = os.path.join(cfg.figures_dir, model_name + "_training_curve.png")
    plt.tight_layout()
    plt.savefig(fig_curve, dpi=150)
    plt.close()

    # Next-interval forecast over ~24 h
    fig_next_interval = plot_next_interval_over_24h(
        df=prep.df,
        val_end_indices=prep.val_end_indices,
        y_true_seq=y_val,
        y_pred_seq=y_val_pred,
        horizon_steps=prep.horizon_steps,
        step_minutes=prep.step_minutes,
        figures_dir=cfg.figures_dir,
        model_name=model_name,
        backend_tag="lstm",
        plot_hours=cfg.plot_hours,
    )

    # Save model without optimizer
    model.save(ckpt_path, include_optimizer=False)

    # Save config and metrics
    run_cfg = dict(
        csv_path=cfg.csv_path,
        backend="lstm",
        horizon_minutes=cfg.horizon_minutes,
        lookback_hours=cfg.lookback_hours,
        train_ratio=cfg.train_ratio,
        step_minutes=prep.step_minutes,
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
            "next_interval_png": fig_next_interval,
        },
    }
    with open(os.path.join(cfg.models_dir, model_name + "_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Latest forecast for the last available window
    i_last = len(prep.df) - 1 - prep.horizon_steps
    X_last, _ = build_end_aligned_windows(
        prep.X_all, prep.y_all, prep.lookback_steps, prep.horizon_steps, [i_last]
    )
    y_last_pred = model.predict(X_last, verbose=0)[0]
    horizon_idx = prep.df.index[i_last + 1 : i_last + 1 + prep.horizon_steps]
    df_out = pd.DataFrame({"timestamp": horizon_idx, "y_pred_lstm": y_last_pred})
    out_csv = os.path.join(cfg.pred_dir, f"{model_name}_latest_forecast.csv")
    df_out.to_csv(out_csv, index=False)

    print(f"[LSTM] Saved checkpoint: {ckpt_path}")
    print(
        f"[LSTM] Validation: MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f} "
        f"(next-interval plot: {os.path.basename(fig_next_interval)})"
    )
    print(f"[LSTM] Latest forecast CSV: {out_csv}")

    return {"MAE": mae, "MSE": mse, "R2": r2, "ckpt_path": ckpt_path}


def run_xgb_direct_branch(
    prep: PreparedData, cfg: Config, model_name: str
) -> Dict[str, Any]:
    """
    Train one XGBoost model per horizon step (direct multi-step strategy)
    on 5-min VSTF data.
    """
    if not XGB_AVAILABLE:
        raise ImportError("xgboost not available. Install with: pip install xgboost")

    # Lag features (from the PV target) for train/val
    X_lag_tr = make_lag_matrix_from_series(
        prep.y_all, prep.lookback_steps, prep.train_end_indices
    )
    X_lag_va = make_lag_matrix_from_series(
        prep.y_all, prep.lookback_steps, prep.val_end_indices
    )

    # Time features at target timestamps for each horizon
    feats_tr_per_h = time_feats_for_targets(
        prep.df.index, prep.train_end_indices, prep.horizon_steps
    )
    feats_va_per_h = time_feats_for_targets(
        prep.df.index, prep.val_end_indices, prep.horizon_steps
    )

    # Targets (ground truth) for full validation horizons
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

    split_ts = prep.df.index[prep.boundary_idx]
    ahead_hours = prep.horizon_steps * prep.step_minutes / 60.0
    print(
        f">> Backend XGB | step={prep.step_minutes} min | lookback_steps={prep.lookback_steps} | "
        f"horizon_steps={prep.horizon_steps} (ahead={ahead_hours:.1f} h)"
    )
    print(
        f"   points={len(prep.df)} | split@{split_ts} | "
        f"train_samples={len(prep.train_end_indices)} | val_samples={len(prep.val_end_indices)}"
    )
    print(f"   Training {prep.horizon_steps} direct models...")

    progress_every = max(1, prep.horizon_steps // 6)

    for h in range(1, prep.horizon_steps + 1):
        # Assemble features for horizon h
        X_tr_h = np.hstack([X_lag_tr, feats_tr_per_h[h - 1]])
        X_va_h = np.hstack([X_lag_va, feats_va_per_h[h - 1]])

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

        # Validation predictions for step h
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

    # Plot MAE per horizon step
    plt.figure(figsize=(9, 4))
    plt.plot(np.arange(1, prep.horizon_steps + 1), step_mae, marker="o")
    plt.xlabel("Horizon step (5-min)")
    plt.ylabel("MAE")
    plt.title("[XGB] Validation MAE per step (VSTF 5-min)")
    plt.grid(True, alpha=0.3)
    fig_steps = os.path.join(cfg.figures_dir, model_name + "_val_mae_per_step.png")
    plt.tight_layout()
    plt.savefig(fig_steps, dpi=150)
    plt.close()

    # Next-interval forecast over ~24 h
    fig_next_interval = plot_next_interval_over_24h(
        df=prep.df,
        val_end_indices=prep.val_end_indices,
        y_true_seq=Y_val_full,
        y_pred_seq=val_preds,
        horizon_steps=prep.horizon_steps,
        step_minutes=prep.step_minutes,
        figures_dir=cfg.figures_dir,
        model_name=model_name,
        backend_tag="xgb",
        plot_hours=cfg.plot_hours,
    )

    # Persist models
    if JOBLIB_AVAILABLE:
        joblib.dump(models_h, ckpt_path)
    else:
        import pickle

        with open(ckpt_path, "wb") as f:
            pickle.dump(models_h, f)

    # Save config and metrics
    run_cfg = dict(
        csv_path=cfg.csv_path,
        backend="xgb_direct",
        horizon_minutes=cfg.horizon_minutes,
        lookback_hours=cfg.lookback_hours,
        train_ratio=cfg.train_ratio,
        step_minutes=prep.step_minutes,
        lookback_steps=prep.lookback_steps,
        horizon_steps=prep.horizon_steps,
        lag_feats=prep.lookback_steps,
        time_feats_per_h=4,
        seed=cfg.seed,
        model_name=model_name,
        xgb_params=xgb_params,
        split_sizes=dict(
            n_points=int(len(prep.df)),
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
            "next_interval_png": fig_next_interval,
        },
    }
    with open(os.path.join(cfg.models_dir, model_name + "_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Latest H-step forecast for the last available window
    i_last = len(prep.df) - 1 - prep.horizon_steps
    X_lag_last = make_lag_matrix_from_series(prep.y_all, prep.lookback_steps, [i_last])
    feats_last_per_h = time_feats_for_targets(
        prep.df.index, [i_last], prep.horizon_steps
    )
    preds_last = []
    for h in range(1, prep.horizon_steps + 1):
        X_last_h = np.hstack([X_lag_last, feats_last_per_h[h - 1]])
        preds_last.append(models_h[h - 1].predict(X_last_h)[0])
    preds_last = np.array(preds_last, dtype=np.float32)

    horizon_idx = prep.df.index[i_last + 1 : i_last + 1 + prep.horizon_steps]
    df_out = pd.DataFrame({"timestamp": horizon_idx, "y_pred_xgb": preds_last})
    out_csv = os.path.join(cfg.pred_dir, f"{model_name}_latest_forecast.csv")
    df_out.to_csv(out_csv, index=False)

    print(f"[XGB] Saved models: {ckpt_path}")
    print(
        f"[XGB] Validation: MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f} "
        f"(next-interval plot: {os.path.basename(fig_next_interval)})"
    )
    print(f"[XGB] Latest forecast CSV: {out_csv}")

    return {"MAE": mae, "MSE": mse, "R2": r2, "ckpt_path": ckpt_path}


# -------------------------
# CLI helpers
# -------------------------
def build_model_name(cfg: Config, step_minutes: int) -> str:
    backend_tag = "lstm" if cfg.backend == "lstm" else "xgb"
    return f"{cfg.model_prefix}_{backend_tag}_{cfg.horizon_minutes}minH_{cfg.lookback_hours}lb_{step_minutes}step"


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="VSTF PV next-interval forecast (5-min data, LSTM or XGBoost)"
    )
    parser.add_argument("--backend", choices=["lstm", "xgb_direct"], default=Config.backend)
    parser.add_argument("--csv-path", default=Config.csv_path)
    parser.add_argument("--horizon-minutes", type=int, default=Config.horizon_minutes)
    parser.add_argument("--lookback-hours", type=float, default=Config.lookback_hours)
    parser.add_argument("--train-ratio", type=float, default=Config.train_ratio)
    parser.add_argument("--model-prefix", default=Config.model_prefix)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--plot-hours", type=float, default=Config.plot_hours)

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
        backend=args.backend,
        horizon_minutes=args.horizon_minutes,
        lookback_hours=args.lookback_hours,
        train_ratio=args.train_ratio,
        model_prefix=args.model_prefix,
        lstm_units_1=args.lstm_units_1,
        lstm_units_2=args.lstm_units_2,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
        plot_hours=args.plot_hours,
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
    model_name = build_model_name(cfg, prep.step_minutes)

    print(
        f">> VSTF | step={prep.step_minutes} min | lookback_steps={prep.lookback_steps} | "
        f"horizon_steps={prep.horizon_steps} (ahead={prep.horizon_steps * prep.step_minutes / 60.0:.1f} h)"
    )
    print(
        f"        points={len(prep.df)} | split@{prep.df.index[prep.boundary_idx]} | "
        f"train={len(prep.train_end_indices)} | val={len(prep.val_end_indices)}"
    )
    print(f"        backend={cfg.backend}")

    if cfg.backend == "lstm":
        metrics = run_lstm_branch(prep, cfg, model_name)
    elif cfg.backend == "xgb_direct":
        metrics = run_xgb_direct_branch(prep, cfg, model_name)
    else:
        raise ValueError("Unknown backend. Use 'lstm' or 'xgb_direct'.")

    print(f"Best checkpoint: {metrics['ckpt_path']}")
    print(f"Validation metrics: {metrics}")


if __name__ == "__main__":
    main()
