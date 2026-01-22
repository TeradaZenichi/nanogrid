# -*- coding: utf-8 -*-
"""
Unified LOAD & PV real-time simulation combining very-short-term (5-min) and
short-term hourly forecasts over a 36h horizon.

Behavior
--------
- Output grid: every 5 minutes for 36 hours (432 points).
- Default: VSTF (5-min) covers its full horizon; remaining horizon is filled
  with ST hourly forecasts interpolated to 5-min.
- Optional legacy mode: use VSTF only until the next full hour.
- PV: p_norm is forced to zero outside the daylight window.
- Debug overlays keep recent queries for quick visual checks.
- Simulation window is flexible so runs can span multiple queries.
"""

import os
import json
import time
import pickle
from collections import deque
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model

try:
    import joblib
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

try:
    import xgboost  # noqa: F401
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# =========================
# User config
# =========================
LOAD_TEST_CSV = "data/load_5min_test.csv"
PV_TEST_CSV   = "data/pv_5min_test.csv"

DAY_START = "06:00"
DAY_END   = "19:00"

TRAIN_RESAMPLE_AGG = "mean"  # mean | median | sum

TOTAL_HOURS = 36
STEP_MIN    = 5
H_STEPS     = TOTAL_HOURS * 12  # 432

# Defaults per domain (used if config JSON is missing)
LOAD_VSTF_DEFAULT = {"step_minutes": 5,  "lookback_steps": 96, "horizon_steps": 48}
LOAD_ST_DEFAULT   = {"step_minutes": 60, "lookback_steps": 8,  "horizon_steps": 36}
PV_VSTF_DEFAULT   = {"step_minutes": 5,  "lookback_steps": 60, "horizon_steps": 48}
PV_ST_DEFAULT     = {"step_minutes": 60, "lookback_steps": 16, "horizon_steps": 36}

# Candidate model paths (XGBoost first, LSTM fallback)
LOAD_VSTF_CANDIDATES = [
    "models/xgb_vstf_load_5min_60min_48.joblib",
    "models/xgb_vstf_load_5min_60min_48.pkl",
    "models/lstm_vstf_load_5min_60min_48.keras",
    "models/llstm_vstf_load_5min_60min_48.keras",
    "models/lstm_vstf_load_5min_240min_96.keras",
]

LOAD_ST_CANDIDATES = [
    "models/xgb_hourly_load_36h_8.joblib",
    "models/xgb_hourly_load_36h_8.pkl",
    "models/lstm_hourly_load_36h_8.keras",
]

PV_VSTF_CANDIDATES = [
    "models/xgb_vstf_pv_5min_60min_48.joblib",
    "models/xgb_vstf_pv_5min_60min_48.pkl",
    "models/lstm_vstf_pv_5min_60min_48.keras",
    "models/llstm_vstf_pv_5min_60min_48.keras",
]

PV_ST_CANDIDATES = [
    "models/xgb_hourly_pv_36h_16.joblib",
    "models/xgb_hourly_pv_36h_16.pkl",
    "models/lstm_hourly_pv_36h_8.keras",
]

INITIAL_QUERY_TIME   = "2009-04-30 12:00:00"
AUTO_START_MODE      = "earliest"   # "latest" | "latest_minus_hours" | "earliest"
AUTO_START_BACK_HOURS= 72
STOP_AFTER_DAYS      = 10
MAX_QUERIES          = 200

SIM_ADVANCE_STEPS = 1
DEBUG_ENABLED     = True
DEBUG_PLOT_EVERY  = 7
DEBUG_SHOW_LAST_K = 3
DEBUG_MAX_PLOTS   = 48

USE_VSTF_FULL_HORIZON = True

SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# Daylight & PV utilities
# =========================
def _hhmm_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)

def _daylight_mask(index: pd.DatetimeIndex, start_hhmm: str, end_hhmm: str) -> np.ndarray:
    start = _hhmm_to_minutes(start_hhmm)
    end   = _hhmm_to_minutes(end_hhmm)
    minutes = (index.hour * 60 + index.minute).astype(int)
    if start <= end:
        return (minutes >= start) & (minutes < end)
    else:
        return (minutes >= start) | (minutes < end)

def enforce_pv_zero_outside_window(df: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    if "p_norm" not in df.columns:
        raise ValueError("Input DataFrame must contain 'p_norm'.")
    out = df.copy()
    mask = _daylight_mask(out.index, start_hhmm, end_hhmm)
    out.loc[~mask, "p_norm"] = 0.0
    return out

def clamp_series_to_daylight(s: pd.Series, start_hhmm: str, end_hhmm: str) -> pd.Series:
    s2 = s.copy()
    mask = _daylight_mask(s2.index, start_hhmm, end_hhmm)
    s2.loc[~mask] = 0.0
    return s2

# =========================
# Feature engineering
# =========================
def add_cyclical_features_5min(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    dow = idx.dayofweek.values.astype(float)
    dow_ang = 2.0 * np.pi * dow / 7.0
    df["dow_sin"] = np.sin(dow_ang)
    df["dow_cos"] = np.cos(dow_ang)
    tod_min = (idx.hour * 60 + idx.minute).values.astype(float)
    tod_ang = 2.0 * np.pi * tod_min / (24.0 * 60.0)
    df["tod_sin"] = np.sin(tod_ang)
    df["tod_cos"] = np.cos(tod_ang)
    return df

def add_cyclical_features_hourly(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    dow = idx.dayofweek.values.astype(float)
    dow_ang = 2.0 * np.pi * dow / 7.0
    df["dow_sin"] = np.sin(dow_ang)
    df["dow_cos"] = np.cos(dow_ang)
    hod = idx.hour.values.astype(float)
    hod_ang = 2.0 * np.pi * hod / 24.0
    df["hod_sin"] = np.sin(hod_ang)
    df["hod_cos"] = np.cos(hod_ang)
    return df

# =========================
# Model loading helpers
# =========================
def _try_load_config(model_path: str) -> Optional[Dict]:
    """Load companion _config.json next to keras/joblib/pkl artifacts."""
    if model_path.endswith(".keras"):
        base = model_path[: -len(".keras")]
    elif model_path.endswith(".joblib"):
        base = model_path[: -len(".joblib")]
    elif model_path.endswith(".pkl"):
        base = model_path[: -len(".pkl")]
    else:
        base = model_path
    cfg_path = base + "_config.json"
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            meta = json.load(f)
        return meta.get("config", meta)
    return None


def get_model_cfg(path: str, fallback: Dict) -> Dict:
    cfg = _try_load_config(path) or fallback.copy()
    return {
        "step_minutes": int(cfg.get("step_minutes", fallback.get("step_minutes", 5))),
        "lookback_steps": int(cfg.get("lookback_steps", fallback.get("lookback_steps", 1))),
        "horizon_steps": int(cfg.get("horizon_steps", fallback.get("horizon_steps", 1))),
        "backend": cfg.get("backend"),
    }


def load_keras_safely(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    try:
        return load_model(path, compile=False, safe_mode=False)
    except TypeError:
        return load_model(path, compile=False)


def load_xgb_models(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    if JOBLIB_AVAILABLE:
        return joblib.load(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_existing_path(cands: List[str]) -> str:
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("None of the candidate model paths exist:\n" + "\n".join(cands))


def load_model_bundle(path: str, fallback: Dict) -> Dict:
    cfg = get_model_cfg(path, fallback)
    inferred_backend = cfg.get("backend")
    if inferred_backend not in ("lstm", "xgb_direct"):
        inferred_backend = "xgb_direct" if path.endswith((".joblib", ".pkl")) else "lstm"

    if inferred_backend == "xgb_direct":
        model = load_xgb_models(path)
    else:
        model = load_keras_safely(path)

    return {
        "backend": inferred_backend,
        "model": model,
        "cfg": {
            "step_minutes": cfg["step_minutes"],
            "lookback_steps": cfg["lookback_steps"],
            "horizon_steps": cfg["horizon_steps"],
        },
    }

# =========================
# Inputs (end-aligned)
# =========================
def build_vstf_input(df5: pd.DataFrame, t_now: pd.Timestamp, lookback_steps: int) -> np.ndarray:
    req = ["p_norm","dow_sin","dow_cos","tod_sin","tod_cos"]
    if not all(c in df5.columns for c in req):
        raise ValueError("Missing required 5-min features.")
    idx_window = pd.date_range(end=t_now, periods=lookback_steps, freq="5min")
    Xw = df5.reindex(idx_window)
    if Xw["p_norm"].isna().any():
        raise ValueError("Historial 5-min insuficiente para la ventana VSTF.")
    return Xw[req].values.astype(np.float32)[np.newaxis, :, :]

def build_st_input_for_anchor(dfH: pd.DataFrame, anchor_hour: pd.Timestamp, lookback_steps: int) -> np.ndarray:
    req = ["p_norm","dow_sin","dow_cos","hod_sin","hod_cos"]
    if not all(c in dfH.columns for c in req):
        raise ValueError("Missing required hourly features.")
    anchor_hour = pd.Timestamp(anchor_hour).floor("h")
    idx_window = pd.date_range(end=anchor_hour, periods=lookback_steps, freq="h")
    Xw = dfH.reindex(idx_window)
    if Xw["p_norm"].isna().any():
        raise ValueError("Not enough hourly history for the ST window.")
    return Xw[req].values.astype(np.float32)[np.newaxis, :, :]


# =========================
# XGBoost inference helpers
# =========================
def _lags_from_series(series: pd.Series, end_time: pd.Timestamp, step_minutes: int, lookback_steps: int) -> np.ndarray:
    idx_window = pd.date_range(end=end_time, periods=lookback_steps, freq=f"{step_minutes}min")
    window = series.reindex(idx_window)
    if window.isna().any():
        raise ValueError("Insufficient history for requested lookback window.")
    return window.values.astype(np.float32)


def predict_xgb_vstf(models_h: List, df5: pd.DataFrame, t_now: pd.Timestamp, cfg: Dict) -> np.ndarray:
    if not XGB_AVAILABLE:
        raise ImportError("xgboost is required to run XGB models.")
    step_minutes = int(cfg["step_minutes"])
    lags = _lags_from_series(df5["p_norm"], t_now, step_minutes, cfg["lookback_steps"])

    preds = []
    for h, model_h in enumerate(models_h, start=1):
        tgt_time = t_now + pd.Timedelta(minutes=step_minutes * h)
        dow = float(tgt_time.dayofweek)
        tod_min = float(tgt_time.hour * 60 + tgt_time.minute)
        dow_ang = 2.0 * np.pi * dow / 7.0
        tod_ang = 2.0 * np.pi * tod_min / (24.0 * 60.0)
        feats = np.hstack([
            lags,
            [np.sin(dow_ang), np.cos(dow_ang), np.sin(tod_ang), np.cos(tod_ang)],
        ]).reshape(1, -1)
        preds.append(model_h.predict(feats)[0])
        if len(preds) >= cfg["horizon_steps"]:
            break
    return np.array(preds, dtype=np.float32)


def predict_xgb_hourly(models_h: List, dfH: pd.DataFrame, anchor_hour: pd.Timestamp, cfg: Dict) -> np.ndarray:
    if not XGB_AVAILABLE:
        raise ImportError("xgboost is required to run XGB models.")
    lags = _lags_from_series(dfH["p_norm"], anchor_hour, 60, cfg["lookback_steps"])

    preds = []
    for h, model_h in enumerate(models_h, start=1):
        tgt_time = pd.Timestamp(anchor_hour).floor("h") + pd.Timedelta(hours=h)
        dow = float(tgt_time.dayofweek)
        hod = float(tgt_time.hour)
        dow_ang = 2.0 * np.pi * dow / 7.0
        hod_ang = 2.0 * np.pi * hod / 24.0
        feats = np.hstack([
            lags,
            [np.sin(dow_ang), np.cos(dow_ang), np.sin(hod_ang), np.cos(hod_ang)],
        ]).reshape(1, -1)
        preds.append(model_h.predict(feats)[0])
        if len(preds) >= cfg["horizon_steps"]:
            break
    return np.array(preds, dtype=np.float32)


def predict_vstf(bundle: Dict, df5: pd.DataFrame, t_now: pd.Timestamp) -> np.ndarray:
    cfg = bundle["cfg"]
    if bundle["backend"] == "xgb_direct":
        return predict_xgb_vstf(bundle["model"], df5, t_now, cfg)
    X_vstf = build_vstf_input(df5, t_now, lookback_steps=cfg["lookback_steps"])
    return bundle["model"].predict(X_vstf, verbose=0).reshape(-1)


def predict_hourly(bundle: Dict, dfH: pd.DataFrame, anchor_hour: pd.Timestamp) -> np.ndarray:
    cfg = bundle["cfg"]
    if bundle["backend"] == "xgb_direct":
        return predict_xgb_hourly(bundle["model"], dfH, anchor_hour, cfg)
    X_st = build_st_input_for_anchor(dfH, anchor_hour, lookback_steps=cfg["lookback_steps"])
    return bundle["model"].predict(X_st, verbose=0).reshape(-1)

# =========================
# Interpolación & combinación
# =========================
def interpolate_hourly_to_5min_full(times_5: pd.DatetimeIndex, st_hourly: pd.Series) -> pd.Series:
    """
    Interpola la serie horaria ST a 5-min y rellena bordes para cobertura completa:
      - Cabeza: backfill con el primer valor horario.
      - Cola: forward fill con el último valor horario.
    """
    union_idx = st_hourly.index.union(times_5)
    s = st_hourly.reindex(union_idx).interpolate(method="time")
    s = s.reindex(times_5)
    if s.isna().any():
        # Arreglo de FutureWarning: usar .bfill().ffill() en lugar de fillna(method=...)
        s = s.bfill().ffill()
    return s

def steps_until_next_full_hour(t_now: pd.Timestamp) -> int:
    start_5 = t_now + pd.Timedelta(minutes=5)
    next_full = start_5.ceil("h")
    if start_5 >= next_full:
        return 0
    delta_min = int((next_full - start_5).total_seconds() // 60)
    return delta_min // 5  # 0..11

def forecast_combined_36h_5min(
    t_now: pd.Timestamp,
    df5: pd.DataFrame,
    dfH: pd.DataFrame,
    model_vstf: Dict,
    model_st: Dict,
    vstf_cfg: Dict,
    st_cfg: Dict,
    clamp_daylight: Optional[Tuple[str, str]] = None,
    st_cache: Optional[Dict[pd.Timestamp, pd.Series]] = None,
    use_vstf_full_horizon: bool = True,
    total_hours: int = TOTAL_HOURS,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Devuelve: combined_5min, vstf_5min_usado, st_hourly_points, st_only_5min
    """
    assert vstf_cfg["step_minutes"] == 5
    assert st_cfg["step_minutes"] == 60
    if st_cfg["horizon_steps"] < total_hours:
        raise ValueError("Hourly model horizon is shorter than requested total_hours.")

    # VSTF (5-min)
    vstf_pred = predict_vstf(model_vstf, df5, t_now)

    # Cuántos pasos VSTF usar
    if use_vstf_full_horizon:
        use_vstf_steps = min(vstf_cfg["horizon_steps"], total_hours * 12)
    else:
        use_vstf_steps = steps_until_next_full_hour(t_now)

    # ST hourly (cache by next_full_hour)
    next_full_hour = (t_now + pd.Timedelta(minutes=5)).ceil("h")
    if st_cache is not None and next_full_hour in st_cache:
        st_hourly = st_cache[next_full_hour]
    else:
        anchor_hour = next_full_hour - pd.Timedelta(hours=1)
        st_pred = predict_hourly(model_st, dfH, anchor_hour)[: st_cfg["horizon_steps"]]
        st_times_hourly = pd.date_range(start=next_full_hour, periods=st_cfg["horizon_steps"], freq="h")
        st_hourly = pd.Series(st_pred, index=st_times_hourly)
        if clamp_daylight is not None:
            st_hourly = clamp_series_to_daylight(st_hourly, *clamp_daylight)
        if st_cache is not None:
            st_cache[next_full_hour] = st_hourly

    # Rejilla 5-min
    times_5 = pd.date_range(start=t_now + pd.Timedelta(minutes=5), periods=total_hours * 12, freq="5min")

    # ST → 5-min
    st_only_5min = interpolate_hourly_to_5min_full(times_5, st_hourly)
    if clamp_daylight is not None:
        st_only_5min = clamp_series_to_daylight(st_only_5min, *clamp_daylight)

    # VSTF (5-min) para los primeros use_vstf_steps
    vstf_5min = pd.Series(vstf_pred[:use_vstf_steps], index=times_5[:use_vstf_steps])
    if clamp_daylight is not None and len(vstf_5min) > 0:
        vstf_5min = clamp_series_to_daylight(vstf_5min, *clamp_daylight)

    # Combinar
    combined = st_only_5min.copy()
    if use_vstf_steps > 0:
        combined.iloc[:use_vstf_steps] = vstf_5min.values

    # Clamp final PV
    if clamp_daylight is not None:
        combined = clamp_series_to_daylight(combined, *clamp_daylight)

    assert len(combined) == H_STEPS, f"Combined length {len(combined)} != {H_STEPS}"
    assert len(st_only_5min) == H_STEPS, f"ST-5min length {len(st_only_5min)} != {H_STEPS}"

    return combined, vstf_5min, st_hourly, st_only_5min

# =========================
# Debug plot (overlay)
# =========================
def plot_overlay_both(samples: List[Dict], save_dir: str, fig_idx: int):
    assert len(samples) > 0
    K = len(samples)
    plt.figure(figsize=(14.5, 6.5))

    s = samples[-1]
    t_now = s["t_now"]
    last_day_idx = pd.date_range(end=t_now, periods=24*12, freq="5min")
    plt.plot(last_day_idx, s["dfl5"]["p_norm"].reindex(last_day_idx),
             label="LOAD last 24h (5-min)", linewidth=1.0, alpha=0.7)
    plt.plot(last_day_idx, s["dfp5"]["p_norm"].reindex(last_day_idx),
             label="PV last 24h (5-min)", linewidth=1.0, alpha=0.7)

    for j, ss in enumerate(samples):
        alpha = 0.35 + 0.65 * (j + 1) / K
        lw_pred = 1.2 if j < K - 1 else 2.0
        marker_size = 2.5 if j < K - 1 else 3.5

        gt_load = ss["dfl5"]["p_norm"].reindex(ss["comb_load"].index).values
        gt_pv   = ss["dfp5"]["p_norm"].reindex(ss["comb_pv"].index).values

        plt.plot(ss["comb_load"].index, gt_load, marker='.', markersize=2.0,
                 linewidth=0.8, alpha=alpha*0.8, label=f"LOAD GT 36h (q{j+1}/{K})" if j==K-1 else None)
        plt.plot(ss["comb_pv"].index,   gt_pv,   marker='.', markersize=2.0,
                 linewidth=0.8, alpha=alpha*0.8, label=f"PV GT 36h (q{j+1}/{K})" if j==K-1 else None)

        plt.plot(ss["comb_load"].index, ss["comb_load"].values,
                 marker='.', markersize=marker_size, linewidth=lw_pred, alpha=alpha,
                 label=f"LOAD Combined (q{j+1}/{K})" if j==K-1 else None)
        plt.plot(ss["comb_pv"].index,   ss["comb_pv"].values,
                 marker='.', markersize=marker_size, linewidth=lw_pred, alpha=alpha,
                 label=f"PV Combined (q{j+1}/{K})" if j==K-1 else None)

    plt.plot(s["st5_load"].index, s["st5_load"].values, linestyle="--", linewidth=1.0, alpha=0.8, label="LOAD ST-only→5min")
    plt.plot(s["st5_pv"].index,   s["st5_pv"].values,   linestyle="--", linewidth=1.0, alpha=0.8, label="PV ST-only→5min")
    if len(s["vstf_load"]) > 0:
        plt.plot(s["vstf_load"].index, s["vstf_load"].values, linewidth=2.2, label="LOAD VSTF (used)")
    if len(s["vstf_pv"]) > 0:
        plt.plot(s["vstf_pv"].index, s["vstf_pv"].values, linewidth=2.2, label="PV VSTF (used)")
    plt.scatter(s["stH_load"].index, s["stH_load"].values, s=24, label="LOAD ST hourly pts", zorder=5)
    plt.scatter(s["stH_pv"].index,   s["stH_pv"].values,   s=24, label="PV ST hourly pts",   zorder=5)

    next_full_hour = (t_now + pd.Timedelta(minutes=5)).ceil("h")
    plt.axvline(t_now, color="k", linestyle="--", alpha=0.6, label="t_now")
    plt.axvline(next_full_hour, color="gray", linestyle="--", alpha=0.5, label="next_full_hour")

    plt.title(f"LOAD & PV — 36h @ 5-min (overlay last {K} queries)  |  {t_now}")
    plt.xlabel("Time")
    plt.ylabel("p_norm")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    out_png = os.path.join(save_dir, f"debug_overlay_{fig_idx:04d}_{t_now.strftime('%Y%m%d_%H%M')}.png")
    plt.savefig(out_png, dpi=140)
    plt.close()

# =========================
# Main simulation
# =========================
def main():
    # ---------- LOAD ----------
    dfl = pd.read_csv(LOAD_TEST_CSV, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    if "p_norm" not in dfl.columns:
        raise ValueError("LOAD CSV must contain 'p_norm'.")
    dfl5 = add_cyclical_features_5min(dfl[["p_norm"]].copy())

    if TRAIN_RESAMPLE_AGG == "mean":
        dflH = dfl[["p_norm"]].resample("h").mean(numeric_only=True).dropna()
    elif TRAIN_RESAMPLE_AGG == "median":
        dflH = dfl[["p_norm"]].resample("h").median(numeric_only=True).dropna()
    elif TRAIN_RESAMPLE_AGG == "sum":
        dflH = dfl[["p_norm"]].resample("h").sum(numeric_only=True).dropna()
    else:
        raise ValueError("TRAIN_RESAMPLE_AGG not supported.")
    dflH = add_cyclical_features_hourly(dflH)

    # ---------- PV ----------
    dfp_raw = pd.read_csv(PV_TEST_CSV, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    if "p_norm" not in dfp_raw.columns:
        raise ValueError("PV CSV must contain 'p_norm'.")
    dfp = enforce_pv_zero_outside_window(dfp_raw[["p_norm"]], DAY_START, DAY_END)
    dfp5 = add_cyclical_features_5min(dfp.copy())

    if TRAIN_RESAMPLE_AGG == "mean":
        dfpH = dfp[["p_norm"]].resample("h").mean(numeric_only=True).dropna()
    elif TRAIN_RESAMPLE_AGG == "median":
        dfpH = dfp[["p_norm"]].resample("h").median(numeric_only=True).dropna()
    elif TRAIN_RESAMPLE_AGG == "sum":
        dfpH = dfp[["p_norm"]].resample("h").sum(numeric_only=True).dropna()
    else:
        raise ValueError("TRAIN_RESAMPLE_AGG not supported.")
    dfpH = add_cyclical_features_hourly(dfpH)

    # ---------- Model paths (prefer XGBoost, fallback to LSTM) ----------
    load_vstf_path = resolve_existing_path(LOAD_VSTF_CANDIDATES)
    load_st_path   = resolve_existing_path(LOAD_ST_CANDIDATES)
    pv_vstf_path   = resolve_existing_path(PV_VSTF_CANDIDATES)
    pv_st_path     = resolve_existing_path(PV_ST_CANDIDATES)

    # ---------- Load models & configs ----------
    m_vstf_load = load_model_bundle(load_vstf_path, LOAD_VSTF_DEFAULT)
    m_st_load   = load_model_bundle(load_st_path,   LOAD_ST_DEFAULT)
    m_vstf_pv   = load_model_bundle(pv_vstf_path,   PV_VSTF_DEFAULT)
    m_st_pv     = load_model_bundle(pv_st_path,     PV_ST_DEFAULT)

    vstf_cfg_load = m_vstf_load["cfg"]
    st_cfg_load   = m_st_load["cfg"]
    vstf_cfg_pv   = m_vstf_pv["cfg"]
    st_cfg_pv     = m_st_pv["cfg"]

    # ---------- Rango de simulación ----------
    last_idx_common   = min(dfl5.index.max(), dfp5.index.max())
    t_stop_by_horizon = last_idx_common - pd.Timedelta(hours=TOTAL_HOURS)  # latest feasible t_now
    t_now_auto_latest = t_stop_by_horizon - pd.Timedelta(minutes=5)        # margen para el primer +5
    max_lb_steps      = max(vstf_cfg_load["lookback_steps"], vstf_cfg_pv["lookback_steps"])
    earliest_ok       = max(dfl5.index.min(), dfp5.index.min()) + pd.Timedelta(minutes=5 * max_lb_steps)

    if INITIAL_QUERY_TIME is not None:
        t_start = pd.Timestamp(INITIAL_QUERY_TIME)
    else:
        if AUTO_START_MODE == "latest":
            t_candidate = t_now_auto_latest
        elif AUTO_START_MODE == "latest_minus_hours":
            t_candidate = t_now_auto_latest - pd.Timedelta(hours=AUTO_START_BACK_HOURS)
        elif AUTO_START_MODE == "earliest":
            t_candidate = earliest_ok
        else:
            t_candidate = t_now_auto_latest
        t_start = min(max(t_candidate, earliest_ok), t_stop_by_horizon)

    t_stop_by_days = t_start + pd.Timedelta(days=STOP_AFTER_DAYS)
    t_stop         = min(t_stop_by_horizon, t_stop_by_days)
    if t_stop <= t_start:
        raise ValueError("Insufficient test span. Adjust start/window parameters.")

    step_delta = pd.Timedelta(minutes=5 * SIM_ADVANCE_STEPS)
    total_minutes = int((t_stop - t_start).total_seconds() // 60)
    est_iters = total_minutes // (5 * SIM_ADVANCE_STEPS) + 1
    print(
        f"Simulating from {t_start} to {t_stop} every {SIM_ADVANCE_STEPS*5} min "
        f"(~{est_iters} iters) [debug={'on' if DEBUG_ENABLED else 'off'}, "
        f"plot_every={DEBUG_PLOT_EVERY}, show_last_k={DEBUG_SHOW_LAST_K}, "
        f"use_vstf_full_horizon={USE_VSTF_FULL_HORIZON}]"
    )

    # ---------- Loop ----------
    st_cache_load: Dict[pd.Timestamp, pd.Series] = {}
    st_cache_pv:   Dict[pd.Timestamp, pd.Series] = {}
    overlay_buf = deque(maxlen=max(1, DEBUG_SHOW_LAST_K))

    saved_example_load = False
    saved_example_pv   = False

    rows = []
    plot_counter = 0
    cur = t_start
    qcount = 0

    while cur <= t_stop and (MAX_QUERIES is None or qcount < MAX_QUERIES):
        try:
            t0 = time.perf_counter()
            comb_load, vstf_load, stH_load, st5_load = forecast_combined_36h_5min(
                t_now=cur, df5=dfl5, dfH=dflH,
                model_vstf=m_vstf_load, model_st=m_st_load,
                vstf_cfg=vstf_cfg_load, st_cfg=st_cfg_load,
                clamp_daylight=None, st_cache=st_cache_load,
                use_vstf_full_horizon=USE_VSTF_FULL_HORIZON,
                total_hours=TOTAL_HOURS
            )
            comb_pv, vstf_pv, stH_pv, st5_pv = forecast_combined_36h_5min(
                t_now=cur, df5=dfp5, dfH=dfpH,
                model_vstf=m_vstf_pv, model_st=m_st_pv,
                vstf_cfg=vstf_cfg_pv, st_cfg=st_cfg_pv,
                clamp_daylight=(DAY_START, DAY_END), st_cache=st_cache_pv,
                use_vstf_full_horizon=USE_VSTF_FULL_HORIZON,
                total_hours=TOTAL_HOURS
            )
            runtime_s = time.perf_counter() - t0
            comb_pv = comb_pv.clip(lower=0.0)
            vstf_pv = vstf_pv.clip(lower=0.0)
            stH_pv = stH_pv.clip(lower=0.0)
            st5_pv  = st5_pv.clip(lower=0.0)

        except Exception as e:
            print(f"[WARN] Query @ {cur} failed: {e}")
            cur += step_delta
            qcount += 1
            continue

        if not saved_example_load:
            pd.DataFrame({
                "timestamp": comb_load.index,
                "pred_combined_load": comb_load.values.astype(np.float32),
                "pred_vstf_used_load": np.r_[vstf_load.values, [np.nan]*(len(comb_load)-len(vstf_load))],
                "pred_st_only_36h_load": st5_load.values.astype(np.float32),
                "gt_load": dfl5["p_norm"].reindex(comb_load.index).values.astype(np.float32),
            }).to_csv(os.path.join(SAVE_DIR, "example_forecast_36h_5min_LOAD.csv"), index=False)
            saved_example_load = True

        if not saved_example_pv:
            pd.DataFrame({
                "timestamp": comb_pv.index,
                "pred_combined_pv": comb_pv.values.astype(np.float32),
                "pred_vstf_used_pv": np.r_[vstf_pv.values, [np.nan]*(len(comb_pv)-len(vstf_pv))],
                "pred_st_only_36h_pv": st5_pv.values.astype(np.float32),
                "gt_pv": dfp5["p_norm"].reindex(comb_pv.index).values.astype(np.float32),
            }).to_csv(os.path.join(SAVE_DIR, "example_forecast_36h_5min_PV.csv"), index=False)
            saved_example_pv = True

        overlay_buf.append({
            "t_now": cur,
            "dfl5": dfl5,
            "dfp5": dfp5,
            "comb_load": comb_load, "comb_pv": comb_pv,
            "vstf_load": vstf_load, "vstf_pv": vstf_pv,
            "st5_load": st5_load,   "st5_pv": st5_pv,
            "stH_load": stH_load,   "stH_pv": stH_pv
        })

        if DEBUG_ENABLED and DEBUG_PLOT_EVERY > 0 and (plot_counter < DEBUG_MAX_PLOTS):
            if (qcount % DEBUG_PLOT_EVERY) == 0:
                plot_overlay_both(list(overlay_buf), SAVE_DIR, plot_counter)
                plot_counter += 1

        rows.append({"query_time": cur, "runtime_s": runtime_s})
        cur += step_delta
        qcount += 1

    pd.DataFrame(rows).to_csv(os.path.join(SAVE_DIR, "realtime_runtime_log.csv"), index=False)
    print("Simulation finished.")

if __name__ == "__main__":
    main()
