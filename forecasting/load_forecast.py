# -*- coding: utf-8 -*-
"""
LOAD forecasting API (5-min resolution).
---------------------------------------

Function
--------
load(dt, intervals=MAX_INTERVALS, model_vstf=None, model_st=None,
     vstf_cfg=None, st_cfg=None) -> np.ndarray

- Returns a 1D numpy array of length `intervals` (5-min steps).
- Combines: VSTF (5-min) for the first K steps + ST (hourly -> 5-min interpolation).
- Models are provided by the caller (no model loading here).
- CSV I/O is cached at module level to avoid repeated disk reads.

Expected data
-------------
- CSV 'data/load_5min_test.csv' with columns: ['timestamp', 'p_norm'].

Config dicts (if provided)
--------------------------
- vstf_cfg: {"step_minutes": 5, "lookback_steps": int, "horizon_steps": int}
- st_cfg:   {"step_minutes": 60, "lookback_steps": int, "horizon_steps": int}

Author: you
"""
import os
from typing import Optional, Dict
import numpy as np
import pandas as pd

# --------------------
# Module configuration
# --------------------
LOAD_5MIN_CSV = os.getenv("LOAD_5MIN_CSV", "data/load_5min_test.csv")
TRAIN_RESAMPLE_AGG = os.getenv("TRAIN_RESAMPLE_AGG", "mean")  # mean | median | sum
MAX_INTERVALS = int(os.getenv("FORECAST_MAX_INTERVALS", "432"))  # default: 36h @ 5-min → 180 intervals

# Fallback configs (used only if caller does not pass configs)
LOAD_VSTF_DEFAULT = {"step_minutes": 5, "lookback_steps": 96, "horizon_steps": 48}
LOAD_ST_DEFAULT   = {"step_minutes": 60, "lookback_steps": 8,  "horizon_steps": 36}

# --------------------
# Lazy, in-memory data
# --------------------
_DF5: Optional[pd.DataFrame] = None
_DFH: Optional[pd.DataFrame] = None

# --------------------
# Helpers
# --------------------
def _add_cyc_5min(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    dow = idx.dayofweek.values.astype(float)
    tod_min = (idx.hour * 60 + idx.minute).values.astype(float)
    out = df.copy()
    out["dow_sin"] = np.sin(2*np.pi*dow/7.0)
    out["dow_cos"] = np.cos(2*np.pi*dow/7.0)
    out["tod_sin"] = np.sin(2*np.pi*tod_min/(24.0*60.0))
    out["tod_cos"] = np.cos(2*np.pi*tod_min/(24.0*60.0))
    return out

def _add_cyc_hourly(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    dow = idx.dayofweek.values.astype(float)
    hod = idx.hour.values.astype(float)
    out = df.copy()
    out["dow_sin"] = np.sin(2*np.pi*dow/7.0)
    out["dow_cos"] = np.cos(2*np.pi*dow/7.0)
    out["hod_sin"] = np.sin(2*np.pi*hod/24.0)
    out["hod_cos"] = np.cos(2*np.pi*hod/24.0)
    return out

def _ensure_dataframes():
    """Load and cache 5-min and hourly feature frames."""
    global _DF5, _DFH
    if _DF5 is not None and _DFH is not None:
        return
    df = pd.read_csv(LOAD_5MIN_CSV, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    if "p_norm" not in df.columns:
        raise ValueError("LOAD CSV must contain column 'p_norm'.")
    _DF5 = _add_cyc_5min(df[["p_norm"]].copy())

    if TRAIN_RESAMPLE_AGG == "mean":
        dfH = df[["p_norm"]].resample("h").mean(numeric_only=True).dropna()
    elif TRAIN_RESAMPLE_AGG == "median":
        dfH = df[["p_norm"]].resample("h").median(numeric_only=True).dropna()
    elif TRAIN_RESAMPLE_AGG == "sum":
        dfH = df[["p_norm"]].resample("h").sum(numeric_only=True).dropna()
    else:
        raise ValueError("Unsupported TRAIN_RESAMPLE_AGG.")
    _DFH = _add_cyc_hourly(dfH)

def _build_vstf_input(df5: pd.DataFrame, t_now: pd.Timestamp, lookback_steps: int) -> np.ndarray:
    """End-aligned window for VSTF input."""
    req = ["p_norm", "dow_sin", "dow_cos", "tod_sin", "tod_cos"]
    idx_window = pd.date_range(end=t_now, periods=lookback_steps, freq="5min")
    Xw = df5.reindex(idx_window)
    if Xw[req].isna().any().any():
        raise ValueError("Insufficient 5-min history for VSTF window.")
    return Xw[req].values.astype(np.float32)[np.newaxis, :, :]

def _build_st_input(dfH: pd.DataFrame, anchor_hour: pd.Timestamp, lookback_steps: int) -> np.ndarray:
    """End-aligned window for ST hourly input (anchor = next_full_hour - 1h)."""
    req = ["p_norm", "dow_sin", "dow_cos", "hod_sin", "hod_cos"]
    anchor_hour = pd.Timestamp(anchor_hour).floor("h")
    idx_window = pd.date_range(end=anchor_hour, periods=lookback_steps, freq="h")
    Xw = dfH.reindex(idx_window)
    if Xw[req].isna().any().any():
        raise ValueError("Insufficient hourly history for ST window.")
    return Xw[req].values.astype(np.float32)[np.newaxis, :, :]

def _interp_hourly_to_5min(times_5: pd.DatetimeIndex, st_hourly: pd.Series) -> pd.Series:
    """Time interpolation to 5-min grid with edge fill."""
    union_idx = st_hourly.index.union(times_5)
    s = st_hourly.reindex(union_idx).interpolate(method="time").reindex(times_5)
    if s.isna().any():
        s = s.bfill().ffill()
    return s

# --------------------
# Public API
# --------------------
def load(
    dt,
    intervals: Optional[int] = None,
    *,
    model_vstf,
    model_st,
    vstf_cfg: Optional[Dict] = None,
    st_cfg: Optional[Dict] = None,
) -> np.ndarray:
    """
    5-min load forecast.

    Parameters
    ----------
    dt : str | pandas.Timestamp | datetime
        Current time (t_now), e.g., '2009-04-30 12:00:00'.
    intervals : int, optional
        Number of 5-min steps to predict (max MAX_INTERVALS). Default MAX_INTERVALS.
    model_vstf : keras.Model
        Preloaded VSTF model (5-min).
    model_st : keras.Model
        Preloaded ST model (hourly).
    vstf_cfg : dict, optional
        Model config dict for VSTF (step_minutes, lookback_steps, horizon_steps).
    st_cfg : dict, optional
        Model config dict for ST (step_minutes, lookback_steps, horizon_steps).

    Returns
    -------
    np.ndarray of shape (intervals,)
    """
    if model_vstf is None or model_st is None:
        raise ValueError("Both model_vstf and model_st must be provided.")

    cfg_v = (vstf_cfg or LOAD_VSTF_DEFAULT).copy()
    cfg_h = (st_cfg or LOAD_ST_DEFAULT).copy()
    if cfg_v["step_minutes"] != 5:
        raise ValueError("VSTF step_minutes must be 5.")
    if cfg_h["step_minutes"] != 60:
        raise ValueError("ST step_minutes must be 60.")

    _ensure_dataframes()

    t_now = pd.Timestamp(dt)
    if intervals is None:
        intervals = MAX_INTERVALS
    intervals = int(intervals)
    if intervals <= 0:
        raise ValueError("intervals must be > 0.")
    if intervals > MAX_INTERVALS:
        intervals = MAX_INTERVALS

    # VSTF (5-min)
    X_v = _build_vstf_input(_DF5, t_now, cfg_v["lookback_steps"])
    v_pred = model_vstf.predict(X_v, verbose=0).reshape(-1)
    use_vstf_steps = int(min(cfg_v["horizon_steps"], len(v_pred), intervals))

    # ST (hourly) anchored at next full hour
    next_full_hour = (t_now + pd.Timedelta(minutes=5)).ceil("h")
    anchor_hour = next_full_hour - pd.Timedelta(hours=1)
    X_h = _build_st_input(_DFH, anchor_hour, cfg_h["lookback_steps"])
    st_pred = model_st.predict(X_h, verbose=0).reshape(-1)

    # Determine hours needed to cover `intervals`
    hours_needed = int(np.ceil(intervals / 12.0))
    st_pred = st_pred[:max(hours_needed, 1)]
    st_times = pd.date_range(start=next_full_hour, periods=len(st_pred), freq="h")
    st_hourly = pd.Series(st_pred, index=st_times)

    # Interpolate to 5-min grid and combine
    times_5 = pd.date_range(start=t_now + pd.Timedelta(minutes=5), periods=intervals, freq="5min")
    st_5 = _interp_hourly_to_5min(times_5, st_hourly).values.astype(np.float32)

    out = st_5.copy()
    if use_vstf_steps > 0:
        out[:use_vstf_steps] = v_pred[:use_vstf_steps].astype(np.float32)
    return out
