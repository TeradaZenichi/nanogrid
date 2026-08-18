"""Five-minute PV forecasting from short- and long-term models."""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .causal import hourly_forecast_start, last_completed_hour

PV_5MIN_CSV = "data/pv_5min_test.csv"
TRAIN_RESAMPLE_AGG = "mean"
MAX_INTERVALS = 432

PV_VSTF_DEFAULT = {"step_minutes": 5, "lookback_steps": 48, "horizon_steps": 12}
PV_ST_DEFAULT = {"step_minutes": 60, "lookback_steps": 8, "horizon_steps": 36}

_DF5: Optional[pd.DataFrame] = None
_DFH: Optional[pd.DataFrame] = None

def _hhmm_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _daylight_mask(index: pd.DatetimeIndex, start_hhmm: str, end_hhmm: str) -> np.ndarray:
    start = _hhmm_to_minutes(start_hhmm)
    end = _hhmm_to_minutes(end_hhmm)
    minutes = (index.hour * 60 + index.minute).astype(int)
    if start <= end:
        return (minutes >= start) & (minutes < end)
    return (minutes >= start) | (minutes < end)


def _enforce_pv_zero_outside_window(df: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    out = df.copy()
    mask = _daylight_mask(out.index, start_hhmm, end_hhmm)
    out.loc[~mask, "p_norm"] = 0.0
    return out


def _clamp_series_to_daylight(s: pd.Series, start_hhmm: str, end_hhmm: str) -> pd.Series:
    s2 = s.copy()
    mask = _daylight_mask(s2.index, start_hhmm, end_hhmm)
    s2.loc[~mask] = 0.0
    return s2


def _add_cyc_5min(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    dow = idx.dayofweek.values.astype(float)
    tod_min = (idx.hour * 60 + idx.minute).values.astype(float)
    out = df.copy()
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["tod_sin"] = np.sin(2 * np.pi * tod_min / (24.0 * 60.0))
    out["tod_cos"] = np.cos(2 * np.pi * tod_min / (24.0 * 60.0))
    return out


def _add_cyc_hourly(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    dow = idx.dayofweek.values.astype(float)
    hod = idx.hour.values.astype(float)
    out = df.copy()
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["hod_sin"] = np.sin(2 * np.pi * hod / 24.0)
    out["hod_cos"] = np.cos(2 * np.pi * hod / 24.0)
    return out


def _ensure_dataframes(day_start: str, day_end: str) -> None:
    """Load and cache PV 5-min and hourly feature frames, applying daylight zeros."""
    global _DF5, _DFH
    if _DF5 is not None and _DFH is not None:
        return
    df_raw = pd.read_csv(PV_5MIN_CSV, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    if "p_norm" not in df_raw.columns:
        raise ValueError("PV CSV must contain column 'p_norm'.")
    df = _enforce_pv_zero_outside_window(df_raw[["p_norm"]], day_start, day_end)
    _DF5 = _add_cyc_5min(df.copy())

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
    req = ["p_norm", "dow_sin", "dow_cos", "tod_sin", "tod_cos"]
    idx_window = pd.date_range(end=t_now, periods=lookback_steps, freq="5min")
    Xw = df5.reindex(idx_window)
    if Xw[req].isna().any().any():
        raise ValueError("Insufficient 5-min history for VSTF window.")
    return Xw[req].values.astype(np.float32)[np.newaxis, :, :]


def _build_st_input(dfH: pd.DataFrame, anchor_hour: pd.Timestamp, lookback_steps: int) -> np.ndarray:
    req = ["p_norm", "dow_sin", "dow_cos", "hod_sin", "hod_cos"]
    anchor_hour = pd.Timestamp(anchor_hour).floor("h")
    idx_window = pd.date_range(end=anchor_hour, periods=lookback_steps, freq="h")
    Xw = dfH.reindex(idx_window)
    if Xw[req].isna().any().any():
        raise ValueError("Insufficient hourly history for ST window.")
    return Xw[req].values.astype(np.float32)[np.newaxis, :, :]


def _interp_hourly_to_5min(times_5: pd.DatetimeIndex, st_hourly: pd.Series) -> pd.Series:
    union_idx = st_hourly.index.union(times_5)
    s = st_hourly.reindex(union_idx).interpolate(method="time").reindex(times_5)
    if s.isna().any():
        s = s.bfill().ffill()
    return s

def pv(
    dt,
    intervals: Optional[int] = None,
    *,
    model_vstf,
    model_st,
    vstf_cfg: Optional[Dict] = None,
    st_cfg: Optional[Dict] = None,
    day_start: str = "06:00",
    day_end: str = "19:00",
    clip_to_unit: bool = False,
) -> np.ndarray:
    """Return a normalized five-minute PV forecast."""
    if model_vstf is None or model_st is None:
        raise ValueError("Both model_vstf and model_st must be provided.")

    cfg_v = (vstf_cfg or PV_VSTF_DEFAULT).copy()
    cfg_h = (st_cfg or PV_ST_DEFAULT).copy()
    if cfg_v["step_minutes"] != 5:
        raise ValueError("VSTF step_minutes must be 5.")
    if cfg_h["step_minutes"] != 60:
        raise ValueError("ST step_minutes must be 60.")

    _ensure_dataframes(day_start, day_end)

    t_now = pd.Timestamp(dt)
    if intervals is None:
        intervals = MAX_INTERVALS
    intervals = int(intervals)
    if intervals <= 0:
        raise ValueError("intervals must be > 0.")
    if intervals > MAX_INTERVALS:
        intervals = MAX_INTERVALS

    X_v = _build_vstf_input(_DF5, t_now, cfg_v["lookback_steps"])
    v_pred = model_vstf.predict(X_v, verbose=0).reshape(-1)
    use_vstf_steps = int(min(cfg_v["horizon_steps"], len(v_pred), intervals))

    # Hourly rows are left-labelled, so the active hour is not yet causal.
    anchor_hour = last_completed_hour(t_now)
    X_h = _build_st_input(_DFH, anchor_hour, cfg_h["lookback_steps"])
    st_pred = model_st.predict(X_h, verbose=0).reshape(-1)

    hours_needed = int(np.ceil(intervals / 12.0))
    st_pred = st_pred[:max(hours_needed, 1)]
    st_times = pd.date_range(start=hourly_forecast_start(anchor_hour), periods=len(st_pred), freq="h")
    st_hourly = pd.Series(st_pred, index=st_times)

    times_5 = pd.date_range(start=t_now + pd.Timedelta(minutes=5), periods=intervals, freq="5min")
    st_5 = _interp_hourly_to_5min(times_5, st_hourly)
    st_5 = _clamp_series_to_daylight(st_5, day_start, day_end)

    out = st_5.values.astype(np.float32)
    if use_vstf_steps > 0:
        vstf_5 = pd.Series(v_pred[:use_vstf_steps], index=times_5[:use_vstf_steps])
        vstf_5 = _clamp_series_to_daylight(vstf_5, day_start, day_end)
        out[:use_vstf_steps] = vstf_5.values.astype(np.float32)

    out = np.maximum(out, 0.0)
    if clip_to_unit:
        out = np.minimum(out, 1.0)
    return out
