# -*- coding: utf-8 -*-
"""
Unified LOAD & PV VSTF+ST real-time simulation (36h ahead @ 5-min)

Key behavior
------------
- Output grid: strictly every 5 minutes for 36h (432 points).
- VSTF (5-min model) is used ONLY until the NEXT full hour after t_now.
- ST (hourly model) is used FROM the next full hour onward, interpolated to 5-min.
- PV preprocessing: p_norm = 0 outside [DAY_START, DAY_END); forecasts also clamped.
- Debug plots:
    * DEBUG_ENABLED toggles plotting.
    * DEBUG_PLOT_EVERY controls frequency (in queries).
    * DEBUG_SHOW_LAST_K overlays the last K queries on one figure (K>=1).
    * SIM_ADVANCE_STEPS controls time step between queries (1=5min, 12=1h, etc.).
- Separate model defaults for LOAD vs PV (VSTF/ST can differ).

Author: You
"""

import os
import json
import time
from collections import deque
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model

# =========================
# User config
# =========================
# Test CSVs (must contain columns: timestamp,p_norm)
LOAD_TEST_CSV = "data/load_5min_test.csv"
PV_TEST_CSV   = "data/pv_5min_test.csv"

# PV daylight window (inclusive start, exclusive end)
DAY_START = "06:00"
DAY_END   = "19:00"

# Hourly aggregation used in training (must match)
TRAIN_RESAMPLE_AGG = "mean"  # mean | median | sum

# Horizon / cadence
TOTAL_HOURS = 36                 # final horizon (36h at 5-min cadence)
STEP_MIN    = 5                  # do not change (VSTF trained at 5-min)
H_STEPS     = TOTAL_HOURS * 60 // STEP_MIN  # 432

# --------- Separate defaults per DOMAIN (used if no *_config.json) ---------
# LOAD models
LOAD_VSTF_DEFAULT = {"step_minutes": 5,  "lookback_steps": 96, "horizon_steps": 48}
LOAD_ST_DEFAULT   = {"step_minutes": 60, "lookback_steps": 8,  "horizon_steps": 36}
# PV models
PV_VSTF_DEFAULT   = {"step_minutes": 5,  "lookback_steps": 48, "horizon_steps": 12}
PV_ST_DEFAULT     = {"step_minutes": 60, "lookback_steps": 8,  "horizon_steps": 36}

# Model path helpers (edit if your filenames differ)
def _candidates_vstf(domain: str, step_min: int, total_min: int, lb: int) -> List[str]:
    return [
        f"models/lstm_vstf_{domain}_{step_min}min_{total_min}min_{lb}.keras",
        f"models/llstm_vstf_{domain}_{step_min}min_{total_min}min_{lb}.keras",
    ]

def _path_hourly(domain: str, H: int, LB: int) -> str:
    return f"models/lstm_hourly_{domain}_{H}h_{LB}.keras"

# --------- Simulation window ---------
INITIAL_QUERY_TIME = pd.Timestamp("2009-04-30 23:55:00")  #None  # None -> auto; or set e.g. "2009-04-30 23:55:00"
STOP_AFTER_DAYS    = 20     # simulate up to N days (wall clock) from INITIAL_QUERY_TIME
MAX_QUERIES        = 800  # None or an integer to limit number of iterations

# --------- Simulation step & debug ---------
SIM_ADVANCE_STEPS = 1   # how many 5-min steps to advance per query (1=5min, 12=1h, 36=3h, …)
DEBUG_ENABLED     = True
DEBUG_PLOT_EVERY  = 7  # plot every N queries (0 disables; typical 12=~hourly if SIM_ADVANCE_STEPS=1)
DEBUG_SHOW_LAST_K = 1   # overlay the last K forecasts on each debug figure (K>=1)
DEBUG_MAX_PLOTS   = 48

# Output
SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# Daylight & PV utilities
# =========================
def _hhmm_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)

def _daylight_mask(index: pd.DatetimeIndex, start_hhmm: str, end_hhmm: str) -> np.ndarray:
    """True if timestamp is within [start, end) by local clock; handles midnight crossing."""
    start = _hhmm_to_minutes(start_hhmm)
    end   = _hhmm_to_minutes(end_hhmm)
    minutes = (index.hour * 60 + index.minute).astype(int)
    if start <= end:
        return (minutes >= start) & (minutes < end)
    else:
        # window crosses midnight
        return (minutes >= start) | (minutes < end)

def enforce_pv_zero_outside_window(df: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    if "p_norm" not in df.columns:
        raise ValueError("DataFrame must contain 'p_norm'.")
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
    """Add DOW and time-of-day sin/cos at 5-min cadence."""
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
    """Add DOW and hour-of-day sin/cos at hourly cadence."""
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
    cfg_path = model_path[:-6] + "_config.json" if model_path.endswith(".keras") else model_path + "_config.json"
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            meta = json.load(f)
        return meta.get("config", meta)
    return None

def get_model_cfg(path: str, fallback: Dict) -> Dict:
    cfg = _try_load_config(path)
    if cfg is None:
        cfg = fallback.copy()
    return {
        "step_minutes": int(cfg.get("step_minutes", fallback["step_minutes"])),
        "lookback_steps": int(cfg.get("lookback_steps", fallback["lookback_steps"])),
        "horizon_steps": int(cfg.get("horizon_steps", fallback["horizon_steps"])),
    }

def load_keras_safely(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return load_model(path, compile=False, safe_mode=False)

def resolve_existing_path(cands: List[str]) -> str:
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("None of these model paths exist:\n" + "\n".join(cands))

# =========================
# Inputs (end-aligned)
# =========================
def build_vstf_input(df5: pd.DataFrame, t_now: pd.Timestamp, lookback_steps: int) -> np.ndarray:
    """Build a 5-min VSTF input window ending at t_now (inclusive)."""
    req = ["p_norm","dow_sin","dow_cos","tod_sin","tod_cos"]
    if not all(c in df5.columns for c in req):
        raise ValueError("Missing 5-min features.")
    idx_window = pd.date_range(end=t_now, periods=lookback_steps, freq="5min")
    Xw = df5.reindex(idx_window)
    if Xw["p_norm"].isna().any():
        raise ValueError("Not enough 5-min history for VSTF window.")
    return Xw[req].values.astype(np.float32)[np.newaxis, :, :]

def build_st_input_for_anchor(dfH: pd.DataFrame, anchor_hour: pd.Timestamp, lookback_steps: int) -> np.ndarray:
    """
    Build hourly ST input ending at anchor_hour (inclusive).
    ST will predict starting at (anchor_hour + 1h).
    """
    req = ["p_norm","dow_sin","dow_cos","hod_sin","hod_cos"]
    if not all(c in dfH.columns for c in req):
        raise ValueError("Missing hourly features.")
    anchor_hour = pd.Timestamp(anchor_hour).floor("h")
    idx_window = pd.date_range(end=anchor_hour, periods=lookback_steps, freq="h")
    Xw = dfH.reindex(idx_window)
    if Xw["p_norm"].isna().any():
        raise ValueError("Not enough hourly history for ST window.")
    return Xw[req].values.astype(np.float32)[np.newaxis, :, :]

# =========================
# Interpolation & combination
# =========================
def interpolate_hourly_to_5min(times_5: pd.DatetimeIndex, st_hourly: pd.Series) -> pd.Series:
    """Time interpolation of hourly ST to the 5-min grid."""
    union_idx = st_hourly.index.union(times_5)
    return st_hourly.reindex(union_idx).interpolate(method="time").reindex(times_5)

def steps_until_next_full_hour(t_now: pd.Timestamp) -> int:
    """
    Number of 5-min steps from (t_now + 5min) up to the next full hour (exclusive).
    Enforces: VSTF only up to next full hour; ST from that boundary onward.
    """
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
    model_vstf,
    model_st,
    vstf_cfg: Dict,
    st_cfg: Dict,
    clamp_daylight: Optional[Tuple[str, str]] = None,
    st_cache: Optional[Dict[pd.Timestamp, pd.Series]] = None,
    total_hours: int = TOTAL_HOURS,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Build a 5-min forecast for 36h combining VSTF (up to next full hour) and ST thereafter.
    Returns: combined_5min, vstf_5min_used, st_hourly_points, st_only_5min
    """
    assert vstf_cfg["step_minutes"] == 5
    assert st_cfg["step_minutes"] == 60
    assert st_cfg["horizon_steps"] >= total_hours

    # VSTF
    X_vstf = build_vstf_input(df5, t_now, lookback_steps=vstf_cfg["lookback_steps"])
    vstf_pred = model_vstf.predict(X_vstf, verbose=0).reshape(-1)  # 12 steps available

    # ST (cache by next_full_hour to avoid recomputation)
    next_full_hour = (t_now + pd.Timedelta(minutes=5)).ceil("h")
    if st_cache is not None and next_full_hour in st_cache:
        st_hourly = st_cache[next_full_hour]
    else:
        anchor_hour = next_full_hour - pd.Timedelta(hours=1)
        X_st = build_st_input_for_anchor(dfH, anchor_hour, lookback_steps=st_cfg["lookback_steps"])
        st_pred = model_st.predict(X_st, verbose=0).reshape(-1)[:st_cfg["horizon_steps"]]
        st_times_hourly = pd.date_range(start=next_full_hour, periods=st_cfg["horizon_steps"], freq="h")
        st_hourly = pd.Series(st_pred, index=st_times_hourly)
        if clamp_daylight is not None:
            st_hourly = clamp_series_to_daylight(st_hourly, *clamp_daylight)
        if st_cache is not None:
            st_cache[next_full_hour] = st_hourly

    # 5-min time grid across full horizon
    times_5 = pd.date_range(start=t_now + pd.Timedelta(minutes=5), periods=total_hours * 12, freq="5min")

    # Interpolate ST to 5-min and clamp if PV
    st_only_5min = interpolate_hourly_to_5min(times_5, st_hourly)
    if clamp_daylight is not None:
        st_only_5min = clamp_series_to_daylight(st_only_5min, *clamp_daylight)

    # VSTF chunk actually used: ONLY until the next full hour
    use_vstf_steps = steps_until_next_full_hour(t_now)  # 0..11
    vstf_5min = pd.Series(vstf_pred[:use_vstf_steps], index=times_5[:use_vstf_steps])
    if clamp_daylight is not None and len(vstf_5min) > 0:
        vstf_5min = clamp_series_to_daylight(vstf_5min, *clamp_daylight)

    # Combine: start from ST-only and overwrite the first 'use_vstf_steps' with VSTF
    combined = st_only_5min.copy()
    if use_vstf_steps > 0:
        combined.iloc[:use_vstf_steps] = vstf_5min.values

    #min value should be 0
    combined[combined < 0] = 0.0
    vstf_5min[vstf_5min < 0] = 0.0
    st_only_5min[st_only_5min < 0] = 0.0
    st_hourly[st_hourly < 0] = 0.0

    # Final clamp for PV (idempotent)
    if clamp_daylight is not None:
        combined = clamp_series_to_daylight(combined, *clamp_daylight)

    # Safety: lengths must be exactly 432 (36h @ 5 min)
    assert len(combined) == H_STEPS, f"Combined length {len(combined)} != {H_STEPS}"
    assert len(st_only_5min) == H_STEPS, f"ST-5min length {len(st_only_5min)} != {H_STEPS}"

    return combined, vstf_5min, st_hourly, st_only_5min

# =========================
# Debug plot (overlay last K queries)
# =========================
def plot_overlay_both(
    samples: List[Dict],
    save_dir: str,
    fig_idx: int
):
    """
    Overlay up to K recent queries for LOAD & PV on a single figure.
    For K>1: older queries are lighter; newest is bold.
    Each sample dict must contain:
      - 't_now', 'dfl5', 'dfp5'
      - 'comb_load', 'comb_pv', 'vstf_load', 'vstf_pv', 'st5_load', 'st5_pv', 'stH_load', 'stH_pv'
    """
    assert len(samples) > 0
    K = len(samples)
    plt.figure(figsize=(14.5, 6.5))

    # Plot last 24h GT for the MOST RECENT sample only (background context)
    s = samples[-1]
    t_now = s["t_now"]
    last_day_idx = pd.date_range(end=t_now, periods=24*12, freq="5min")
    plt.plot(last_day_idx, s["dfl5"]["p_norm"].reindex(last_day_idx),
             label="LOAD last 24h (5-min)", linewidth=1.0, alpha=0.7)
    plt.plot(last_day_idx, s["dfp5"]["p_norm"].reindex(last_day_idx),
             label="PV last 24h (5-min)", linewidth=1.0, alpha=0.7)

    # Overlay each sample's 36h GT + Combined (older -> lighter)
    for j, ss in enumerate(samples):
        alpha = 0.35 + 0.65 * (j + 1) / K  # fade older ones
        lw_pred = 1.2 if j < K - 1 else 2.0  # newest thicker
        marker_size = 2.5 if j < K - 1 else 3.5

        gt_load = ss["dfl5"]["p_norm"].reindex(ss["comb_load"].index).values
        gt_pv   = ss["dfp5"]["p_norm"].reindex(ss["comb_pv"].index).values

        # Show ALL 5-min points with markers to emphasize cadence
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

    # For the newest sample only, also show ST-only and VSTF chunk + hourly points
    plt.plot(s["st5_load"].index, s["st5_load"].values, linestyle="--", linewidth=1.0, alpha=0.8, label="LOAD ST-only→5min")
    plt.plot(s["st5_pv"].index,   s["st5_pv"].values,   linestyle="--", linewidth=1.0, alpha=0.8, label="PV ST-only→5min")
    if len(s["vstf_load"]) > 0:
        plt.plot(s["vstf_load"].index, s["vstf_load"].values, linewidth=2.2, label="LOAD VSTF (to next hour)")
    if len(s["vstf_pv"]) > 0:
        plt.plot(s["vstf_pv"].index, s["vstf_pv"].values, linewidth=2.2, label="PV VSTF (to next hour)")
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
    plt.show()
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

    # hourly resample for LOAD
    if TRAIN_RESAMPLE_AGG == "mean":
        dflH = dfl[["p_norm"]].resample("h").mean(numeric_only=True).dropna()
    elif TRAIN_RESAMPLE_AGG == "median":
        dflH = dfl[["p_norm"]].resample("h").median(numeric_only=True).dropna()
    elif TRAIN_RESAMPLE_AGG == "sum":
        dflH = dfl[["p_norm"]].resample("h").sum(numeric_only=True).dropna()
    else:
        raise ValueError("Unsupported TRAIN_RESAMPLE_AGG.")
    dflH = add_cyclical_features_hourly(dflH)

    # ---------- PV ----------
    dfp_raw = pd.read_csv(PV_TEST_CSV, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    if "p_norm" not in dfp_raw.columns:
        raise ValueError("PV CSV must contain 'p_norm'.")
    # preprocess PV: zero at night BEFORE features
    dfp = enforce_pv_zero_outside_window(dfp_raw[["p_norm"]], DAY_START, DAY_END)
    dfp5 = add_cyclical_features_5min(dfp.copy())

    # hourly resample for PV
    if TRAIN_RESAMPLE_AGG == "mean":
        dfpH = dfp[["p_norm"]].resample("h").mean(numeric_only=True).dropna()
    elif TRAIN_RESAMPLE_AGG == "median":
        dfpH = dfp[["p_norm"]].resample("h").median(numeric_only=True).dropna()
    elif TRAIN_RESAMPLE_AGG == "sum":
        dfpH = dfp[["p_norm"]].resample("h").sum(numeric_only=True).dropna()
    else:
        raise ValueError("Unsupported TRAIN_RESAMPLE_AGG.")
    dfpH = add_cyclical_features_hourly(dfpH)

    # ---------- Resolve model paths (separate defaults) ----------
    # LOAD models
    load_vstf_path = resolve_existing_path(
        _candidates_vstf("load", LOAD_VSTF_DEFAULT["step_minutes"],
                         LOAD_VSTF_DEFAULT["horizon_steps"]*LOAD_VSTF_DEFAULT["step_minutes"],
                         LOAD_VSTF_DEFAULT["lookback_steps"])
    )
    load_st_path   = _path_hourly("load", LOAD_ST_DEFAULT["horizon_steps"], LOAD_ST_DEFAULT["lookback_steps"])
    if not os.path.exists(load_st_path):
        raise FileNotFoundError(load_st_path)

    # PV models
    pv_vstf_path = resolve_existing_path(
        _candidates_vstf("pv", PV_VSTF_DEFAULT["step_minutes"],
                         PV_VSTF_DEFAULT["horizon_steps"]*PV_VSTF_DEFAULT["step_minutes"],
                         PV_VSTF_DEFAULT["lookback_steps"])
    )
    pv_st_path   = _path_hourly("pv", PV_ST_DEFAULT["horizon_steps"], PV_ST_DEFAULT["lookback_steps"])
    if not os.path.exists(pv_st_path):
        raise FileNotFoundError(pv_st_path)

    # ---------- Load models & their configs ----------
    m_vstf_load = load_keras_safely(load_vstf_path)
    m_st_load   = load_keras_safely(load_st_path)
    m_vstf_pv   = load_keras_safely(pv_vstf_path)
    m_st_pv     = load_keras_safely(pv_st_path)

    vstf_cfg_load = get_model_cfg(load_vstf_path, LOAD_VSTF_DEFAULT)
    st_cfg_load   = get_model_cfg(load_st_path,   LOAD_ST_DEFAULT)
    vstf_cfg_pv   = get_model_cfg(pv_vstf_path,   PV_VSTF_DEFAULT)
    st_cfg_pv     = get_model_cfg(pv_st_path,     PV_ST_DEFAULT)

    # ---------- Determine simulation range ----------
    # Auto INITIAL_QUERY_TIME if None: as late as possible while keeping 36h horizon and VSTF lookback for both series
    last_idx_common = min(dfl5.index.max(), dfp5.index.max())
    # ensure full 36h horizon available after t_now
    t_now_auto = last_idx_common - pd.Timedelta(hours=TOTAL_HOURS, minutes=5)
    # ensure VSTF history available (max lookback across domains)
    max_lb_steps = max(vstf_cfg_load["lookback_steps"], vstf_cfg_pv["lookback_steps"])
    earliest_ok = max(dfl5.index.min(), dfp5.index.min()) + pd.Timedelta(minutes=5 * max_lb_steps)

    t_start = pd.Timestamp(INITIAL_QUERY_TIME) if INITIAL_QUERY_TIME else max(t_now_auto, earliest_ok)
    # stop conditions
    t_last = last_idx_common  # data end
    t_stop_by_horizon = t_last - pd.Timedelta(hours=TOTAL_HOURS)  # last t_now that allows 36h ahead
    t_stop_by_days = t_start + pd.Timedelta(days=STOP_AFTER_DAYS)
    t_stop = min(t_stop_by_horizon, t_stop_by_days)
    if t_stop <= t_start:
        raise ValueError("Insufficient test span for the requested simulation window.")

    step_delta = pd.Timedelta(minutes=5 * SIM_ADVANCE_STEPS)
    print(f"Simulating from {t_start} to {t_stop} every {SIM_ADVANCE_STEPS*5} minutes..."
          f"  (debug={'on' if DEBUG_ENABLED else 'off'}, plot_every={DEBUG_PLOT_EVERY}, show_last_k={DEBUG_SHOW_LAST_K})")

    # ---------- Simulation loop ----------
    st_cache_load: Dict[pd.Timestamp, pd.Series] = {}
    st_cache_pv:   Dict[pd.Timestamp, pd.Series] = {}

    # Buffers for debug overlays
    overlay_buf = deque(maxlen=max(1, DEBUG_SHOW_LAST_K))

    # Save just one example CSV per domain
    saved_example_load = False
    saved_example_pv   = False

    rows = []  # runtime log (optional)
    plot_counter = 0
    cur = t_start
    qcount = 0

    while cur <= t_stop and (MAX_QUERIES is None or qcount < MAX_QUERIES):
        try:
            t0 = time.perf_counter()
            # LOAD
            comb_load, vstf_load, stH_load, st5_load = forecast_combined_36h_5min(
                t_now=cur, df5=dfl5, dfH=dflH,
                model_vstf=m_vstf_load, model_st=m_st_load,
                vstf_cfg=vstf_cfg_load, st_cfg=st_cfg_load,
                clamp_daylight=None, st_cache=st_cache_load,
                total_hours=TOTAL_HOURS
            )
            # PV
            comb_pv, vstf_pv, stH_pv, st5_pv = forecast_combined_36h_5min(
                t_now=cur, df5=dfp5, dfH=dfpH,
                model_vstf=m_vstf_pv, model_st=m_st_pv,
                vstf_cfg=vstf_cfg_pv, st_cfg=st_cfg_pv,
                clamp_daylight=(DAY_START, DAY_END), st_cache=st_cache_pv,
                total_hours=TOTAL_HOURS
            )
            runtime_s = time.perf_counter() - t0
        except Exception as e:
            # often the very first steps lack enough history; skip forward
            cur += step_delta
            continue

        # Save one example CSV per domain (first successful query)
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

        # Prepare sample for overlay buffer
        overlay_buf.append({
            "t_now": cur,
            "dfl5": dfl5,
            "dfp5": dfp5,
            "comb_load": comb_load, "comb_pv": comb_pv,
            "vstf_load": vstf_load, "vstf_pv": vstf_pv,
            "st5_load": st5_load,   "st5_pv": st5_pv,
            "stH_load": stH_load,   "stH_pv": stH_pv
        })

        # Optional debug plot with frequency and overlay of last K queries
        if DEBUG_ENABLED and DEBUG_PLOT_EVERY > 0 and (plot_counter < DEBUG_MAX_PLOTS):
            if (qcount % DEBUG_PLOT_EVERY) == 0:
                plot_overlay_both(list(overlay_buf), SAVE_DIR, plot_counter)
                plot_counter += 1

        rows.append({"query_time": cur, "runtime_s": runtime_s})
        cur += step_delta
        qcount += 1

    # Runtime log
    pd.DataFrame(rows).to_csv(os.path.join(SAVE_DIR, "realtime_runtime_log.csv"), index=False)
    print("Simulation finished.")

if __name__ == "__main__":
    main()
