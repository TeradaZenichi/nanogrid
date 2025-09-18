# -*- coding: utf-8 -*-
"""
Real-time simulation of 36-hour ahead PV forecasting at 5-min resolution
using two pre-trained Keras models:
 - Very-Short-Term (VSTF): 5-min model that outputs the next hour (12 steps)
 - Short-Term (ST): hourly model that outputs the next H hours (default 36)

PV daylight enforcement
-----------------------
- As a preprocessing step, PV is forced to zero outside the daylight window.
- After forecasting, predictions are clamped to zero outside the daylight window too.
- Default daylight window: 06:00 (inclusive) to 19:00 (exclusive).
  Adjust via DAY_START / DAY_END if needed.

Combination (robust):
 - Interpolate the ST hourly forecast to a 5-min grid across the full 36 h horizon.
 - Replace the first 'vstf_hours' * 12 steps with the VSTF (5-min) output.
 - Clamp combined/ST/VSTF series to daylight window (zeros at night).

Debug plotting:
 - (Optional) last real 24h (5-min), 36h GT (5-min), combined (5-min), VSTF (5-min),
   ST-only (5-min), plus ST hourly points.

Author: You
"""

import os
import time
import json
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# ----------------------------
# User configuration
# ----------------------------
TEST_CSV_PATH = r"data/pv_5min_test.csv"  # <-- PV file with columns: timestamp,p_norm

# Daylight window (default PV-producing hours); inclusive start, exclusive end
DAY_START = "06:00"
DAY_END   = "19:00"

# Optional config sidecar JSONs (created by the training scripts). If present, they define lookbacks/horizons.
VSTF_DEFAULT = {"step_minutes": 5, "lookback_steps": 48, "horizon_steps": 12}   # 2h lookback, 1h horizon
ST_DEFAULT   = {"step_minutes": 60, "lookback_steps": 8, "horizon_steps": 36}   # 8h lookback, 36h horizon

# Model paths constructed based on the config values above (PV models!)
VSTF_MODEL_PATH = (
    f"models/lstm_vstf_pv_{VSTF_DEFAULT['step_minutes']}min_"
    f"{VSTF_DEFAULT['horizon_steps']*VSTF_DEFAULT['step_minutes']}min_"
    f"{VSTF_DEFAULT['lookback_steps']}.keras"
)
ST_MODEL_PATH = (
    f"models/lstm_hourly_pv_{ST_DEFAULT['horizon_steps']}h_"
    f"{ST_DEFAULT['lookback_steps']}.keras"
)

# Combination parameters
TOTAL_HOURS        = 36     # final horizon in hours (vector at 5-min)
VSTF_HOURS         = 1      # first hours provided by VSTF
TRAIN_RESAMPLE_AGG = "mean" # must match the hourly training aggregation

# Simulation window
INITIAL_QUERY_TIME = pd.Timestamp("2009-04-30 23:55:00")  # inclusive
STOP_AFTER_DAYS    = 1                                    # simulate N days

# Debug plot control
DEBUG_PLOT_EVERY = 12    # 1 => every sample; 12 => 1 per hour; 0 => no plots
DEBUG_MAX_PLOTS  = 48

# Persist ST hourly cache (useful for audits)
PERSIST_ST_CACHE = True
SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)
ST_CACHE_PATH = os.path.join(SAVE_DIR, "st_hourly_cache_pv.csv")

# ----------------------------
# Metrics
# ----------------------------
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true - y_pred
    return float(np.mean(diff * diff))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)

# ----------------------------
# Keras custom (for some environments)
# ----------------------------
@tf.function
def r2_keras(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    mean_y = tf.reduce_mean(y_true)
    ss_tot = tf.reduce_sum(tf.square(y_true - mean_y))
    return 1.0 - ss_res / (ss_tot + tf.keras.backend.epsilon())

# ----------------------------
# Daylight utilities (PV zeroing)
# ----------------------------
def _hhmm_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)

def _daylight_mask(index: pd.DatetimeIndex, start_hhmm: str, end_hhmm: str) -> np.ndarray:
    """
    Build a boolean mask that's True if timestamp is within [start, end) by local clock.
    Works for both 5-min and hourly indices. Handles windows crossing midnight too.
    """
    start = _hhmm_to_minutes(start_hhmm)
    end   = _hhmm_to_minutes(end_hhmm)
    minutes = (index.hour * 60 + index.minute).astype(int)
    if start <= end:
        # e.g., 06:00..19:00
        return (minutes >= start) & (minutes < end)
    else:
        # window crosses midnight, e.g., 19:00..06:00
        return (minutes >= start) | (minutes < end)

def enforce_pv_zero_outside_window(df: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    """
    Force column 'p_norm' to zero outside the daylight window, in-place on a copy.
    """
    if "p_norm" not in df.columns:
        raise ValueError("DataFrame must contain column 'p_norm'.")
    df2 = df.copy()
    mask = _daylight_mask(df2.index, start_hhmm, end_hhmm)
    df2.loc[~mask, "p_norm"] = 0.0
    return df2

def clamp_series_to_daylight(s: pd.Series, start_hhmm: str, end_hhmm: str) -> pd.Series:
    """
    Return a copy of 's' (PV prediction) with values set to 0 outside the daylight window.
    """
    s2 = s.copy()
    mask = _daylight_mask(s2.index, start_hhmm, end_hhmm)
    s2.loc[~mask] = 0.0
    return s2

# ----------------------------
# Feature engineering
# ----------------------------
def add_cyclical_features_5min(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    dow = idx.dayofweek.values.astype(float)
    dow_ang = 2.0 * np.pi * dow / 7.0
    df["dow_sin"] = np.sin(dow_ang)
    df["dow_cos"] = np.cos(dow_ang)

    tod_min = (idx.hour * 60 + idx.minute).values.astype(float)  # 0..1435
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

    hod = idx.hour.values.astype(float)  # 0..23
    hod_ang = 2.0 * np.pi * hod / 24.0
    df["hod_sin"] = np.sin(hod_ang)
    df["hod_cos"] = np.cos(hod_ang)
    return df

# ----------------------------
# Model/config loading
# ----------------------------
def _try_load_config(model_path: str) -> Optional[Dict]:
    if model_path.lower().endswith(".keras"):
        cfg_path = model_path[:-6] + "_config.json"
    else:
        cfg_path = model_path + "_config.json"
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            meta = json.load(f)
        return meta.get("config", meta)
    return None

def load_keras_safely(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return load_model(path, compile=False, safe_mode=False)

def get_model_cfg(path: str, fallback: Dict) -> Dict:
    cfg = _try_load_config(path)
    if cfg is None:
        cfg = fallback.copy()
    return {
        "step_minutes": int(cfg.get("step_minutes", fallback["step_minutes"])),
        "lookback_steps": int(cfg.get("lookback_steps", fallback["lookback_steps"])),
        "horizon_steps": int(cfg.get("horizon_steps", fallback["horizon_steps"])),
    }

# ----------------------------
# Inputs builders (end-aligned)
# ----------------------------
def build_vstf_input(df5: pd.DataFrame, t_now: pd.Timestamp, lookback_steps: int) -> np.ndarray:
    """5-min input ending at t_now (inclusive)."""
    req_cols = ["p_norm","dow_sin","dow_cos","tod_sin","tod_cos"]
    if not all(c in df5.columns for c in req_cols):
        raise ValueError("Missing 5-min features. Run add_cyclical_features_5min first.")
    idx_window = pd.date_range(end=t_now, periods=lookback_steps, freq="5min")
    Xw = df5.reindex(idx_window)
    if Xw["p_norm"].isna().any():
        raise ValueError("Not enough 5-min history to build VSTF window.")
    X = Xw[req_cols].values.astype(np.float32)
    return X[np.newaxis, :, :]

def build_st_input_for_anchor(dfH: pd.DataFrame, anchor_hour: pd.Timestamp, lookback_steps: int) -> np.ndarray:
    """
    Hourly input ending at `anchor_hour` (inclusive).
    The ST forecast will start at anchor_hour + 1h.
    """
    req_cols = ["p_norm","dow_sin","dow_cos","hod_sin","hod_cos"]
    if not all(c in dfH.columns for c in req_cols):
        raise ValueError("Missing hourly features. Run add_cyclical_features_hourly first.")
    anchor_hour = pd.Timestamp(anchor_hour).floor("h")
    idx_window = pd.date_range(end=anchor_hour, periods=lookback_steps, freq="h")
    Xw = dfH.reindex(idx_window)
    if Xw["p_norm"].isna().any():
        raise ValueError("Not enough hourly history to build ST window.")
    X = Xw[req_cols].values.astype(np.float32)
    return X[np.newaxis, :, :]

# ----------------------------
# ST cache utilities
# ----------------------------
def persist_st_hourly(start_hour: pd.Timestamp, st_hourly: pd.Series):
    """Optionally append ST forecast (already daylight-clamped) to a CSV for audit."""
    if not PERSIST_ST_CACHE:
        return
    df_out = pd.DataFrame({
        "start_hour": [start_hour]*len(st_hourly),
        "timestamp": st_hourly.index,
        "pred": st_hourly.values
    })
    header = not os.path.exists(ST_CACHE_PATH)
    df_out.to_csv(ST_CACHE_PATH, mode="a", header=header, index=False)

# ----------------------------
# Forecast functions
# ----------------------------
def interpolate_hourly_to_5min_over(times_5: pd.DatetimeIndex, st_hourly: pd.Series) -> pd.Series:
    """Interpolate hourly ST to the full 5-min grid 'times_5' (no trimming)."""
    union_idx = st_hourly.index.union(times_5)
    st_interp_full = st_hourly.reindex(union_idx).interpolate(method="time").reindex(times_5)
    return st_interp_full

def forecast_combined_36h_5min(
    t_now: pd.Timestamp,
    df5: pd.DataFrame,
    dfH: pd.DataFrame,
    model_vstf,
    model_st,
    vstf_cfg: Dict,
    st_cfg: Dict,
    st_cache: Dict[pd.Timestamp, pd.Series],
    total_hours: int = TOTAL_HOURS,
    vstf_hours: int = VSTF_HOURS,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Produce a combined 36h forecast at 5-min resolution starting right after t_now.

    Returns:
      combined_5min : pd.Series (index 5-min), length total_hours*12
      vstf_5min     : pd.Series (first 'vstf_hours' at 5-min, daylight-clamped)
      st_hourly     : pd.Series (hourly points starting at next_full_hour, daylight-clamped)
      st_only_5min  : pd.Series (ST interpolated to 5-min across the horizon, daylight-clamped)
    """
    assert vstf_cfg["step_minutes"] == 5,  "VSTF model must be 5-min."
    assert st_cfg["step_minutes"] == 60,   "ST model must be hourly."
    assert vstf_cfg["horizon_steps"] >= 12 * vstf_hours
    assert st_cfg["horizon_steps"] >= total_hours

    # VSTF input at t_now
    X_vstf = build_vstf_input(df5, t_now, lookback_steps=vstf_cfg["lookback_steps"])
    vstf_pred = model_vstf.predict(X_vstf, verbose=0).reshape(-1)

    # ST forecast anchored to the next full hour
    next_full_hour = (t_now + pd.Timedelta(minutes=5)).ceil("h")
    if next_full_hour in st_cache:
        st_hourly_raw = st_cache[next_full_hour]
    else:
        anchor_hour = next_full_hour - pd.Timedelta(hours=1)  # input ends here (inclusive)
        X_st = build_st_input_for_anchor(dfH, anchor_hour, lookback_steps=st_cfg["lookback_steps"])
        st_pred = model_st.predict(X_st, verbose=0).reshape(-1)[:st_cfg["horizon_steps"]]
        st_times_hourly = pd.date_range(start=next_full_hour, periods=st_cfg["horizon_steps"], freq="h")
        st_hourly_raw = pd.Series(st_pred, index=st_times_hourly)
        st_cache[next_full_hour] = st_hourly_raw

    # Daylight clamp on ST hourly (avoid night interpolation artifacts)
    st_hourly = clamp_series_to_daylight(st_hourly_raw, DAY_START, DAY_END)
    if PERSIST_ST_CACHE:
        persist_st_hourly(next_full_hour, st_hourly)

    # Time grids
    start_5min = t_now + pd.Timedelta(minutes=5)
    times_5    = pd.date_range(start=start_5min, periods=total_hours*12, freq="5min")

    # Interpolate ST to 5-min over the full horizon and clamp again
    st_only_5min = interpolate_hourly_to_5min_over(times_5, st_hourly)
    st_only_5min = clamp_series_to_daylight(st_only_5min, DAY_START, DAY_END)

    # VSTF as a 5-min series over the first hour and clamp to daylight
    vstf_steps = vstf_hours * 12
    vstf_5min  = pd.Series(vstf_pred[:vstf_steps], index=times_5[:vstf_steps])
    vstf_5min  = clamp_series_to_daylight(vstf_5min, DAY_START, DAY_END)

    # Combine: copy ST and replace first 'vstf_steps' with VSTF
    combined = st_only_5min.copy()
    combined.iloc[:vstf_steps] = vstf_5min.values

    # Final safety clamp (idempotent) to guarantee zeros outside daylight
    combined   = clamp_series_to_daylight(combined, DAY_START, DAY_END)

    return combined, vstf_5min, st_hourly, st_only_5min

# ----------------------------
# Debug plot (optional; disabled by default)
# ----------------------------
def plot_debug(
    t_now: pd.Timestamp,
    df5: pd.DataFrame,
    combined: pd.Series,
    vstf_5min: pd.Series,
    st_only_5min: pd.Series,
    st_hourly: pd.Series,
    gt_36h: np.ndarray,
    save_dir: str,
    idx_plot: int
):
    """
    Plot: last real 24h (5-min), 36h horizon GT (5-min),
          combined (5-min), VSTF (5-min), ST-only (5-min) + ST hourly points.
    """
    plt.figure(figsize=(12.5, 5.6))

    # last real day (5-min)
    last_day_idx = pd.date_range(end=t_now, periods=24*12, freq="5min")
    df_last = df5.reindex(last_day_idx)["p_norm"]
    plt.plot(df_last.index, df_last.values, label="Real last 24h (5-min)", linewidth=1.2)

    # horizon & series
    plt.plot(combined.index, combined.values, label="Pred COMBINED (5-min)", linewidth=1.6)
    plt.plot(combined.index, gt_36h, label="Ground truth (5-min)", linewidth=1.2, alpha=0.8)
    plt.plot(st_only_5min.index, st_only_5min.values, label="Pred ST-only (5-min)", linestyle="--", linewidth=1.0)
    if len(vstf_5min) > 0:
        plt.plot(vstf_5min.index, vstf_5min.values, label="Pred VSTF 1h (5-min)", linewidth=2.0)

    # ST hourly points
    plt.scatter(st_hourly.index, st_hourly.values, s=24, label="ST hourly points", zorder=5)

    # markers
    next_full_hour = (t_now + pd.Timedelta(minutes=5)).ceil("h")
    plt.axvline(t_now, color="k", linestyle="--", alpha=0.6, label="t_now")
    plt.axvline(next_full_hour, color="tab:orange", linestyle="--", alpha=0.6, label="next_full_hour")

    plt.title(f"Debug @ {t_now} → 36h horizon (PV; daylight-clamped)")
    plt.xlabel("Time")
    plt.ylabel("p_norm")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    out_png = os.path.join(save_dir, f"debug_{idx_plot:04d}_{t_now.strftime('%Y%m%d_%H%M')}.png")
    # plt.savefig(out_png, dpi=140)
    plt.show()
    # plt.close()

# ----------------------------
# Main simulation
# ----------------------------
def main():
    # Load test DF
    df_raw = pd.read_csv(TEST_CSV_PATH, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    if "p_norm" not in df_raw.columns:
        raise ValueError("CSV must contain column 'p_norm'.")

    # PV preprocessing: zero out at night BEFORE feature engineering
    df = enforce_pv_zero_outside_window(df_raw[["p_norm"]], DAY_START, DAY_END)

    # Sanity on normalization
    pmin, pmax = float(df["p_norm"].min()), float(df["p_norm"].max())
    if not (-0.05 <= pmin <= 1.05 and -0.05 <= pmax <= 1.05):
        print(f"[WARN] p_norm outside ~[0,1]: min={pmin:.3f}, max={pmax:.3f}. "
              f"Ensure same scaling as training.")

    # 5-min features (VSTF)
    df5 = df.copy()
    df5 = add_cyclical_features_5min(df5)

    # Hourly features (ST) — resample must match training
    if TRAIN_RESAMPLE_AGG == "mean":
        dfH = df[["p_norm"]].resample("h").mean(numeric_only=True)
    elif TRAIN_RESAMPLE_AGG == "median":
        dfH = df[["p_norm"]].resample("h").median(numeric_only=True)
    elif TRAIN_RESAMPLE_AGG == "sum":
        dfH = df[["p_norm"]].resample("h").sum(numeric_only=True)
    else:
        raise ValueError("Unsupported TRAIN_RESAMPLE_AGG.")
    dfH = dfH.dropna()
    dfH = add_cyclical_features_hourly(dfH)

    # Load models + configs (PV models)
    print("Loading PV models...")
    model_vstf = load_keras_safely(VSTF_MODEL_PATH)
    model_st   = load_keras_safely(ST_MODEL_PATH)
    vstf_cfg = get_model_cfg(VSTF_MODEL_PATH, VSTF_DEFAULT)
    st_cfg   = get_model_cfg(ST_MODEL_PATH,   ST_DEFAULT)
    print("VSTF cfg:", vstf_cfg)
    print("ST   cfg:", st_cfg)

    # Simulation range
    start_t = INITIAL_QUERY_TIME
    last_t  = df5.index.max()
    stop_t  = min(last_t - pd.Timedelta(hours=TOTAL_HOURS), start_t + pd.Timedelta(days=STOP_AFTER_DAYS))
    # Ensure VSTF history
    start_t = max(start_t, df5.index.min() + pd.Timedelta(minutes=5 * vstf_cfg["lookback_steps"]))
    if stop_t <= start_t:
        raise ValueError("Insufficient test span for the requested simulation window.")

    # Per-query metrics
    rows = []
    first_saved = False
    st_cache_raw: Dict[pd.Timestamp, pd.Series] = {}
    plot_counter = 0

    cur = start_t
    step = pd.Timedelta(minutes=5)
    print(f"Simulating from {start_t} to {stop_t} every 5 minutes...")
    while cur <= stop_t:
        try:
            t0 = time.perf_counter()
            combined, vstf_5min, st_hourly, st_only_5min = forecast_combined_36h_5min(
                t_now=cur, df5=df5, dfH=dfH,
                model_vstf=model_vstf, model_st=model_st,
                vstf_cfg=vstf_cfg, st_cfg=st_cfg,
                st_cache=st_cache_raw,
                total_hours=TOTAL_HOURS, vstf_hours=VSTF_HOURS
            )
            runtime_s = time.perf_counter() - t0
        except Exception:
            # often the very first steps lack enough history; skip
            cur += step
            continue

        # Ground truth slices (already daylight-clamped from preprocessing)
        horizon_idx = pd.date_range(start=cur + pd.Timedelta(minutes=5),
                                    periods=TOTAL_HOURS*12, freq="5min")
        gt_36h = df5["p_norm"].reindex(horizon_idx).values.astype(np.float32)
        gt_1h  = gt_36h[:12]

        # Metrics
        y_comb = combined.values.astype(np.float32)
        y_vstf = vstf_5min.values.astype(np.float32)
        y_st36 = st_only_5min.values.astype(np.float32)

        rows.append({
            "query_time": cur,
            "runtime_s": runtime_s,
            "combined_mse": mse(gt_36h, y_comb),
            "combined_rmse": rmse(gt_36h, y_comb),
            "combined_r2": r2_score(gt_36h, y_comb),
            "vstf1h_mse": mse(gt_1h, y_vstf),
            "vstf1h_rmse": rmse(gt_1h, y_vstf),
            "vstf1h_r2": r2_score(gt_1h, y_vstf),
            "st36h_mse": mse(gt_36h, y_st36),
            "st36h_rmse": rmse(gt_36h, y_st36),
            "st36h_r2": r2_score(gt_36h, y_st36),
        })

        # Save one example CSV
        if not first_saved:
            ex_df = pd.DataFrame({
                "timestamp": combined.index,
                "pred_combined": y_comb,
                "pred_vstf_first_hour": np.r_[y_vstf, [np.nan] * (len(y_comb) - len(y_vstf))],
                "pred_st_only_36h": y_st36,
                "ground_truth": gt_36h,
            })
            ex_path = os.path.join(SAVE_DIR, "example_forecast_36h_5min_pv.csv")
            ex_df.to_csv(ex_path, index=False)
            print(f"Saved example forecast to {ex_path}")
            first_saved = True

        # Optional debug plot
        if DEBUG_PLOT_EVERY and (plot_counter < DEBUG_MAX_PLOTS) and ((len(rows)-1) % DEBUG_PLOT_EVERY == 0):
            plot_debug(cur, df5, combined, vstf_5min, st_only_5min, st_hourly, gt_36h, SAVE_DIR, plot_counter)
            plot_counter += 1

        cur += step

    # Aggregate and save results
    res = pd.DataFrame(rows)
    out_csv = os.path.join(SAVE_DIR, "realtime_eval_metrics_pv.csv")
    res.to_csv(out_csv, index=False)
    print(f"\nSaved per-query metrics to {out_csv}")

    # Summary
    def summarize(prefix: str):
        sub = res[[f"{prefix}_mse", f"{prefix}_rmse", f"{prefix}_r2"]]
        return sub.mean().to_dict()

    print("\nAverage metrics across all queries (PV, daylight-clamped):")
    print("Combined 36h:", summarize("combined"))
    print("VSTF 1h    :", summarize("vstf1h"))
    print("ST 36h     :", summarize("st36h"))

if __name__ == "__main__":
    main()
