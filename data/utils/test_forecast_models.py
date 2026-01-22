# -*- coding: utf-8 -*-
"""
Single-run loader: loads models once, then calls forecast.load/pv.
-----------------------------------------------------------------

- Configure model paths here.
- Provide model configs (lookback and horizons) explicitly.
- Call the API with a date from 2009 and 180 intervals (default).

Note: Ensure your CSVs exist at:
- data/load_5min_test.csv  (columns: timestamp,p_norm)
- data/pv_5min_test.csv    (columns: timestamp,p_norm)
"""
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


import forecasting  # local package: folder 'forecast' next to this file

# --------------------
# Model paths (adjust)
# --------------------
LOAD_VSTF_PATH = "models/lstm_vstf_load_5min_240min_96.keras"   # example: 48*5min horizon, 96 lookback
LOAD_ST_PATH   = "models/lstm_hourly_load_36h_8.keras"           # example: 36h horizon, 8 lookback

PV_VSTF_PATH   = "models/lstm_vstf_pv_5min_60min_48.keras"       # example: 12*5min horizon, 48 lookback
PV_ST_PATH     = "models/lstm_hourly_pv_36h_8.keras"              # example: 36h horizon, 8 lookback

# --------------------
# Model configs
# --------------------
# Keep these consistent with how you trained the models.
VSTF_LOAD_CFG = {"step_minutes": 5,  "lookback_steps": 96, "horizon_steps": 48}
ST_LOAD_CFG   = {"step_minutes": 60, "lookback_steps": 8,  "horizon_steps": 36}

VSTF_PV_CFG   = {"step_minutes": 5,  "lookback_steps": 48, "horizon_steps": 12}
ST_PV_CFG     = {"step_minutes": 60, "lookback_steps": 8,  "horizon_steps": 36}

def _load_keras(path: str):
    """Load a Keras model in a TF-version-tolerant way."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        return load_model(path, compile=False, safe_mode=False)
    except TypeError:
        return load_model(path, compile=False)

def main():
    # Load models ONCE
    print("Loading Keras models...")
    m_vstf_load = _load_keras(LOAD_VSTF_PATH)
    m_st_load   = _load_keras(LOAD_ST_PATH)
    m_vstf_pv   = _load_keras(PV_VSTF_PATH)
    m_st_pv     = _load_keras(PV_ST_PATH)
    print("Models loaded.")

    pv_df = pd.read_csv("data/pv_5min_test.csv", parse_dates=["timestamp"], index_col="timestamp")
    load_df = pd.read_csv("data/load_5min_test.csv", parse_dates=["timestamp"], index_col="timestamp")

    

    # Example call with a date from 2009
    dt = "2009-04-30 12:00:00"
    final = str(pd.Timestamp(dt) + pd.Timedelta(minutes=5*432))
    intervals = 432  # default (36h @ 5-min)

    # LOAD forecast (vector of shape (intervals,))
    y_load = forecasting.load(
        dt,
        intervals,
        model_vstf=m_vstf_load,
        model_st=m_st_load,
        vstf_cfg=VSTF_LOAD_CFG,
        st_cfg=ST_LOAD_CFG,
    )

    # PV forecast (vector of shape (intervals,))
    y_pv = forecasting.pv(
        dt,
        intervals,
        model_vstf=m_vstf_pv,
        model_st=m_st_pv,
        vstf_cfg=VSTF_PV_CFG,
        st_cfg=ST_PV_CFG,
        day_start="06:00",
        day_end="19:00",
        clip_to_unit=False,  # set True if your p_norm is in [0,1]
    )

    # Quick sanity prints
    np.set_printoptions(precision=4, suppress=True)
    print("[LOAD] shape:", y_load.shape, "head:", y_load[:10])
    print("[PV]   shape:", y_pv.shape,   "head:", y_pv[:10])
    
    
    plt.figure(figsize=(12,6))
    plt.plot(pv_df.loc[dt:final].index, pv_df.loc[dt:final].p_norm, label="PV Actual", color="blue")
    plt.plot(pd.date_range(start=dt, periods=intervals, freq="5min"), y_pv, label="PV Forecasting", color="cyan", linestyle="--")
    plt.plot(load_df.loc[dt:final].index, load_df.loc[dt:final].p_norm, label="Load Actual", color="red")
    plt.plot(pd.date_range(start=dt, periods=intervals, freq="5min"), y_load, label="Load Forecasting", color="orange", linestyle="--")
    plt.tight_layout()
    plt.xlabel("Timestamp")
    plt.ylabel("Normalized Power")
    plt.title("Forecast Example (5-min resolution)")
    plt.legend()
    plt.savefig("forecast_example.pdf", dpi=150)

    a = 1

if __name__ == "__main__":
    main()
