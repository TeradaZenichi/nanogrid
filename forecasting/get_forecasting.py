# Arquivo: get_forecasting.py

import pandas as pd
from typing import Optional, Dict, Any
from opt.utils import slice_forecasts, build_time_grid

def get_window(start_dt0: pd.Timestamp, load_kw_s: pd.DataFrame, pv_kw_s: pd.DataFrame, dt_min):
    times = build_time_grid(start_dt0, dt_min)
    
    if any(t not in load_kw_s.index for t in times) or any(t not in pv_kw_s.index for t in times):
        print("End of data — get_window() returns None.")
        return None
    
    forecasts = slice_forecasts(times, load_kw_s, pv_kw_s)
    return forecasts
