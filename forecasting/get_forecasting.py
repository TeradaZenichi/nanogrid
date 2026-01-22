# -*- coding: utf-8 -*-
# File: forecasting/get_forecasting.py

import os
from typing import Dict, Any, Optional, Iterable, Union
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from opt.utils import slice_forecasts, build_time_grid
import forecasting


DEFAULT_PATHS: Dict[str, str] = {
    "LOAD_VSTF_PATH": "models/lstm_vstf_load_5min_240min_96.keras",
    "LOAD_ST_PATH":   "models/lstm_hourly_load_36h_8.keras",
    "PV_VSTF_PATH":   "models/lstm_vstf_pv_5min_60min_48.keras",
    "PV_ST_PATH":     "models/lstm_hourly_pv_36h_8.keras",
}

DEFAULT_CFGS: Dict[str, Any] = {
    "VSTF_LOAD_CFG": {"step_minutes": 5,  "lookback_steps": 96, "horizon_steps": 48},
    "ST_LOAD_CFG":   {"step_minutes": 60, "lookback_steps": 8,  "horizon_steps": 36},
    "VSTF_PV_CFG":   {"step_minutes": 5,  "lookback_steps": 48, "horizon_steps": 12},
    "ST_PV_CFG":     {"step_minutes": 60, "lookback_steps": 8,  "horizon_steps": 36},
}

DEFAULT_MISC: Dict[str, Any] = {
    "day_start": "06:00",
    "day_end": "19:00",
    "clip_to_unit": False,
    "default_intervals": 432,  # 36h @ 5min
    "default_dt_min": 5,
}


def _load_keras(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Keras model not found: {path}")
    try:
        return load_model(path, compile=False, safe_mode=False)
    except TypeError:
        return load_model(path, compile=False)


def _is_iterable_ints(x: Any) -> bool:
    if isinstance(x, (list, tuple, np.ndarray)):
        try:
            _ = [int(v) for v in x]
            return True
        except Exception:
            return False
    return False


def _build_variable_grid(start: pd.Timestamp, dt_seq: Iterable[int]) -> pd.DatetimeIndex:
    offsets = np.concatenate(([0], np.cumsum(list(dt_seq)[:-1])))
    return pd.DatetimeIndex([start + pd.Timedelta(minutes=int(off)) for off in offsets])


def _aggregate_to_intervals(s_5min: pd.Series,
                            starts: pd.DatetimeIndex,
                            dt_seq: Iterable[int],
                            how: str = "mean") -> np.ndarray:
    out = []
    dt_seq = [int(x) for x in dt_seq]
    for i, t0 in enumerate(starts):
        t1 = t0 + pd.Timedelta(minutes=dt_seq[i])
        window = s_5min.loc[(s_5min.index >= t0) & (s_5min.index < t1)]
        if len(window) == 0:
            j = s_5min.index.get_indexer([t0], method="nearest")[0]
            val = s_5min.iloc[int(j)]
        else:
            val = window.mean() if how == "mean" else window.iloc[-1]
        out.append(float(val))
    return np.asarray(out, dtype=float)


def _as_series(x: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """Return a numeric Series for actuals from a Series or DataFrame."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        for pref in ("p_norm", "value", "kw", "power"):
            if pref in x.columns and pd.api.types.is_numeric_dtype(x[pref]):
                return x[pref]
        for c in x.columns:
            if pd.api.types.is_numeric_dtype(x[c]):
                return x[c]
    raise ValueError("Could not extract a numeric Series from provided actuals.")


def get_window(start_dt0: pd.Timestamp,
               load_kw_s: pd.DataFrame,
               pv_kw_s: pd.DataFrame,
               dt_min: Union[int, Iterable[int]]):
    if _is_iterable_ints(dt_min):
        times = _build_variable_grid(pd.Timestamp(start_dt0), [int(v) for v in dt_min])
    else:
        times = build_time_grid(pd.Timestamp(start_dt0), int(dt_min))
    if any(t not in load_kw_s.index for t in times) or any(t not in pv_kw_s.index for t in times):
        print("End of data — get_window() returns None.")
        return None
    return slice_forecasts(times, load_kw_s, pv_kw_s)


class ForecastMPC:
    def __init__(self,
                 params: Optional[Dict[str, Any]],
                 load_kw_s: Union[pd.Series, pd.DataFrame],
                 pv_kw_s: Union[pd.Series, pd.DataFrame],
                 pv_scaling: float,
                 load_scaling: float) -> None:
        self.params: Dict[str, Any] = {**DEFAULT_PATHS, **DEFAULT_CFGS, **DEFAULT_MISC, **(params or {})}
        self.load_s = _as_series(load_kw_s)
        self.pv_s   = _as_series(pv_kw_s)
        self.pv_scaling = float(pv_scaling)
        self.load_scaling = float(load_scaling)

        self.m_vstf_load = _load_keras(self.params["LOAD_VSTF_PATH"])
        self.m_st_load   = _load_keras(self.params["LOAD_ST_PATH"])
        self.m_vstf_pv   = _load_keras(self.params["PV_VSTF_PATH"])
        self.m_st_pv     = _load_keras(self.params["PV_ST_PATH"])

        self.vstf_load_cfg = dict(self.params.get("VSTF_LOAD_CFG", {}))
        self.st_load_cfg   = dict(self.params.get("ST_LOAD_CFG", {}))
        self.vstf_pv_cfg   = dict(self.params.get("VSTF_PV_CFG", {}))
        self.st_pv_cfg     = dict(self.params.get("ST_PV_CFG", {}))

        self.day_start = self.params["day_start"]
        self.day_end = self.params["day_end"]
        self.clip_to_unit = bool(self.params["clip_to_unit"])
        self.default_intervals = int(self.params["default_intervals"])  # 432
        self.default_dt_min = int(self.params["default_dt_min"])        # 5

    def _infer_intervals(self, dt_min: int) -> int:
        if dt_min == int(self.vstf_load_cfg.get("step_minutes", 5)):
            hs = self.vstf_load_cfg.get("horizon_steps")
            if isinstance(hs, int) and hs > 0:
                return hs
        if dt_min == int(self.st_load_cfg.get("step_minutes", 60)):
            hs = self.st_load_cfg.get("horizon_steps")
            if isinstance(hs, int) and hs > 0:
                return hs
        return self.default_intervals

    def get_forecasts(self,
                      start_dt0: Union[pd.Timestamp, str],
                      intervals: Optional[int] = None,
                      dt_min: Optional[Union[int, Iterable[int]]] = None,
                      include_actuals: bool = False) -> Optional[Dict[str, Dict[pd.Timestamp, float]]]:
        start_dt0 = pd.Timestamp(start_dt0)

        # --- Variable-step: map from 5-min base to dt_seq intervals
        if _is_iterable_ints(dt_min):
            dt_seq = [int(v) for v in dt_min]
            total_min = int(np.sum(dt_seq))
            if total_min > self.default_intervals * 5:
                raise ValueError(f"dt_min total minutes ({total_min}) exceed {self.default_intervals * 5}.")
            times_var = _build_variable_grid(start_dt0, dt_seq)

            y_load_5 = np.asarray(forecasting.load(
                start_dt0, self.default_intervals,
                model_vstf=self.m_vstf_load, model_st=self.m_st_load,
                vstf_cfg=self.vstf_load_cfg, st_cfg=self.st_load_cfg,
            ), dtype=float) * self.load_scaling

            y_pv_5 = np.asarray(forecasting.pv(
                start_dt0, self.default_intervals,
                model_vstf=self.m_vstf_pv, model_st=self.m_st_pv,
                vstf_cfg=self.vstf_pv_cfg, st_cfg=self.st_pv_cfg,
                day_start=self.day_start, day_end=self.day_end,
                clip_to_unit=self.clip_to_unit,
            ), dtype=float) * self.pv_scaling

            # prepend actual at start_dt0 and drop last to keep length
            if start_dt0 not in self.load_s.index or start_dt0 not in self.pv_s.index:
                raise KeyError("start_dt0 must exist in actual series to anchor the first value.")
            y_load_5 = np.concatenate(([float(self.load_s.at[start_dt0])], y_load_5[:-1]))
            y_pv_5   = np.concatenate(([float(self.pv_s.at[start_dt0])],   y_pv_5[:-1]))

            times_5 = pd.date_range(start=start_dt0, periods=self.default_intervals, freq="5min")
            s_load_5 = pd.Series(y_load_5, index=times_5)
            s_pv_5   = pd.Series(y_pv_5,   index=times_5)

            y_load = _aggregate_to_intervals(s_load_5, times_var, dt_seq, how="mean")
            y_pv   = _aggregate_to_intervals(s_pv_5,   times_var, dt_seq, how="mean")

            return {
                "load_kw": dict(zip(times_var, y_load.tolist())),
                "pv_kw":   dict(zip(times_var, y_pv.tolist())),
            }

        # --- Uniform-step: direct grid
        step = int(dt_min) if dt_min is not None else self.default_dt_min
        steps_out = int(intervals) if intervals is not None else self._infer_intervals(step)
        times = pd.date_range(start=start_dt0, periods=steps_out, freq=f"{step}min")

        y_load_fc = np.asarray(forecasting.load(
            start_dt0, steps_out,
            model_vstf=self.m_vstf_load, model_st=self.m_st_load,
            vstf_cfg=self.vstf_load_cfg, st_cfg=self.st_load_cfg,
        ), dtype=float) * self.load_scaling

        y_pv_fc = np.asarray(forecasting.pv(
            start_dt0, steps_out,
            model_vstf=self.m_vstf_pv, model_st=self.m_st_pv,
            vstf_cfg=self.vstf_pv_cfg, st_cfg=self.st_pv_cfg,
            day_start=self.day_start, day_end=self.day_end,
            clip_to_unit=self.clip_to_unit,
        ), dtype=float) * self.pv_scaling

        if start_dt0 not in self.load_s.index or start_dt0 not in self.pv_s.index:
            raise KeyError("start_dt0 must exist in actual series to anchor the first value.")

        y_load_out = np.empty(steps_out, dtype=float)
        y_pv_out   = np.empty(steps_out, dtype=float)
        y_load_out[0] = float(self.load_s.at[start_dt0])
        y_pv_out[0]   = float(self.pv_s.at[start_dt0])
        if steps_out > 1:
            y_load_out[1:] = y_load_fc[:-1]
            y_pv_out[1:]   = y_pv_fc[:-1]

        return {
            "load_kw": dict(zip(times, y_load_out.tolist())),
            "pv_kw":   dict(zip(times, y_pv_out.tolist())),
        }
