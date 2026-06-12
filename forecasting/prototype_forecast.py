# -*- coding: utf-8 -*-
# File: forecasting/prototype_forecast.py
"""
Analog-day (cluster prototype) forecaster for the MPC pipeline.

Uses the DTW cluster prototypes produced by forecasting/clustering (48 slots
of 30 min, normalized [0,1]) as the long-duration forecast. The current day's
cluster is selected by one of three strategies:

- "calendar": most probable cluster from the train marginals (works at 00:00).
- "prefix":   classify the observed prefix of the current day against the
              prototypes truncated to the same slots (Euclidean distance).
- "knn":      same prefix distance, but against the real train days; the
              forecast is the remainder of the nearest day (k=1).

Days beyond the current one (the 36h horizon crosses midnight) always use the
"calendar" cluster, since no prefix exists for the future.

Output format is identical to ForecastMPC.get_forecasts(); intentionally free
of any TensorFlow dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd

SLOTS_PER_DAY = 48
SLOT_MIN = 30

DEFAULT_DATA = {
    "LOAD_PROTOTYPES": "data/sizing/prototypes_load_dtw_all_train.csv",
    "PV_PROTOTYPES": "data/sizing/prototypes_pv_dtw_train.csv",
    "PROB_LOAD": "data/sizing/prob_load.csv",
    "PROB_PV": "data/sizing/prob_pv.csv",
    "LOAD_TRAIN_CSV": "data/load_5min_train.csv",
    "PV_TRAIN_CSV": "data/pv_5min_train.csv",
}

DEFAULT_MISC = {
    "day_start_hour": 6.0,   # PV daylight window for prefix classification
    "day_end_hour": 19.0,
    "min_prefix_slots": 4,   # fall back to "calendar" before 2h of observations
    "blend_steps": 6,        # 5-min steps blended from the anchor observation
    "default_intervals": 432,  # 36h @ 5 min
    "default_dt_min": 5,
}

VALID_STRATEGIES = ("calendar", "prefix", "knn")


def _as_series(x: Union[pd.Series, pd.DataFrame]) -> pd.Series:
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
                            dt_seq: Iterable[int]) -> np.ndarray:
    out = []
    dt_seq = [int(x) for x in dt_seq]
    for i, t0 in enumerate(starts):
        t1 = t0 + pd.Timedelta(minutes=dt_seq[i])
        window = s_5min.loc[(s_5min.index >= t0) & (s_5min.index < t1)]
        if len(window) == 0:
            j = s_5min.index.get_indexer([t0], method="nearest")[0]
            val = s_5min.iloc[int(j)]
        else:
            val = window.mean()
        out.append(float(val))
    return np.asarray(out, dtype=float)


def _load_prototypes(path: Path) -> Dict[int, np.ndarray]:
    df = pd.read_csv(path)
    protos: Dict[int, np.ndarray] = {}
    for cl, g in df.groupby("cluster"):
        arr = g.sort_values("slot")["value"].to_numpy(dtype=float)
        if len(arr) != SLOTS_PER_DAY:
            raise ValueError(f"Prototype cluster {cl} in {path} has {len(arr)} slots, expected {SLOTS_PER_DAY}.")
        protos[int(cl)] = arr
    if not protos:
        raise ValueError(f"No prototypes found in {path}.")
    return protos


def _most_probable_cluster(path: Path, cluster_col: str) -> int:
    df = pd.read_csv(path)
    row = df.loc[df["probability"].idxmax()]
    return int(row[cluster_col])


def _daily_matrix_30min(csv_path: Path) -> pd.DataFrame:
    """Train days as rows x 48 normalized 30-min slots (incomplete days dropped)."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    s = df.set_index("timestamp")["p_norm"].astype(float)
    s30 = s.resample(f"{SLOT_MIN}min").mean()
    mat = pd.DataFrame({
        "date": s30.index.date,
        "slot": (s30.index.hour * 60 + s30.index.minute) // SLOT_MIN,
        "value": s30.values,
    }).pivot_table(index="date", columns="slot", values="value")
    mat = mat.dropna()
    mat = mat.loc[:, sorted(mat.columns)]
    if mat.shape[1] != SLOTS_PER_DAY:
        raise ValueError(f"Train matrix from {csv_path} has {mat.shape[1]} slots, expected {SLOTS_PER_DAY}.")
    return mat


class PerfectForecast:
    """Oracle forecaster: returns the actual series aggregated to the MPC grid.

    Same get_forecasts() interface as the other forecasters; used as the
    perfect-information upper bound (former "ideal" pipeline).
    """

    def __init__(self,
                 load_kw_s: Union[pd.Series, pd.DataFrame],
                 pv_kw_s: Union[pd.Series, pd.DataFrame]) -> None:
        self.load_s = _as_series(load_kw_s)
        self.pv_s = _as_series(pv_kw_s)

    def get_forecasts(self,
                      start_dt0: Union[pd.Timestamp, str],
                      intervals: Optional[int] = None,
                      dt_min: Optional[Union[int, Iterable[int]]] = None,
                      include_actuals: bool = False) -> Optional[Dict[str, Dict[pd.Timestamp, float]]]:
        start_dt0 = pd.Timestamp(start_dt0)
        if _is_iterable_ints(dt_min):
            dt_seq = [int(v) for v in dt_min]
        else:
            step = int(dt_min) if dt_min is not None else 5
            steps_out = int(intervals) if intervals is not None else 432
            dt_seq = [step] * steps_out
        times = _build_variable_grid(start_dt0, dt_seq)
        horizon_end = times[-1] + pd.Timedelta(minutes=dt_seq[-1])
        if times[0] not in self.load_s.index or horizon_end - pd.Timedelta(minutes=5) > self.load_s.index.max():
            return None  # window leaves the data range
        y_load = _aggregate_to_intervals(self.load_s, times, dt_seq)
        y_pv = _aggregate_to_intervals(self.pv_s, times, dt_seq)
        return {
            "load_kw": dict(zip(times, y_load.tolist())),
            "pv_kw": dict(zip(times, y_pv.tolist())),
        }


class HybridForecast:
    """Composition: one forecaster on the fine (dt1) steps, another on the coarse steps.

    Typical use: HybridForecast(fine=ForecastMPC(...), coarse=PrototypeForecast(...)).
    """

    def __init__(self, fine, coarse) -> None:
        self.fine = fine
        self.coarse = coarse

    def get_forecasts(self, start_dt0, intervals=None, dt_min=None, include_actuals=False):
        out_fine = self.fine.get_forecasts(start_dt0, intervals, dt_min, include_actuals)
        out_coarse = self.coarse.get_forecasts(start_dt0, intervals, dt_min, include_actuals)
        if out_fine is None or out_coarse is None:
            return None
        if not _is_iterable_ints(dt_min):
            return out_fine
        dt_seq = [int(v) for v in dt_min]
        dt_fine = dt_seq[0]
        out: Dict[str, Dict[pd.Timestamp, float]] = {"load_kw": {}, "pv_kw": {}}
        for key in out:
            for (ts, v_fine), v_coarse, dt in zip(
                out_fine[key].items(), out_coarse[key].values(), dt_seq
            ):
                out[key][ts] = v_fine if dt == dt_fine else v_coarse
        return out


class PrototypeForecast:
    """Analog-day forecaster with the same get_forecasts() interface as ForecastMPC."""

    def __init__(self,
                 params: Optional[Dict[str, Any]],
                 load_kw_s: Union[pd.Series, pd.DataFrame],
                 pv_kw_s: Union[pd.Series, pd.DataFrame],
                 pv_scaling: float,
                 load_scaling: float,
                 strategy: str = "prefix") -> None:
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {VALID_STRATEGIES}, got '{strategy}'.")
        self.params: Dict[str, Any] = {**DEFAULT_DATA, **DEFAULT_MISC, **(params or {})}
        self.strategy = strategy
        self.load_s = _as_series(load_kw_s)
        self.pv_s = _as_series(pv_kw_s)
        self.pv_scaling = float(pv_scaling)
        self.load_scaling = float(load_scaling)

        self.proto_load = _load_prototypes(Path(self.params["LOAD_PROTOTYPES"]))
        self.proto_pv = _load_prototypes(Path(self.params["PV_PROTOTYPES"]))
        self.calendar_load = _most_probable_cluster(Path(self.params["PROB_LOAD"]), "cluster_load")
        self.calendar_pv = _most_probable_cluster(Path(self.params["PROB_PV"]), "cluster_pv")

        self.min_prefix_slots = int(self.params["min_prefix_slots"])
        self.blend_steps = int(self.params["blend_steps"])
        self.default_intervals = int(self.params["default_intervals"])
        self.default_dt_min = int(self.params["default_dt_min"])

        h0, h1 = float(self.params["day_start_hour"]), float(self.params["day_end_hour"])
        slot_hours = np.arange(SLOTS_PER_DAY) * (SLOT_MIN / 60.0)
        self._pv_daylight_slots = (slot_hours >= h0) & (slot_hours < h1)

        # Lazy: train day matrices are only loaded for the "knn" strategy.
        self._train_load: Optional[pd.DataFrame] = None
        self._train_pv: Optional[pd.DataFrame] = None
        if self.strategy == "knn":
            self._train_load = _daily_matrix_30min(Path(self.params["LOAD_TRAIN_CSV"]))
            self._train_pv = _daily_matrix_30min(Path(self.params["PV_TRAIN_CSV"]))

    # ------------------------------------------------------------------
    # Prefix extraction and classification
    # ------------------------------------------------------------------
    def _observed_prefix_norm(self, kind: str, now: pd.Timestamp) -> np.ndarray:
        """Normalized 30-min means of the current day in [00:00, now); NaN-padded to 48."""
        s = self.load_s if kind == "load" else self.pv_s
        scale = self.load_scaling if kind == "load" else self.pv_scaling
        day0 = now.normalize()
        window = s.loc[(s.index >= day0) & (s.index < now)]
        out = np.full(SLOTS_PER_DAY, np.nan)
        if window.empty or scale <= 0:
            return out
        g = window.groupby((window.index.hour * 60 + window.index.minute) // SLOT_MIN).mean()
        # Only slots fully elapsed before `now` count as observed.
        n_complete = int((now - day0).total_seconds() // (SLOT_MIN * 60))
        for slot, val in g.items():
            if int(slot) < n_complete:
                out[int(slot)] = float(val) / scale
        return out

    def _classify(self, kind: str, now: pd.Timestamp) -> int:
        calendar = self.calendar_load if kind == "load" else self.calendar_pv
        if self.strategy == "calendar":
            return calendar
        prefix = self._observed_prefix_norm(kind, now)
        mask = ~np.isnan(prefix)
        if kind == "pv":
            mask &= self._pv_daylight_slots
        if mask.sum() < self.min_prefix_slots:
            return calendar
        protos = self.proto_load if kind == "load" else self.proto_pv
        dists = {cl: float(np.linalg.norm(prefix[mask] - arr[mask])) for cl, arr in protos.items()}
        return min(dists, key=dists.get)

    def _knn_profile(self, kind: str, now: pd.Timestamp) -> Optional[np.ndarray]:
        """Nearest train day (k=1) by observed-prefix distance; None if no usable prefix."""
        mat = self._train_load if kind == "load" else self._train_pv
        prefix = self._observed_prefix_norm(kind, now)
        mask = ~np.isnan(prefix)
        if kind == "pv":
            mask &= self._pv_daylight_slots
        if mat is None or mask.sum() < self.min_prefix_slots:
            return None
        d = np.linalg.norm(mat.values[:, mask] - prefix[mask], axis=1)
        return mat.values[int(np.argmin(d))].astype(float)

    # ------------------------------------------------------------------
    # Profile assembly
    # ------------------------------------------------------------------
    def _day_profile_norm(self, kind: str, day: pd.Timestamp, now: pd.Timestamp) -> np.ndarray:
        """Normalized 48-slot profile for a given calendar day within the horizon."""
        protos = self.proto_load if kind == "load" else self.proto_pv
        calendar = self.calendar_load if kind == "load" else self.calendar_pv
        if day.normalize() != now.normalize():
            return protos[calendar]  # future days: no prefix exists
        if self.strategy == "knn":
            profile = self._knn_profile(kind, now)
            if profile is not None:
                return profile
            return protos[calendar]
        return protos[self._classify(kind, now)]

    def _series_5min(self, kind: str, start: pd.Timestamp, total_min: int) -> pd.Series:
        scale = self.load_scaling if kind == "load" else self.pv_scaling
        times = pd.date_range(start=start, periods=int(np.ceil(total_min / 5)), freq="5min")
        days = sorted({t.normalize() for t in times})
        profiles = {d: self._day_profile_norm(kind, d, start) for d in days}
        slots = (times.hour * 60 + times.minute) // SLOT_MIN
        vals = np.array(
            [profiles[t.normalize()][int(sl)] for t, sl in zip(times, slots)],
            dtype=float,
        ) * scale
        return pd.Series(vals, index=times)

    def _anchor_and_blend(self, y: pd.Series, actual0: float) -> pd.Series:
        y = y.copy()
        y.iloc[0] = actual0
        n = min(self.blend_steps, len(y) - 1)
        for i in range(1, n + 1):
            w = 1.0 - i / (n + 1)
            y.iloc[i] = w * actual0 + (1.0 - w) * y.iloc[i]
        return y

    # ------------------------------------------------------------------
    # Public interface (mirrors ForecastMPC.get_forecasts)
    # ------------------------------------------------------------------
    def get_forecasts(self,
                      start_dt0: Union[pd.Timestamp, str],
                      intervals: Optional[int] = None,
                      dt_min: Optional[Union[int, Iterable[int]]] = None,
                      include_actuals: bool = False) -> Optional[Dict[str, Dict[pd.Timestamp, float]]]:
        start_dt0 = pd.Timestamp(start_dt0)
        if start_dt0 not in self.load_s.index or start_dt0 not in self.pv_s.index:
            raise KeyError("start_dt0 must exist in actual series to anchor the first value.")
        load0 = float(self.load_s.at[start_dt0])
        pv0 = float(self.pv_s.at[start_dt0])

        if _is_iterable_ints(dt_min):
            dt_seq = [int(v) for v in dt_min]
            total_min = int(np.sum(dt_seq))
            times_var = _build_variable_grid(start_dt0, dt_seq)

            s_load = self._anchor_and_blend(self._series_5min("load", start_dt0, total_min), load0)
            s_pv = self._anchor_and_blend(self._series_5min("pv", start_dt0, total_min), pv0)

            y_load = _aggregate_to_intervals(s_load, times_var, dt_seq)
            y_pv = _aggregate_to_intervals(s_pv, times_var, dt_seq)
            return {
                "load_kw": dict(zip(times_var, y_load.tolist())),
                "pv_kw": dict(zip(times_var, y_pv.tolist())),
            }

        step = int(dt_min) if dt_min is not None else self.default_dt_min
        steps_out = int(intervals) if intervals is not None else self.default_intervals
        times = pd.date_range(start=start_dt0, periods=steps_out, freq=f"{step}min")

        s_load = self._series_5min("load", start_dt0, steps_out * step)
        s_pv = self._series_5min("pv", start_dt0, steps_out * step)
        y_load = _aggregate_to_intervals(s_load, times, [step] * steps_out)
        y_pv = _aggregate_to_intervals(s_pv, times, [step] * steps_out)
        y_load[0], y_pv[0] = load0, pv0
        n = min(self.blend_steps, steps_out - 1)
        for i in range(1, n + 1):
            w = 1.0 - i / (n + 1)
            y_load[i] = w * load0 + (1.0 - w) * y_load[i]
            y_pv[i] = w * pv0 + (1.0 - w) * y_pv[i]

        return {
            "load_kw": dict(zip(times, y_load.tolist())),
            "pv_kw": dict(zip(times, y_pv.tolist())),
        }
