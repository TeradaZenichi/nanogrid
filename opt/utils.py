# -*- coding: utf-8 -*-
"""
opt/utils.py
Utilitários reutilizáveis para MPCs (Off-Grid e futuro On-Grid).

Inclui:
- construção de grade temporal e pares predecessores
- normalização/checagem de colunas
- leitura e escala de séries (PV e Carga)
- montagem de previsões alinhadas à grade temporal
- helpers de captura/saving de resultados
"""
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

import pandas as pd

# ---------------------------
# Tempo e grade temporal
# ---------------------------

def build_dt_vector(horizon_hours: int, fine_hours: int, dt1_min: int, dt2_min: int) -> List[int]:
    if horizon_hours <= 0:
        raise ValueError("horizon_hours deve ser > 0")
    if not (0 <= fine_hours <= horizon_hours):
        raise ValueError("fine_hours deve estar em [0, horizon_hours]")
    if dt1_min <= 0 or dt2_min <= 0:
        raise ValueError("timestep_1_min e timestep_2_min devem ser > 0 (min)")
    steps_fine = (fine_hours * 60) // dt1_min
    if steps_fine * dt1_min != fine_hours * 60:
        raise ValueError("fine_hours * 60 deve ser múltiplo exato de timestep_1_min")
    steps_coarse = ((horizon_hours - fine_hours) * 60) // dt2_min
    if steps_coarse * dt2_min != (horizon_hours - fine_hours) * 60:
        raise ValueError("(horizon_hours - fine_hours) * 60 deve ser múltiplo exato de timestep_2_min")
    dt_min = [dt1_min] * steps_fine + [dt2_min] * steps_coarse
    if len(dt_min) < 2:
        raise ValueError("O horizonte precisa ter pelo menos 2 passos.")
    return dt_min

def build_time_grid(start_dt: datetime, dt_min: List[int]) -> List[datetime]:
    times = [start_dt]
    for dm in dt_min[:-1]:
        times.append(times[-1] + timedelta(minutes=dm))
    return times

def predecessor_pairs(times: List[datetime]) -> List[Tuple[datetime, datetime]]:
    return list(zip(times[:-1], times[1:]))

# ---------------------------
# Normalização / checagem
# ---------------------------

def pnorm_columns(df: pd.DataFrame) -> pd.DataFrame:
    def _norm(name: str) -> str:
        s = name.strip().lower()
        s = re.sub(r'[^a-z0-9]+', '_', s)
        s = re.sub(r'_{2,}', '_', s).strip('_')
        return s
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    return df

def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"CSV precisa conter uma das colunas: {candidates}")

# ---------------------------
# Séries e previsões
# ---------------------------

def load_series_scaled(params: Dict[str, Any],
                       load_csv: str,
                       pv_csv: str,
                       col_time: str = "timestamp",
                       col_pu: str = "p_norm") -> Tuple[pd.Series, pd.Series]:
    load_df = pd.read_csv(load_csv)
    pv_df   = pd.read_csv(pv_csv)
    for name, df in [("load", load_df), ("pv", pv_df)]:
        if col_time not in df.columns or col_pu not in df.columns:
            raise ValueError(f"CSV de {name} deve conter colunas '{col_time}' e '{col_pu}'.")
    load_df[col_time] = pd.to_datetime(load_df[col_time])
    pv_df[col_time]   = pd.to_datetime(pv_df[col_time])
    load_df.set_index(col_time, inplace=True)
    pv_df.set_index(col_time, inplace=True)
    load_df.sort_index(inplace=True)
    pv_df.sort_index(inplace=True)
    load_pu = load_df[col_pu].astype(float).clip(0.0, 1.0)
    pv_pu   = pv_df[col_pu].astype(float).clip(0.0, 1.0)
    P_L_max  = float(params["P_L_nom_kw"])
    P_PV_max = float(params["P_PV_nom_kw"])
    load_kw = load_pu * P_L_max
    pv_kw   = pv_pu * P_PV_max
    return load_kw, pv_kw

def slice_forecasts(times: List[datetime],
                    load_series_kw: pd.Series,
                    pv_series_kw: pd.Series) -> Dict[str, Dict[datetime, float]]:
    missing_load = [t for t in times if t not in load_series_kw.index]
    missing_pv   = [t for t in times if t not in pv_series_kw.index]
    if missing_load:
        raise KeyError(f"Faltam valores de carga para timestamps: {missing_load[:3]}...")
    if missing_pv:
        raise KeyError(f"Faltam valores de PV para timestamps: {missing_pv[:3]}...")
    fc_load = {t: float(load_series_kw.loc[t]) for t in times}
    fc_pv   = {t: float(pv_series_kw.loc[t])   for t in times}
    return {"load_kw": fc_load, "pv_kw": fc_pv}

# ---------------------------
# Captura / saving de resultados
# ---------------------------

def _val(v, default=0.0):
    try:
        from pyomo.environ import value
        x = value(v)
        return float(x if x is not None else default)
    except Exception:
        try:
            return float(v)
        except Exception:
            return float(default)

def horizon_snapshot(model, times):
    ts = list(times)
    out = {
        "timestamps": [t.isoformat() for t in ts],
        "dt_h":       [_val(model.dt_h[t]) for t in ts],
        "Load_kw":    [_val(model.Load_kw[t]) for t in ts],
        "PV_kw":      [_val(model.PV_kw[t]) for t in ts],
        "X_L":        [_val(model.X_L[t]) for t in ts],
        "X_PV":       [_val(model.X_PV[t]) for t in ts],
        "P_bess_kw":  [_val(model.P_bess[t]) for t in ts],
        "P_ch_kw":    [_val(model.P_ch[t]) for t in ts],
        "P_dis_kw":   [_val(model.P_dis[t]) for t in ts],
        "gamma":      [int(round(_val(model.gamma[t], 0))) for t in ts],
        "E_kwh":      [_val(model.E[t]) for t in ts],
    }
    if getattr(model, "P_EDS", None) is not None:
        out["EDS_kw"] = [_val(model.P_EDS[t]) for t in ts]
    return out

def save_log(results_log, path="outputs/dispatch_log.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import json as _json
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(results_log, f, ensure_ascii=False, indent=2)
    print(f"[log] salvo em {path}")
