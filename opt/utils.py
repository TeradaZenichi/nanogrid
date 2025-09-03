# -*- coding: utf-8 -*-
"""
opt/utils.py
Utilitários reutilizáveis para MPCs (Off-Grid e On-Grid).

Inclui:
- construção de grade temporal e pares predecessores
- criação do conjunto de contingências (subconjunto de T)
- normalização/checagem de colunas
- leitura e escala de séries (PV e Carga)
- montagem de previsões alinhadas à grade temporal
- helpers de captura/saving de resultados
"""
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd





# ---------------------------
# Tempo, grade temporal e contingências
# ---------------------------

def build_dt_vector(horizon_hours: int,
                    outage_duration_hours: int,
                    dt1_min: int,
                    dt2_min: int) -> List[int]:
    """
    Cria vetor de passos (em minutos) para o horizonte:
      - Primeiras 'outage_duration_hours' horas com resolução fina (dt1_min).
      - Restante do horizonte com resolução grossa (dt2_min).

    Substitui completamente o antigo uso de 'fine_hours'.
    """
    if horizon_hours <= 0:
        raise ValueError("horizon_hours deve ser > 0")
    if not (0 <= outage_duration_hours <= horizon_hours):
        raise ValueError("outage_duration_hours deve estar em [0, horizon_hours]")
    if dt1_min <= 0 or dt2_min <= 0:
        raise ValueError("timestep_1_min e timestep_2_min devem ser > 0 (min)")

    # passos finos durante a janela de outage
    steps_fine = (outage_duration_hours * 60) // dt1_min
    if steps_fine * dt1_min != outage_duration_hours * 60:
        raise ValueError("outage_duration_hours * 60 deve ser múltiplo exato de timestep_1_min")

    # passos grossos no restante
    steps_coarse = ((horizon_hours - outage_duration_hours) * 60) // dt2_min
    if steps_coarse * dt2_min != (horizon_hours - outage_duration_hours) * 60:
        raise ValueError("(horizon_hours - outage_duration_hours) * 60 deve ser múltiplo exato de timestep_2_min")

    dt_min = [dt1_min] * steps_fine + [dt2_min] * steps_coarse
    if len(dt_min) < 2:
        raise ValueError("O horizonte precisa ter pelo menos 2 passos.")
    return dt_min


def build_time_grid(start_dt: datetime, dt_min: List[int]) -> List[datetime]:
    """
    Constrói a lista de timestamps T a partir de um instante inicial e do vetor de passos em minutos.
    Retorna uma lista de tamanho igual a len(dt_min), com o último índice representando o fim do horizonte.
    """
    times = [start_dt]
    for dm in dt_min[:-1]:
        times.append(times[-1] + timedelta(minutes=dm))
    return times


def predecessor_pairs(times: List[datetime]) -> List[Tuple[datetime, datetime]]:
    """Cria pares predecessores consecutivos (t0, t1) a partir de T."""
    return list(zip(times[:-1], times[1:]))


def build_contingency_times(start_dt: datetime,
                            horizon_hours: int,
                            outage_duration_hours: int,
                            dt1_min: int,
                            dt2_min: int) -> List[datetime]:
    """
    Gera os 'start times' c para cenários de contingência (outage) com a regra:
      - incluir metade dos passos finos (dt1) no começo,
      - pular a outra metade dos passos finos,
      - incluir todos os passos grossos (dt2) exceto o último.

    Observações:
      - Se outage_duration_hours = 0, não haverá parte fina; entram todos os passos grossos exceto o último.
      - Retorna uma lista ordenada e sem duplicatas.
    """
    if horizon_hours <= 0:
        raise ValueError("horizon_hours deve ser > 0")
    if not (0 <= outage_duration_hours <= horizon_hours):
        raise ValueError("outage_duration_hours deve estar em [0, horizon_hours]")
    if dt1_min <= 0 or dt2_min <= 0:
        raise ValueError("timestep_1_min e timestep_2_min devem ser > 0 (min)")

    # número de passos finos e grossos (mesmas divisibilidades de build_dt_vector)
    steps_fine = (outage_duration_hours * 60) // dt1_min
    if steps_fine * dt1_min != outage_duration_hours * 60:
        raise ValueError("outage_duration_hours * 60 deve ser múltiplo exato de timestep_1_min")

    steps_coarse = ((horizon_hours - outage_duration_hours) * 60) // dt2_min
    if steps_coarse * dt2_min != (horizon_hours - outage_duration_hours) * 60:
        raise ValueError("(horizon_hours - outage_duration_hours) * 60 deve ser múltiplo exato de timestep_2_min")

    # grade de tempo completa (mesma lógica do build_time_grid)
    dt_min = [dt1_min] * steps_fine + [dt2_min] * steps_coarse
    times: List[datetime] = [start_dt]
    for dm in dt_min[:-1]:
        times.append(times[-1] + timedelta(minutes=dm))

    # índices que entram no conjunto de contingências
    # metade dos finos (floor)
    half_fine = steps_fine // 2
    idx_fine_in = list(range(0, half_fine))  # inclui a 1ª metade
    # grossos: todos exceto o último
    idx_coarse_in = list(range(steps_fine, steps_fine + max(steps_coarse - 1, 0)))

    # resultado final (ordenado e sem repetição)
    cont_idx = sorted(set(idx_fine_in + idx_coarse_in))
    contingencies = [times[i] for i in cont_idx]
    return contingencies


def build_time_and_contingencies_from_params(params: Dict[str, Any],
                                             start_dt: datetime) -> Tuple[List[datetime], List[datetime]]:
    horizon_hours = int(params["horizon_hours"])
    dt1_min = int(params["timestep_1_min"])
    dt2_min = int(params["timestep_2_min"])
    outage_duration_hours = int(params.get("outage_duration_hours", 0))

    dt_min = build_dt_vector(horizon_hours, outage_duration_hours, dt1_min, dt2_min)
    times = build_time_grid(start_dt, dt_min)
    contingencies = build_contingency_times(start_dt, horizon_hours, outage_duration_hours, dt1_min, dt2_min)
    return times, contingencies


# ---------------------------
# Normalização / checagem
# ---------------------------

def pnorm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nomes de colunas (minúsculas, separadores em '_', sem símbolos estranhos)."""
    def _norm(name: str) -> str:
        s = name.strip().lower()
        s = re.sub(r'[^a-z0-9]+', '_', s)
        s = re.sub(r'_{2,}', '_', s).strip('_')
        return s
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    return df


def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Retorna a primeira coluna encontrada dentre os candidatos ou lança ValueError."""
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
    """
    Extraí previsões para os timestamps da grade T.
    Lança KeyError se houver timestamps sem valor correspondente em alguma série.
    """
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
    """Tenta extrair valor numérico (compatível com pyomo.environ.value)."""
    try:
        from pyomo.environ import value
        x = value(v)
        return float(x if x is not None else default)
    except Exception:
        try:
            return float(v)
        except Exception:
            return float(default)


def horizon_snapshot(model, times: List[datetime]) -> Dict[str, List[float]]:
    """
    Snapshot 1D (para modelos com variáveis indexadas só em T).
    Mantido por compatibilidade com scripts Off-Grid anteriores.
    """
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


def horizon_snapshot_2d(model,
                        times: List[datetime],
                        scenario: Any) -> Dict[str, List[float]]:
    """
    Snapshot 2D (para modelos com variáveis indexadas em (T, C)), extraindo um cenário específico.
    Útil para o On-Grid estocástico.
    """
    ts = list(times)
    c = scenario
    out = {
        "scenario":   str(c),
        "timestamps": [t.isoformat() for t in ts],
        "dt_h":       [_val(model.dt_h[t]) for t in ts],
        "Load_kw":    [_val(model.Load_kw[t]) for t in ts],
        "PV_kw":      [_val(model.PV_kw[t]) for t in ts],
        "X_L":        [_val(model.X_L[t, c]) for t in ts],
        "X_PV":       [_val(model.X_PV[t, c]) for t in ts],
        "P_bess_kw":  [_val(model.P_bess[t, c]) for t in ts],
        "P_ch_kw":    [_val(model.P_ch[t, c]) for t in ts],
        "P_dis_kw":   [_val(model.P_dis[t, c]) for t in ts],
        "gamma":      [int(round(_val(model.gamma[t, c], 0))) for t in ts],
        "E_kwh":      [_val(model.E[t, c]) for t in ts],
        "P_grid_in_kw":  [_val(model.P_gin[t, c]) for t in ts] if hasattr(model, "P_gin") else [],
        "P_grid_out_kw": [_val(model.P_gout[t, c]) for t in ts] if hasattr(model, "P_gout") else [],
        "c_grid":        [_val(model.c_grid[t]) for t in ts] if hasattr(model, "c_grid") else [],
    }
    return out


def save_log(results_log, path: str = "outputs/dispatch_log.json"):
    """Salva um dicionário JSON com indentação."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import json as _json
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(results_log, f, ensure_ascii=False, indent=2)
    print(f"[log] salvo em {path}")
