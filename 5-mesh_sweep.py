# -*- coding: utf-8 -*-
"""E3 — Sweep da malha temporal (h x dt1 x dt2): eixo central do paper.

Para cada combinacao valida, roda o MPC com prototipo (e opcionalmente o
ideal e o estocastico) na mesma janela com os mesmos outages, medindo custo
operacional e tempo de solve. Combos invalidos (dt2 nao divide h - outage)
sao pulados antes de qualquer solve.
"""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

import pandas as pd

from forecasting import PerfectForecast, PrototypeForecast
from opt import simulate_mpc, simulate_stochastic
from opt.operation import validate_time_mesh
from opt.utils import load_series_scaled

START_TS = pd.Timestamp("2009-05-01 00:00:00")
N_ITERS = 2880  # 10 dias
WORKERS = 8     # processos em paralelo (1 thread de solver cada)
OUT_ROOT = "Results/5-mesh_sweep"
PARAMS_JSON = "data/parameters.json"

HORIZONS_H = [6, 12, 24, 36]
T1_MIN = [5, 10, 15]
T2_MIN = [30, 60, 120, 180]
CONTROLLERS = ["prototype", "ideal", "stochastic"]  # reduza para acelerar


def run_combo(controller: str, h: int, t1: int, t2: int) -> dict:
    """Executa um (controlador, h, dt1, dt2); roda em subprocesso do pool."""
    params = json.load(open(PARAMS_JSON, encoding="utf-8"))
    params["time"]["horizon_hours"] = h
    params["time"]["timestep_1_min"] = t1
    params["time"]["timestep_2_min"] = t2

    out_dir = f"{OUT_ROOT}/{controller}/h{h}_t1_{t1}_t2_{t2}"
    if controller == "stochastic":
        metrics = simulate_stochastic(params, START_TS, N_ITERS, out_dir)
    else:
        scaling = {"P_L_nom_kw": params["Load"]["Pmax_kw"], "P_PV_nom_kw": params["PV"]["Pmax_kw"]}
        load_s, pv_s = load_series_scaled(scaling, "data/load_5min_test.csv", "data/pv_5min_test.csv")
        if controller == "ideal":
            forecaster = PerfectForecast(load_s, pv_s)
        else:
            forecaster = PrototypeForecast(
                None, load_s, pv_s,
                float(params["PV"]["Pmax_kw"]), float(params["Load"]["Pmax_kw"]),
                strategy="prefix",
            )
        metrics = simulate_mpc(params, forecaster, START_TS, N_ITERS, out_dir,
                               forecaster_name=controller)
    metrics["combo"] = f"h{h}_t1_{t1}_t2_{t2}"
    metrics["controller_name"] = controller
    return metrics


if __name__ == "__main__":
    base = json.load(open(PARAMS_JSON, encoding="utf-8"))

    # Monta a lista de combos validos (a validacao e barata e nao usa solver).
    combos = []
    for h in HORIZONS_H:
        for t1 in T1_MIN:
            for t2 in T2_MIN:
                trial = deepcopy(base)
                trial["time"].update(horizon_hours=h, timestep_1_min=t1, timestep_2_min=t2)
                try:
                    validate_time_mesh(trial)
                except Exception:
                    continue  # combo invalido: nem vira tarefa
                combos.append((h, t1, t2))
    tasks = [(c, h, t1, t2) for c in CONTROLLERS for (h, t1, t2) in combos]
    print(f"{len(combos)} combos validos x {len(CONTROLLERS)} controladores = {len(tasks)} runs")

    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_combo, *t): t for t in tasks}
        for k, fut in enumerate(as_completed(futures), start=1):
            c, h, t1, t2 = futures[fut]
            try:
                rows.append(fut.result())
                print(f"[{k}/{len(tasks)}] ok: {c} h{h}_t1_{t1}_t2_{t2}")
            except Exception as e:
                print(f"[{k}/{len(tasks)}] FALHOU: {c} h{h}_t1_{t1}_t2_{t2}: {e}")
                rows.append({"controller_name": c, "combo": f"h{h}_t1_{t1}_t2_{t2}",
                             "status": "error", "error": str(e)})

    summary = pd.DataFrame(rows)
    summary.to_csv(f"{OUT_ROOT}/summary.csv", index=False)
    print(f"\nresumo ({len(rows)} runs) salvo em {OUT_ROOT}/summary.csv")
