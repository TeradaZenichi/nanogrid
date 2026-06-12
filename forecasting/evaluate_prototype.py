# -*- coding: utf-8 -*-
# File: forecasting/evaluate_prototype.py
"""
Offline evaluation of the analog-day (prototype) forecaster against naive
baselines (and optionally the LSTM pipeline), on the test series.

For each forecast origin (default: every 60 min), each method produces a
36h-ahead forecast on a 5-min grid. Errors are reported by lead-time bucket
at two resolutions (native 5-min and 60-min block means), with skill scores
relative to the seasonal-naive baseline. Also reports cluster-classification
accuracy of the prefix strategy as a function of the hour of day, using the
full-day nearest-prototype assignment as the reference.

Usage: call run_evaluation(days=..., every_min=..., with_lstm=...) — see 2-forecast_eval.py.

Outputs (Results/forecasting/):
    prototype_eval_metrics.csv
    prototype_eval_summary.json
    prototype_eval_error_by_lead.png
    prototype_eval_cluster_accuracy.png
"""

from __future__ import annotations


import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting.prototype_forecast import (  # noqa: E402
    SLOT_MIN,
    SLOTS_PER_DAY,
    PrototypeForecast,
    _load_prototypes,
)
from opt.utils import load_series_scaled  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

LOAD_CSV = "data/load_5min_test.csv"
PV_CSV = "data/pv_5min_test.csv"
PARAMS_JSON = "data/parameters.json"
OUT_DIR = Path("Results/forecasting")

LEAD_BUCKETS_H = [(0, 1), (1, 6), (6, 12), (12, 24), (24, 36)]
HORIZON_MIN = 36 * 60
STEP_MIN = 5
N_STEPS = HORIZON_MIN // STEP_MIN


def _forecast_to_array(fc_dict: dict) -> np.ndarray:
    return np.asarray(list(fc_dict.values()), dtype=float)


def _seasonal_naive(s: pd.Series, t0: pd.Timestamp) -> np.ndarray:
    """forecast(t0 + l) = actual(t0 - 24h + (l mod 24h)) — causal cyclic repeat of yesterday."""
    out = np.empty(N_STEPS)
    for i in range(N_STEPS):
        lead = (i * STEP_MIN) % (24 * 60)
        ts = t0 - pd.Timedelta(hours=24) + pd.Timedelta(minutes=lead)
        out[i] = float(s.get(ts, np.nan))
    return out


def _persistence(s: pd.Series, t0: pd.Timestamp) -> np.ndarray:
    return np.full(N_STEPS, float(s.get(t0, np.nan)))


def _actuals(s: pd.Series, t0: pd.Timestamp) -> np.ndarray:
    idx = pd.date_range(t0, periods=N_STEPS, freq=f"{STEP_MIN}min")
    return s.reindex(idx).to_numpy(dtype=float)


def _bucket_metrics(err_rows: list[dict]) -> pd.DataFrame:
    """err_rows: dicts with method, target, resolution, lead_h, abs_err, sq_err."""
    df = pd.DataFrame(err_rows)
    records = []
    for (method, target, res), g in df.groupby(["method", "target", "resolution"]):
        for lo, hi in LEAD_BUCKETS_H:
            gb = g[(g["lead_h"] > lo) & (g["lead_h"] <= hi)]
            if gb.empty:
                continue
            records.append(
                {
                    "method": method,
                    "target": target,
                    "resolution": res,
                    "lead_bucket_h": f"({lo},{hi}]",
                    "n": int(len(gb)),
                    "mae": float(gb["abs_err"].mean()),
                    "rmse": float(np.sqrt(gb["sq_err"].mean())),
                }
            )
    out = pd.DataFrame(records)
    # Skill vs seasonal-naive (same target/resolution/bucket)
    base = out[out["method"] == "seasonal-naive"].set_index(
        ["target", "resolution", "lead_bucket_h"]
    )["mae"]
    out["skill_vs_seasonal"] = out.apply(
        lambda r: 1.0 - r["mae"] / base.get((r["target"], r["resolution"], r["lead_bucket_h"]), np.nan),
        axis=1,
    )
    return out


def _accumulate_errors(rows: list, method: str, target: str,
                       y_hat: np.ndarray, y_true: np.ndarray) -> None:
    lead_h = (np.arange(N_STEPS) + 1) * STEP_MIN / 60.0
    ok = ~(np.isnan(y_hat) | np.isnan(y_true))
    err = y_hat[ok] - y_true[ok]
    for lh, e in zip(lead_h[ok], err):
        rows.append(
            {"method": method, "target": target, "resolution": "5min",
             "lead_h": float(lh), "abs_err": abs(float(e)), "sq_err": float(e) ** 2}
        )
    # 60-min block means (12 x 5-min steps per block)
    n_blocks = N_STEPS // 12
    yh = y_hat[: n_blocks * 12].reshape(n_blocks, 12)
    yt = y_true[: n_blocks * 12].reshape(n_blocks, 12)
    bh = np.nanmean(yh, axis=1)
    bt = np.nanmean(yt, axis=1)
    for b in range(n_blocks):
        if np.isnan(bh[b]) or np.isnan(bt[b]):
            continue
        e = bh[b] - bt[b]
        rows.append(
            {"method": method, "target": target, "resolution": "60min",
             "lead_h": float(b + 1), "abs_err": abs(float(e)), "sq_err": float(e) ** 2}
        )


def _classification_accuracy(fc: PrototypeForecast, s: pd.Series, scale: float,
                             protos: dict, kind: str, days: list) -> pd.DataFrame:
    """Prefix-classification accuracy by hour of day vs full-day nearest prototype."""
    recs = []
    for day in days:
        day = pd.Timestamp(day)
        idx = pd.date_range(day, periods=24 * 60 // STEP_MIN, freq=f"{STEP_MIN}min")
        vals = s.reindex(idx).to_numpy(dtype=float) / scale
        if np.isnan(vals).any():
            continue
        slots = vals.reshape(SLOTS_PER_DAY, (SLOT_MIN // STEP_MIN)).mean(axis=1)
        oracle = min(protos, key=lambda cl: float(np.linalg.norm(slots - protos[cl])))
        for hour in range(1, 24):
            now = day + pd.Timedelta(hours=hour)
            pred = fc._classify(kind, now)
            recs.append({"target": kind, "hour": hour, "correct": int(pred == oracle)})
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    return df.groupby(["target", "hour"], as_index=False)["correct"].mean().rename(
        columns={"correct": "accuracy"}
    )


def run_evaluation(
    days: int = 28,
    every_min: int = 60,
    start: str | None = None,
    with_lstm: bool = False,
    out_dir=OUT_DIR,
) -> None:
    """Evaluate prototype/naive (and optionally LSTM) forecasts over the test series.

    days      : evaluation window length in days
    every_min : spacing between forecast origins
    start     : first forecast origin (default: data start + 24h)
    with_lstm : also evaluate ForecastMPC (loads TensorFlow)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = json.load(open(PARAMS_JSON, encoding="utf-8"))
    load_scale = float(params["Load"]["Pmax_kw"])
    pv_scale = float(params["PV"]["Pmax_kw"])
    scaling = {"P_L_nom_kw": load_scale, "P_PV_nom_kw": pv_scale}
    load_s, pv_s = load_series_scaled(scaling, LOAD_CSV, PV_CSV)

    data_start = max(load_s.index.min(), pv_s.index.min())
    data_end = min(load_s.index.max(), pv_s.index.max())
    t_first = pd.Timestamp(start) if start else (data_start + pd.Timedelta(hours=24)).ceil("h")
    t_last = min(t_first + pd.Timedelta(days=days), data_end - pd.Timedelta(minutes=HORIZON_MIN))
    origins = pd.date_range(t_first, t_last, freq=f"{every_min}min")
    print(f"[eval] {len(origins)} forecast origins in [{t_first}, {t_last}], horizon {HORIZON_MIN/60:.0f}h")

    forecasters = {
        f"prototype-{st}": PrototypeForecast(None, load_s, pv_s, pv_scale, load_scale, strategy=st)
        for st in ("calendar", "prefix", "knn")
    }
    lstm = None
    if with_lstm:
        from forecasting.get_forecasting import ForecastMPC

        lstm = ForecastMPC({}, load_s, pv_s, pv_scale, load_scale)

    rows: list[dict] = []
    for k, t0 in enumerate(origins):
        y_true_load = _actuals(load_s, t0)
        y_true_pv = _actuals(pv_s, t0)
        if np.isnan(y_true_load).all():
            continue

        methods: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "seasonal-naive": (_seasonal_naive(load_s, t0), _seasonal_naive(pv_s, t0)),
            "persistence": (_persistence(load_s, t0), _persistence(pv_s, t0)),
        }
        for name, fc in forecasters.items():
            try:
                out = fc.get_forecasts(t0, intervals=N_STEPS, dt_min=STEP_MIN)
            except KeyError:
                continue
            methods[name] = (_forecast_to_array(out["load_kw"]), _forecast_to_array(out["pv_kw"]))
        if lstm is not None:
            try:
                out = lstm.get_forecasts(t0, intervals=N_STEPS, dt_min=STEP_MIN)
                methods["lstm"] = (_forecast_to_array(out["load_kw"]), _forecast_to_array(out["pv_kw"]))
            except Exception as e:
                print(f"[eval] lstm failed at {t0}: {e}")

        for name, (yl, yp) in methods.items():
            _accumulate_errors(rows, name, "load", yl, y_true_load)
            _accumulate_errors(rows, name, "pv", yp, y_true_pv)

        if (k + 1) % 50 == 0:
            print(f"[eval] {k + 1}/{len(origins)} origins done")

    metrics = _bucket_metrics(rows)
    metrics_csv = out_dir / "prototype_eval_metrics.csv"
    metrics.to_csv(metrics_csv, index=False)
    print(f"[eval] metrics saved: {metrics_csv.as_posix()}")

    # Cluster-classification accuracy by hour (prefix strategy)
    fc_prefix = forecasters["prototype-prefix"]
    eval_days = sorted({ts.normalize() for ts in origins})
    acc_load = _classification_accuracy(
        fc_prefix, load_s, load_scale, _load_prototypes(Path(fc_prefix.params["LOAD_PROTOTYPES"])), "load", eval_days
    )
    acc_pv = _classification_accuracy(
        fc_prefix, pv_s, pv_scale, _load_prototypes(Path(fc_prefix.params["PV_PROTOTYPES"])), "pv", eval_days
    )
    acc = pd.concat([acc_load, acc_pv], ignore_index=True)
    acc_csv = out_dir / "prototype_eval_cluster_accuracy.csv"
    acc.to_csv(acc_csv, index=False)

    summary = {
        "origins": int(len(origins)),
        "window": [str(t_first), str(t_last)],
        "methods": sorted(metrics["method"].unique().tolist()),
        "load_mae_5min_by_method": {
            m: float(g["mae"].mean())
            for m, g in metrics[(metrics["target"] == "load") & (metrics["resolution"] == "5min")].groupby("method")
        },
        "load_skill_vs_seasonal_60min": {
            m: float(g["skill_vs_seasonal"].mean())
            for m, g in metrics[(metrics["target"] == "load") & (metrics["resolution"] == "60min")].groupby("method")
        },
    }
    (out_dir / "prototype_eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if plt is not None:
        for target in ("load", "pv"):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sub = metrics[(metrics["target"] == target) & (metrics["resolution"] == "60min")]
            for m, g in sub.groupby("method"):
                ax.plot(g["lead_bucket_h"], g["mae"], marker="o", label=m)
            ax.set_title(f"{target.upper()} MAE by lead-time bucket (60-min blocks)")
            ax.set_xlabel("Lead bucket (h)")
            ax.set_ylabel("MAE (kW)")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"prototype_eval_error_by_lead_{target}.png", dpi=180)
            plt.close(fig)

        if not acc.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for target, g in acc.groupby("target"):
                ax.plot(g["hour"], g["accuracy"], marker="o", label=target)
            ax.set_title("Prefix cluster-classification accuracy vs hour of day")
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("Accuracy vs full-day assignment")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "prototype_eval_cluster_accuracy.png", dpi=180)
            plt.close(fig)
        print(f"[eval] plots saved in {out_dir.as_posix()}")


if __name__ == "__main__":
    run_evaluation()
