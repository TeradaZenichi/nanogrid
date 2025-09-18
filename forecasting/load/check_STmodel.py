# -*- coding: utf-8 -*-
import os, time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -------- Minimal config --------
CSV_PATH       = "data/load_5min_train.csv"
MODEL_PATH     = "models/lstm_hourly_load_36h_8lb_20250902-222618.keras"

LOOKBACK_HOURS = 8   # debe coincidir con el entrenamiento
HORIZON_HOURS  = 36  # idem
RESAMPLE_AGG   = "mean"  # "mean" | "median" | "sum"

# Lista de horas (fin de la ventana de entrada, redondeadas a la hora)
END_TIMES = [
    # ejemplo (cambia a lo que quieras):
    "2008-05-01 00:00:00",
    "2008-06-01 00:00:00",
]

# Si END_TIMES está vacío, se eligen aleatorias
NUM_SAMPLES_RANDOM = 6
SEED = 42

FIG_DIR = "figures"; os.makedirs(FIG_DIR, exist_ok=True)

# -------- Helpers cortos --------
def add_cyc_feats_hourly(df):
    idx = df.index
    dow = idx.dayofweek.values.astype(float)      # 0..6
    hod = idx.hour.values.astype(float)           # 0..23
    df["dow_sin"] = np.sin(2*np.pi*dow/7);  df["dow_cos"] = np.cos(2*np.pi*dow/7)
    df["hod_sin"] = np.sin(2*np.pi*hod/24); df["hod_cos"] = np.cos(2*np.pi*hod/24)
    return df

def build_end_aligned(X, y, lookback, horizon, end_indices):
    Xs = np.zeros((len(end_indices), lookback, X.shape[1]), dtype=np.float32)
    Ys = np.zeros((len(end_indices), horizon), dtype=np.float32)
    for k,i in enumerate(end_indices):
        Xs[k] = X[i-lookback+1:i+1, :]
        Ys[k] = y[i+1:i+1+horizon]
    return Xs, Ys

def kpi(y, p):
    mae = float(np.mean(np.abs(y - p)))
    mse = float(np.mean((y - p)**2))
    rmse = float(np.sqrt(mse))
    yt, yp = y.reshape(-1), p.reshape(-1)
    ss_res = np.sum((yt-yp)**2); ss_tot = np.sum((yt-yt.mean())**2) + 1e-12
    r2 = float(1 - ss_res/ss_tot)
    return mae, rmse, r2

# -------- Carga datos (→ horario) --------
df5 = pd.read_csv(CSV_PATH, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
if RESAMPLE_AGG == "mean":
    dfH = df5[["p_norm"]].resample("h").mean(numeric_only=True)
elif RESAMPLE_AGG == "median":
    dfH = df5[["p_norm"]].resample("h").median(numeric_only=True)
else:
    dfH = df5[["p_norm"]].resample("h").sum(numeric_only=True)

dfH = dfH.dropna()
dfH = add_cyc_feats_hourly(dfH)

feature_cols = ["p_norm","dow_sin","dow_cos","hod_sin","hod_cos"]
X_all = dfH[feature_cols].values.astype(np.float32)
y_all = dfH["p_norm"].values.astype(np.float32)

LB, H = int(LOOKBACK_HOURS), int(HORIZON_HOURS)
n = len(dfH)
i_min = LB - 1
i_max = n - 1 - H
if i_max < i_min:
    raise ValueError("Not enough hourly data for chosen LOOKBACK/HORIZON.")

# -------- Elegir ventanas (sin restricción de validación) --------
def endtime_to_index(ts_str):
    t = pd.Timestamp(ts_str).floor("h")
    pos = dfH.index.get_indexer([t])[0]  # exact match en índice horario
    if pos == -1:
        raise ValueError(f"{t} not found in hourly index after resample.")
    if pos < i_min or pos > i_max:
        raise ValueError(f"{t} lacks enough history/future for LB={LB}, H={H}. "
                         f"Valid end idx ∈ [{i_min}, {i_max}] → time ∈ "
                         f"[{dfH.index[i_min]}, {dfH.index[i_max]}].")
    return pos

if END_TIMES:
    end_abs_idx = [endtime_to_index(ts) for ts in END_TIMES]
else:
    rng = np.random.default_rng(SEED)
    # Todo el rango válido (train+val): [i_min .. i_max]
    all_valid = np.arange(i_min, i_max+1, dtype=int)
    end_abs_idx = rng.choice(all_valid, size=min(NUM_SAMPLES_RANDOM, len(all_valid)), replace=False).tolist()

# -------- Construir tensores, cargar modelo, predecir --------
X_sel, y_sel = build_end_aligned(X_all, y_all, LB, H, end_abs_idx)

t0 = time.perf_counter()
model = load_model(MODEL_PATH, compile=False, safe_mode=False)
print(f"Model loaded in {time.perf_counter()-t0:.2f}s")

y_hat = model.predict(X_sel, verbose=0)

# -------- Métricas globales (sobre las muestras elegidas) --------
MAE, RMSE, R2 = kpi(y_sel, y_hat)
MAE_1, RMSE_1, _ = kpi(y_sel[:,0], y_hat[:,0])
print(f"\nSelected windows: {len(end_abs_idx)}")
print(f"Full horizon → MAE={MAE:.6f}  RMSE={RMSE:.6f}  R2={R2:.6f}")
print(f"First step   → MAE={MAE_1:.6f} RMSE={RMSE_1:.6f}")

# -------- Plot por muestra --------
cols = min(3, len(end_abs_idx)); rows = int(np.ceil(len(end_abs_idx)/cols))
fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3.2*rows), squeeze=False)

for j, i_end in enumerate(end_abs_idx):
    ax = axes[j//cols][j%cols]
    t_end = dfH.index[i_end]
    horizon_times = pd.date_range(start=t_end + pd.Timedelta(hours=1), periods=H, freq="h")
    ax.plot(horizon_times, y_sel[j], label="True", marker="o", linewidth=1)
    ax.plot(horizon_times, y_hat[j], label="Pred", marker="x", linewidth=1)
    ax.set_title(f"end={t_end}")
    ax.grid(True, alpha=0.3)

# Apaga axes vacíos
for k in range(j+1, rows*cols):
    axes[k//cols][k%cols].axis("off")

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")
fig.suptitle(f"Hourly ST | LB={LB}h, H={H}h | samples={len(end_abs_idx)}", y=1.02, fontsize=12)
fig.tight_layout()
out_png = os.path.join(FIG_DIR, f"st_samples_{int(time.time())}.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved plot: {out_png}")
plt.show()
