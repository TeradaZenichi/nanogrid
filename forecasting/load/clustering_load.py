# -*- coding: utf-8 -*-
"""
Clustering de perfiles diarios (30 min) para demanda:
- Lee load_5min_train.csv (timestamp, p_norm)
- Reindexa/agrupa a 30 min y construye matrices día x 48
- Split temporal 70% train / 30% val por días (sin mezclar)
- Separa WEEKDAY vs WEEKEND y ajusta k-means SOLO en train
- Elbow (inertia) + silhouette para K=2..8
- Visualiza PCA 2D y prototipos por cluster
- Exporta artefactos (labels por día y prototipos) a ./outputs/

Autor: Juan Carlos Cortez
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# =========================
# Configuración
# =========================
CSV_PATH = "data/load_5min_train.csv"      # <-- tu archivo
OUT_DIR  = "data/clustering_30min"
os.makedirs(OUT_DIR, exist_ok=True)

RESAMPLE_RULE = "30min"               # 30 minutos
WEEKEND_DOW = {5, 6}                  # 5=Saturday, 6=Sunday
K_RANGE = range(2, 9)                 # K=2..8
SHAPE_MODE = "none"                   # "none" o "zscore_by_day"
RANDOM_STATE = 42

# =========================
# Utilidades
# =========================
def load_series(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")
    # Asegurar tipo float
    s = df["p_norm"].astype(float).copy()
    # (Opcional) Chequeo de grilla 5 min / gaps menores: s = s.asfreq("5T")...
    return s

def to_30min(s_5min: pd.Series) -> pd.Series:
    # Resampleo a 30 min por media (coherente con normalización min-max previa)
    s_30 = s_5min.resample(RESAMPLE_RULE, label="right", closed="right").mean()
    return s_30

def split_train_val_by_days(s_30: pd.Series, train_ratio: float = 0.7):
    days = pd.Index(s_30.index.normalize().unique()).sort_values()
    n_days = len(days)
    cut = int(math.floor(n_days * train_ratio))
    days_train = days[:cut]
    days_val   = days[cut:]
    return days_train, days_val

def build_daily_matrix(s_30: pd.Series, days: pd.Index, shape_mode: str = "none"):
    """
    Retorna:
      dates: list de fechas (solo días completos)
      X: np.ndarray (n_days, 48) con cada fila = vector diario a 30 min
    """
    rows = []
    keep_dates = []
    for d in days:
        day_slice = s_30.loc[(s_30.index >= d) & (s_30.index < d + pd.Timedelta(days=1))]
        if len(day_slice) == 48 and not day_slice.isna().any():
            x = day_slice.values.astype(float)
            if shape_mode == "zscore_by_day":
                mu = x.mean()
                sd = x.std()
                if sd < 1e-6:
                    x = x - mu  # evita división por ~0
                else:
                    x = (x - mu) / sd
            rows.append(x)
            keep_dates.append(pd.Timestamp(d))
        # si el día está incompleto o tiene NaN, se descarta
    if not rows:
        return pd.DatetimeIndex([]), np.empty((0, 48))
    X = np.vstack(rows)
    dates = pd.DatetimeIndex(keep_dates)
    return dates, X

def evaluate_kmeans_for_K(X: np.ndarray, k_list, random_state=RANDOM_STATE):
    inertia_list = []
    sil_list = []
    pca = PCA(n_components=10, random_state=random_state).fit(X)
    Z = pca.transform(X)
    for k in k_list:
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = km.fit_predict(Z)  # cluster en espacio PCA (más estable/rápido)
        inertia_list.append(km.inertia_)
        # silhouette requiere k>=2 y varianza > 0
        try:
            sil = silhouette_score(Z, labels, metric="euclidean")
        except Exception:
            sil = np.nan
        sil_list.append(sil)
    return np.array(inertia_list), np.array(sil_list)

def suggest_k_elbow(inertia: np.ndarray, k_list):
    """
    Sugerencia simple por 'codo': mayor segunda diferencia discreta
    (heurística; el usuario debe confirmar visualmente).
    """
    # normalizar K a índices 0..len-1
    if len(inertia) < 3:
        return k_list[0]
    # segunda derivada discreta aprox.
    d1 = np.diff(inertia)
    d2 = np.diff(d1)
    elbow_idx = np.argmin(d2) + 1  # más negativo => codo fuerte
    elbow_k = k_list[elbow_idx] if 0 <= elbow_idx < len(k_list) else k_list[0]
    return elbow_k

def fit_kmeans(X: np.ndarray, k: int, random_state=RANDOM_STATE):
    pca = PCA(n_components=min(10, X.shape[1]), random_state=random_state).fit(X)
    Z = pca.transform(X)
    km = KMeans(n_clusters=k, n_init=50, random_state=random_state).fit(Z)
    labels = km.labels_
    return {"pca": pca, "kmeans": km, "labels": labels}

def predict_kmeans(model, X: np.ndarray):
    Z = model["pca"].transform(X)
    return model["kmeans"].predict(Z)

def plot_elbow_and_sil(k_list, inertia, sil, title, savepath_prefix):
    plt.figure()
    plt.plot(list(k_list), inertia, marker="o")
    plt.xlabel("K")
    plt.ylabel("Inertia (PCA-space)")
    plt.title(f"Elbow - {title}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{savepath_prefix}_elbow.png", dpi=140)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(list(k_list), sil, marker="o")
    plt.xlabel("K")
    plt.ylabel("Silhouette (PCA-space)")
    plt.title(f"Silhouette - {title}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{savepath_prefix}_silhouette.png", dpi=140)
    plt.show()
    plt.close()

def plot_pca_scatter(X: np.ndarray, model, labels, title, savepath):
    Z = model["pca"].transform(X)
    plt.figure()
    scatter = plt.scatter(Z[:,0], Z[:,1], c=labels, s=14, alpha=0.8)
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=140)
    plt.close()

def plot_cluster_prototypes(dates, X, labels, title, savepath_prefix):
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure()
    K = int(labels.max() + 1)

    # Dibuja prototipos (media por cluster)
    for k in range(K):
        mask = labels == k
        if mask.sum() == 0:
            continue
        proto = X[mask].mean(axis=0)
        plt.plot(np.arange(X.shape[1]), proto, label=f"Cluster {k} (n={mask.sum()})")

    # Construcción de ticks/labels consistente con el nº de slots por día
    nslots = X.shape[1]  # p.ej., 48 si es 30 min, 24 si es 1 h, 288 si es 5 min, etc.
    if nslots % 24 == 0:
        slots_per_hour = nslots // 24                 # p.ej., 2 para 30 min, 12 para 5 min
        ticks = np.arange(0, nslots, slots_per_hour)  # un tick por hora
        labels = [f"{h:02d}:00" for h in range(24)]   # 24 labels, 00:00..23:00
    else:
        # Caso no múltiplo de 24: usamos 12 ticks equiespaciados y 12 labels (cada 2 horas)
        ticks = np.linspace(0, nslots - 1, 12, dtype=int)
        labels = [f"{h:02d}:00" for h in range(0, 24, 2)]

    plt.xticks(ticks=ticks, labels=labels, rotation=0)
    plt.xlabel("Hora del día")
    plt.ylabel("p_norm (resample)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{savepath_prefix}_prototypes.png", dpi=140)
    plt.show()
    plt.close()


def save_day_labels_csv(dates, labels, group_name, split_name):
    out = pd.DataFrame({"date": dates, "cluster": labels})
    out["group"] = group_name
    out["split"] = split_name
    out.to_csv(os.path.join(OUT_DIR, f"labels_{group_name}_{split_name}.csv"), index=False)

def save_prototypes_csv(X, labels, group_name, split_name):
    K = int(labels.max() + 1)
    rows = []
    for k in range(K):
        mask = labels == k
        if mask.sum() == 0:
            continue
        proto = X[mask].mean(axis=0)
        for h in range(48):
            rows.append({"group": group_name, "split": split_name, "cluster": k, "slot": h, "value": float(proto[h])})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, f"prototypes_{group_name}_{split_name}.csv"), index=False)

# =========================
# Flujo principal
# =========================
def main():
    print(">> Cargando y preparando serie...")
    s = load_series(CSV_PATH)
    s30 = to_30min(s)

    # Calendario
    df30 = s30.to_frame("p_norm")
    df30["dow"] = df30.index.dayofweek
    df30["is_weekend"] = df30["dow"].isin(WEEKEND_DOW).astype(int)

    # Split por días (crono)
    days_train, days_val = split_train_val_by_days(df30["p_norm"], train_ratio=0.7)
    print(f"Días total: {len(df30.index.normalize().unique())} | train: {len(days_train)} | val: {len(days_val)}")

    # Construir matrices día x 48 por grupo (weekday / weekend) y split (train / val)
    results = {}
    for group_name, weekend_flag in [("WEEKDAY", 0), ("WEEKEND", 1)]:
        # TRAIN
        mask_train = df30.index.normalize().isin(days_train) & (df30["is_weekend"] == weekend_flag)
        s_train = df30.loc[mask_train, "p_norm"]
        d_train = s_train.index.normalize().unique()
        dates_tr, X_tr = build_daily_matrix(s_train, d_train, shape_mode=SHAPE_MODE)

        # VAL (se asigna luego, no ajusta)
        mask_val = df30.index.normalize().isin(days_val) & (df30["is_weekend"] == weekend_flag)
        s_val = df30.loc[mask_val, "p_norm"]
        d_val = s_val.index.normalize().unique()
        dates_va, X_va = build_daily_matrix(s_val, d_val, shape_mode=SHAPE_MODE)

        results[group_name] = {"train": (dates_tr, X_tr), "val": (dates_va, X_va)}

        print(f"[{group_name}] días completos -> train={len(dates_tr)}, val={len(dates_va)}")

        if X_tr.shape[0] < 4:
            print(f" !! Pocos días completos en {group_name} para clustering (min 4 recomendado).")
            continue

        # ---- Elbow & silhouette en TRAIN ----
        inertia, sil = evaluate_kmeans_for_K(X_tr, K_RANGE, random_state=RANDOM_STATE)
        elbow_k = suggest_k_elbow(inertia, list(K_RANGE))
        print(f"   Sugerencia K (elbow heurístico) para {group_name}: {elbow_k}")

        prefix = os.path.join(OUT_DIR, f"{group_name}")
        plot_elbow_and_sil(K_RANGE, inertia, sil, f"{group_name} (TRAIN)", f"{prefix}_train")

        # ---- Entrenar modelo final con K sugerido (puedes cambiar manualmente) ----
        model = fit_kmeans(X_tr, k=elbow_k, random_state=RANDOM_STATE)
        labels_tr = model["kmeans"].labels_
        save_day_labels_csv(dates_tr, labels_tr, group_name, "train")
        save_prototypes_csv(X_tr, labels_tr, group_name, "train")

        # Visualizaciones
        plot_pca_scatter(X_tr, model, labels_tr, f"{group_name} TRAIN - PCA by cluster (K={elbow_k})",
                         f"{prefix}_train_pca.png")
        plot_cluster_prototypes(dates_tr, X_tr, labels_tr, f"{group_name} TRAIN - Prototipos (K={elbow_k})",
                                f"{prefix}_train")

        # ---- Asignar clusters en VAL (solo predicción) ----
        if X_va.shape[0] > 0:
            labels_va = predict_kmeans(model, X_va)
            save_day_labels_csv(dates_va, labels_va, group_name, "val")
            save_prototypes_csv(X_va, labels_va, group_name, "val")  # prototipos val (descriptivo)
            # PCA con val (marcado diferente)
            # Reusa plot de train; aquí solo exportamos labels.

    print(">> Listo. Revisa ./outputs/clustering_30min para gráficos y CSVs.")

if __name__ == "__main__":
    main()
