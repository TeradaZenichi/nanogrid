# -*- coding: utf-8 -*-
"""
Daily-profile clustering (30-min) for PV power.

What this script does
---------------------
- Reads pv_5min_train.csv with columns: [timestamp, p_norm].
- Resamples to 30-min means and builds day-by-row matrices restricted to a
  user-defined time window (default full 24h); each row is one day's vector.
- Chronological split: 70% train / 30% val by whole days.
- Two clustering options:
  (A) PCA + KMeans using Euclidean metric (fast, baseline)
  (B) TimeSeriesKMeans with DTW from tslearn (shape-aware)
- For (A) computes elbow (inertia) and silhouette over K=2..8 in PCA space.
- For (B) computes elbow (inertia) only (silhouette w/ DTW omitted).
- Saves artifacts with a `_pv` suffix to keep them separate from load outputs:
    labels_pv_{split}.csv
    prototypes_pv_{split}.csv
    pv_train_pv_elbow.png
    pv_train_pv_silhouette.png (Euclidean only)
    pv_train_pca.png
    pv_train_pv_prototypes.png
    (DTW adds *_dtw_* file names too)
- Plots PCA scatter (for visualization) and cluster prototypes.

Author: Adapted for PV by Juan Carlos Cortez
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
# Configuration
# =========================
CSV_PATH = "data/pv_5min_train.csv"       # Input PV file: [timestamp, p_norm]
TEST_CSV_PATH = "data/pv_5min_test.csv"  
OUT_DIR  = "data/clustering_30min_pv"
os.makedirs(OUT_DIR, exist_ok=True)

RESAMPLE_RULE = "30min"                   # 30-minute profiles
K_RANGE = range(2, 6)                     # K in [2..6]
SHAPE_MODE = "none"                       # "none" or "zscore_by_day"
RANDOM_STATE = 42

# Time window for clustering (use 24h by default).
# Examples:
#   "08:00" -> "19:00": cluster only daytime window
#   "00:00" -> "24:00": full day (default)
TRAIN_HOUR_START = "07:00"
TRAIN_HOUR_END   = "19:00"

# Clustering algorithm: "euclid" (PCA+KMeans) or "dtw" (tslearn TimeSeriesKMeans)
CLUSTER_ALGO = "dtw"                   # change to "dtw" if you want DTW

# =========================
# I/O helpers
# =========================
def load_series(csv_path: str) -> pd.Series:
    """Load CSV into a pandas Series indexed by timestamp with float values."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")
    s = df["p_norm"].astype(float).copy()
    return s

def to_30min(s_5min: pd.Series) -> pd.Series:
    """Resample 5-min series to 30-min mean values."""
    return s_5min.resample(RESAMPLE_RULE, label="right", closed="right").mean()

def split_train_val_by_days(s_30: pd.Series, train_ratio: float = 0.7):
    """Chronological day split without mixing: returns (days_train, days_val)."""
    days = pd.Index(s_30.index.normalize().unique()).sort_values()
    n_days = len(days)
    cut = int(math.floor(n_days * train_ratio))
    return days[:cut], days[cut:]

# =========================
# Windowing & matrices
# =========================
def _normalize_end_time_str(t_end: str) -> tuple[str, bool]:
    """
    Convert "24:00" so pandas between_time can include end-of-day.
    Returns (end_string, force_include_end_flag).
    """
    if t_end.strip() in ("24:00", "24:00:00"):
        return "23:59:59.999999", True
    return t_end, False

def build_daily_matrix(
    s_30: pd.Series,
    days: pd.Index,
    shape_mode: str = "none",
    hour_start: str = "00:00",
    hour_end: str   = "24:00",
):
    """
    Build a day-by-row matrix restricted to [hour_start, hour_end).

    Returns
    -------
    dates : pd.DatetimeIndex
        Dates for which a complete window exists with no NaNs.
    X : np.ndarray, shape (n_days, n_slots)
        One row per day, columns are 30-min slots in the selected window.
    """
    rows = []
    keep_dates = []

    end_str, force_include_end = _normalize_end_time_str(hour_end)
    inclusive_mode = "both" if force_include_end else "left"   # include start; include end only if 24:00

    nslots_target = None  # will be set using the first valid day

    for d in days:
        # Select the day [d, d+1)
        day_slice = s_30.loc[(s_30.index >= d) & (s_30.index < d + pd.Timedelta(days=1))]
        if day_slice.empty or day_slice.isna().all():
            continue

        # Restrict to the desired time window within the day
        day_win = day_slice.between_time(hour_start, end_str, inclusive=inclusive_mode)

        # Fix the expected number of slots from the first valid day
        if nslots_target is None:
            nslots_target = len(day_win)

        # Keep only complete windows with no NaNs
        if len(day_win) == nslots_target and not day_win.isna().any():
            x = day_win.values.astype(float)
            if shape_mode == "zscore_by_day":
                mu = x.mean()
                sd = x.std()
                x = (x - mu) / sd if sd >= 1e-6 else (x - mu)
            rows.append(x)
            keep_dates.append(pd.Timestamp(d))

    if not rows:
        return pd.DatetimeIndex([]), np.empty((0, 0))
    X = np.vstack(rows)
    dates = pd.DatetimeIndex(keep_dates)
    return dates, X

def _time_axis_ticks(nslots: int, hour_start: str):
    """
    Build x-ticks & labels aligned to hours given slot count (30-min slots)
    and the chosen starting hour. If nslots is not multiple of 2, uses ~12 ticks.
    """
    # 30-min slots -> 2 slots per hour
    if nslots % 2 == 0:
        slots_per_hour = 2
        ticks = np.arange(0, nslots, slots_per_hour)
        # Parse hour_start
        h, m = map(int, hour_start.split(":"))
        labels = []
        cur_h = h
        cur_m = 0 if m == 0 else 30  # rough snap for nicer labels
        for _ in range(len(ticks)):
            labels.append(f"{cur_h:02d}:00")
            cur_h = (cur_h + 1) % 24
        return ticks, labels
    else:
        ticks = np.linspace(0, nslots - 1, 12, dtype=int)
        labels = [f"{i}" for i in range(1, 13)]
        return ticks, labels

# =========================
# Euclidean (PCA + KMeans)
# =========================
def evaluate_kmeans_for_K(X: np.ndarray, k_list, random_state=RANDOM_STATE):
    """Compute inertia & silhouette in PCA space for a range of K."""
    inertia_list = []
    sil_list = []
    pca = PCA(n_components=min(10, X.shape[1]), random_state=random_state).fit(X)
    Z = pca.transform(X)
    for k in k_list:
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = km.fit_predict(Z)
        inertia_list.append(km.inertia_)
        # Silhouette requires non-degenerate variance
        try:
            sil = silhouette_score(Z, labels, metric="euclidean")
        except Exception:
            sil = np.nan
        sil_list.append(sil)
    return np.array(inertia_list), np.array(sil_list)

def suggest_k_elbow(inertia: np.ndarray, k_list):
    """Heuristic: pick K at which the 2nd discrete difference is most negative."""
    if len(inertia) < 3:
        return k_list[0]
    d1 = np.diff(inertia)
    d2 = np.diff(d1)
    elbow_idx = np.argmin(d2) + 1
    return k_list[elbow_idx] if 0 <= elbow_idx < len(k_list) else k_list[0]

def fit_kmeans(X: np.ndarray, k: int, random_state=RANDOM_STATE):
    """Fit PCA + KMeans and return the fitted objects and labels."""
    pca = PCA(n_components=min(10, X.shape[1]), random_state=random_state).fit(X)
    Z = pca.transform(X)
    km = KMeans(n_clusters=k, n_init=50, random_state=random_state).fit(Z)
    labels = km.labels_
    return {"pca": pca, "kmeans": km, "labels": labels}

def predict_kmeans(model, X: np.ndarray):
    Z = model["pca"].transform(X)
    return model["kmeans"].predict(Z)

# =========================
# DTW (tslearn)
# =========================
def _require_tslearn():
    try:
        from tslearn.clustering import TimeSeriesKMeans  # noqa: F401
        from tslearn.utils import to_time_series_dataset  # noqa: F401
        return True
    except Exception as e:
        print(">> DTW requested but tslearn is not available. Install with: pip install tslearn")
        print(f"   Detail: {e}")
        return False

def evaluate_dtw_for_K(X: np.ndarray, k_list, random_state=RANDOM_STATE):
    """
    Evaluate DTW clustering across K. Returns (inertia_list, models_dict).
    Silhouette for DTW is omitted here; inertia_ (within-cluster sum) is used.
    """
    from tslearn.clustering import TimeSeriesKMeans
    X_ts = X[:, :, None]  # shape to (n_samples, n_timestamps, 1)
    inertia_list = []
    models = {}
    for k in k_list:
        tskm = TimeSeriesKMeans(n_clusters=k, metric="dtw",
                                n_init=5, random_state=random_state)
        tskm.fit(X_ts)
        inertia_list.append(tskm.inertia_)
        models[k] = tskm
    return np.array(inertia_list), models

def predict_dtw(model, X: np.ndarray):
    X_ts = X[:, :, None]
    return model.predict(X_ts)

# =========================
# Plotting & saving
# =========================
def plot_elbow_and_sil(k_list, inertia, sil, title, savepath_prefix):
    # Elbow
    plt.figure()
    plt.plot(list(k_list), inertia, marker="o")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.title(f"Elbow - {title}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{savepath_prefix}_pv_elbow.png", dpi=140)
    plt.close()

    # Silhouette (if provided)
    if sil is not None:
        plt.figure()
        plt.plot(list(k_list), sil, marker="o")
        plt.xlabel("K")
        plt.ylabel("Silhouette")
        plt.title(f"Silhouette - {title}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{savepath_prefix}_pv_silhouette.png", dpi=140)
        plt.close()

def plot_pca_scatter(X: np.ndarray, model_pca, labels, title, savepath):
    Z = model_pca["pca"].transform(X) if "pca" in model_pca else PCA(
        n_components=min(10, X.shape[1]), random_state=RANDOM_STATE
    ).fit_transform(X)
    plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=14, alpha=0.8)
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=140)
    plt.close()

def plot_cluster_prototypes(X, labels, hour_start, title, savepath_prefix):
    """
    Plot cluster prototypes (mean per cluster) for any algorithm.
    """
    K = int(labels.max() + 1)
    plt.figure()
    for k in range(K):
        mask = labels == k
        if mask.sum() == 0:
            continue
        proto = X[mask].mean(axis=0)
        plt.plot(np.arange(X.shape[1]), proto, label=f"Cluster {k} (n={mask.sum()})")

    nslots = X.shape[1]
    ticks, labels_x = _time_axis_ticks(nslots, hour_start)
    plt.xticks(ticks=ticks, labels=labels_x, rotation=0)
    plt.xlabel("Hour of day (selected window)")
    plt.ylabel("p_norm (30-min)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{savepath_prefix}_pv_prototypes.png", dpi=140)
    plt.close()

def plot_cluster_counts(labels: np.ndarray, title: str, savepath: str):
    """Bar chart of number of samples per cluster."""
    if labels.size == 0:
        print("No labels to plot for counts.")
        return
    K = int(labels.max() + 1)
    counts = [(labels == k).sum() for k in range(K)]

    plt.figure()
    plt.bar(range(K), counts)
    plt.xlabel("Cluster id")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=140)
    plt.close()

def save_day_labels_csv(dates, labels, split_name, algo_tag=""):
    out = pd.DataFrame({"date": dates, "cluster": labels})
    out["group"] = "PV_ALL"
    out["split"] = split_name
    tag = f"_dtw" if algo_tag == "dtw" else ""
    out.to_csv(os.path.join(OUT_DIR, f"labels_pv{tag}_{split_name}.csv"), index=False)

def save_prototypes_csv(X, labels, split_name, algo_tag=""):
    K = int(labels.max() + 1)
    rows = []
    for k in range(K):
        mask = labels == k
        if mask.sum() == 0:
            continue
        proto = X[mask].mean(axis=0)
        for h in range(X.shape[1]):
            rows.append({
                "group": "PV_ALL",
                "split": split_name,
                "cluster": k,
                "slot": h,
                "value": float(proto[h])
            })
    tag = f"_dtw" if algo_tag == "dtw" else ""
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"prototypes_pv{tag}_{split_name}.csv"), index=False)

def predict_on_test_series_pv(model,
                              s30_test: pd.Series,
                              hour_start: str,
                              hour_end: str,
                              algo_tag: str = ""):
    """
    Apply a fitted PV clustering model to an external test series (30-min).
    Saves:
      - labels_pv[_dtw]_test.csv
      - prototypes_pv[_dtw]_test.csv
      - pv_test_counts[_dtw].png
    """
    if model is None:
        print("[PV TEST] No model provided, skipping.")
        return

    # Días presentes en TEST
    days_test = pd.Index(s30_test.index.normalize().unique()).sort_values()
    if len(days_test) == 0:
        print("[PV TEST] No days available in TEST.")
        return

    # Matriz diaria usando la MISMA ventana que el train
    dates_te, X_te = build_daily_matrix(
        s30_test, days_test, shape_mode=SHAPE_MODE,
        hour_start=hour_start, hour_end=hour_end
    )
    if X_te.shape[0] == 0:
        print("[PV TEST] No complete test days after windowing.")
        return

    # Predicción según algoritmo
    if algo_tag == "dtw":
        labels_te = predict_dtw(model, X_te)
        tag = "_dtw"
    else:
        labels_te = predict_kmeans(model, X_te)
        tag = ""

    # Guardar CSVs (respetando nombres *_pv_*)
    save_day_labels_csv(dates_te, labels_te, "test", algo_tag=algo_tag)
    save_prototypes_csv(X_te, labels_te, "test", algo_tag=algo_tag)

    # Gráfico de conteo por cluster
    counts_path = os.path.join(OUT_DIR, f"pv_test_counts{tag}.png")
    plot_cluster_counts(labels_te, title=f"PV TEST - counts ({algo_tag or 'euclid'})",
                        savepath=counts_path)

    print(f"[PV TEST] Saved labels/prototypes CSVs and counts figure: {counts_path}")

# =========================
# Main
# =========================
def main():
    print(">> Loading PV series and resampling...")
    s = load_series(CSV_PATH)
    s30 = to_30min(s)

    # Chronological day split
    days_train, days_val = split_train_val_by_days(s30, train_ratio=0.7)
    total_days = len(s30.index.normalize().unique())
    print(f"Total days: {total_days} | train: {len(days_train)} | val: {len(days_val)}")
    print(f"Time window used for clustering: {TRAIN_HOUR_START} -> {TRAIN_HOUR_END}")
    print(f"Clustering algorithm: {CLUSTER_ALGO}")

    # Build matrices for train/val (no weekday/weekend split for PV)
    dates_tr, X_tr = build_daily_matrix(
        s30, days_train, shape_mode=SHAPE_MODE,
        hour_start=TRAIN_HOUR_START, hour_end=TRAIN_HOUR_END
    )
    dates_va, X_va = build_daily_matrix(
        s30, days_val, shape_mode=SHAPE_MODE,
        hour_start=TRAIN_HOUR_START, hour_end=TRAIN_HOUR_END
    )

    print(f"[PV_ALL] complete days -> train={len(dates_tr)}, val={len(dates_va)}")
    if X_tr.shape[0] < 4:
        print("!! Not enough days for clustering (min 4 recommended). Aborting.")
        return

    if CLUSTER_ALGO == "euclid":
        # ---- Elbow & silhouette in TRAIN (PCA space) ----
        inertia, sil = evaluate_kmeans_for_K(X_tr, K_RANGE, random_state=RANDOM_STATE)
        elbow_k = suggest_k_elbow(inertia, list(K_RANGE))
        print(f"   Suggested K (elbow heuristic): {elbow_k}")

        prefix = os.path.join(OUT_DIR, "pv_train")
        plot_elbow_and_sil(K_RANGE, inertia, sil, "PV ALL (TRAIN)", prefix)

        # ---- Fit final model with suggested K ----
        model = fit_kmeans(X_tr, k=elbow_k, random_state=RANDOM_STATE)
        labels_tr = model["kmeans"].labels_
        save_day_labels_csv(dates_tr, labels_tr, "train", algo_tag="")
        save_prototypes_csv(X_tr, labels_tr, "train", algo_tag="")

        # Visualizations
        plot_pca_scatter(X_tr, model, labels_tr,
                         f"PV TRAIN - PCA by cluster (K={elbow_k})",
                         os.path.join(OUT_DIR, "pv_train_pca.png"))
        plot_cluster_prototypes(X_tr, labels_tr, TRAIN_HOUR_START,
                                f"PV TRAIN - Prototypes (K={elbow_k})",
                                os.path.join(OUT_DIR, "pv_train"))

        # ---- Predict clusters in VAL ----
        if X_va.shape[0] > 0:
            labels_va = predict_kmeans(model, X_va)
            save_day_labels_csv(dates_va, labels_va, "val", algo_tag="")
            save_prototypes_csv(X_va, labels_va, "val", algo_tag="")

        # ---- Predict clusters in TEST (EUCLID) ----
        if TEST_CSV_PATH is not None and os.path.exists(TEST_CSV_PATH):
            print(f">> Loading PV TEST series from {TEST_CSV_PATH} ...")
            s_test = load_series(TEST_CSV_PATH)
            s30_test = to_30min(s_test)
            predict_on_test_series_pv(model, s30_test,
                                      hour_start=TRAIN_HOUR_START,
                                      hour_end=TRAIN_HOUR_END,
                                      algo_tag="")
        else:
            print(">> PV TEST file not found or TEST_CSV_PATH unset. Skipping TEST.")


    elif CLUSTER_ALGO == "dtw":
        # Ensure tslearn is present
        if not _require_tslearn():
            return

        # ---- Elbow using DTW inertia ----
        inertia, models_by_k = evaluate_dtw_for_K(X_tr, K_RANGE, random_state=RANDOM_STATE)
        elbow_k = suggest_k_elbow(inertia, list(K_RANGE))
        print(f"   Suggested K (DTW elbow heuristic): {elbow_k}")

        prefix = os.path.join(OUT_DIR, "pv_train_dtw")
        plot_elbow_and_sil(K_RANGE, inertia, sil=None, title="PV ALL (TRAIN, DTW)", savepath_prefix=prefix)

        # Use the already-fitted model at elbow_k for efficiency
        model = models_by_k[elbow_k]
        labels_tr = model.labels_
        save_day_labels_csv(dates_tr, labels_tr, "train", algo_tag="dtw")
        save_prototypes_csv(X_tr, labels_tr, "train", algo_tag="dtw")

        # PCA scatter just for visualization (embedding), using labels from DTW
        plot_pca_scatter(X_tr, {}, labels_tr,
                         f"PV TRAIN (DTW) - PCA visualization (K={elbow_k})",
                         os.path.join(OUT_DIR, "pv_train_dtw_pca.png"))
        plot_cluster_prototypes(X_tr, labels_tr, TRAIN_HOUR_START,
                                f"PV TRAIN (DTW) - Prototypes (K={elbow_k})",
                                os.path.join(OUT_DIR, "pv_train_dtw"))

        # ---- Predict clusters in VAL (DTW) ----
        if X_va.shape[0] > 0:
            labels_va = predict_dtw(model, X_va)
            save_day_labels_csv(dates_va, labels_va, "val", algo_tag="dtw")
            save_prototypes_csv(X_va, labels_va, "val", algo_tag="dtw")

        # ---- Predict clusters in TEST (DTW) ----
        if TEST_CSV_PATH is not None and os.path.exists(TEST_CSV_PATH):
            print(f">> Loading PV TEST series from {TEST_CSV_PATH} ...")
            s_test = load_series(TEST_CSV_PATH)
            s30_test = to_30min(s_test)
            predict_on_test_series_pv(model, s30_test,
                                      hour_start=TRAIN_HOUR_START,
                                      hour_end=TRAIN_HOUR_END,
                                      algo_tag="dtw")
        else:
            print(">> PV TEST file not found or TEST_CSV_PATH unset. Skipping TEST.")

    else:
        print(f"!! Unknown CLUSTER_ALGO '{CLUSTER_ALGO}'. Use 'euclid' or 'dtw'.")

    print(">> Done. Check ./data/clustering_30min for plots and *_pv CSVs.")

if __name__ == "__main__":
    main()
