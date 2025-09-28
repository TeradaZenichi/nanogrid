# -*- coding: utf-8 -*-
"""
Daily-profile clustering (30-min) for LOAD.

What this script does
---------------------
- Reads load_5min_train.csv with columns: [timestamp, p_norm].
- Resamples to 30-min means and builds full-day (00:00→24:00) day-by-row matrices;
  each row is one day's vector (no start/end hour window for LOAD).
- Chronological split: 70% train / 30% val by whole days.
- Two clustering options:
  (A) PCA + KMeans using Euclidean metric (fast, baseline)
  (B) TimeSeriesKMeans with DTW from tslearn (shape-aware)
- For (A) computes elbow (inertia) and silhouette over K=2..8 in PCA space.
- For (B) computes elbow (inertia) only (silhouette w/ DTW omitted).
- Supports two grouping modes:
  1) "all": cluster all days together (single model)
  2) "split": cluster WEEKDAY vs WEEKEND separately; optionally add “international
     holidays” (fixed-date list) to WEEKEND via a flag.
- Saves artifacts with clear *load* names and a group tag:
    labels_load_{group}_{split}.csv
    prototypes_load_{group}_{split}.csv
    load_{group}_{split}_elbow.png
    load_{group}_{split}_silhouette.png (Euclidean only)
    load_{group}_{split}_pca.png
    load_{group}_{split}_prototypes.png
    (DTW adds *_dtw_* file names too)

Notes
-----
- Input expects two columns: 'timestamp' (parseable datetime) and 'p_norm' (float).
- If you prefer a different column name for load, adjust `VALUE_COL`.
- DTW mode requires: pip install tslearn

Author: Adapted for LOAD based on the PV script
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
CSV_PATH = "data/load_5min_train.csv"          # Input LOAD file: [timestamp, p_norm]
TEST_CSV_PATH = "data/load_5min_test.csv"  
VALUE_COL = "p_norm"                            # Change if your column differs
OUT_DIR  = "data/clustering_30min_load"
os.makedirs(OUT_DIR, exist_ok=True)

RESAMPLE_RULE = "30min"                         # 30-minute profiles
K_RANGE = range(2, 9)                           # K in [2..8]
SHAPE_MODE = "none"                             # "none" or "zscore_by_day"
RANDOM_STATE = 42

# Clustering algorithm: "euclid" (PCA+KMeans) or "dtw" (tslearn TimeSeriesKMeans)
CLUSTER_ALGO = "dtw"                         # change to "dtw" if you want DTW

# Grouping: "all" (everything together) or "split" (weekday vs weekend)
GROUPING_MODE = "all"                           # "all" or "split"

# If GROUPING_MODE == "split", decide whether to include holidays as weekend:
INCLUDE_HOLIDAYS_AS_WEEKEND = True

# International (fixed-date) holidays to consider for ALL dataset years.
# Format "MM-DD". This is a minimal, general list; extend if needed.
HOLIDAYS_MM_DD = [
    "01-01",  # New Year's Day
    "05-01",  # Labour Day / May Day
    "12-25",  # Christmas Day
    "12-26",  # Boxing / St Stephen's Day
]

# =========================
# I/O helpers
# =========================
def load_series(csv_path: str, value_col: str) -> pd.Series:
    """Load CSV into a pandas Series indexed by timestamp with float values."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found. Available: {list(df.columns)}")
    s = df[value_col].astype(float).copy()
    return s

def to_30min(s_5min: pd.Series) -> pd.Series:
    """Resample 5-min series to 30-min mean values."""
    return s_5min.resample(RESAMPLE_RULE, label="right", closed="right").mean()

def split_train_val_by_days(days_index: pd.Index, train_ratio: float = 0.7):
    """Chronological day split without mixing: returns (days_train, days_val)."""
    days = pd.Index(days_index).sort_values()
    n_days = len(days)
    cut = int(math.floor(n_days * train_ratio))
    return days[:cut], days[cut:]

# =========================
# Holidays & grouping
# =========================
def build_holiday_dates(all_years: list[int], mmdd_list: list[str]) -> set[pd.Timestamp]:
    """Build a set of holiday dates (00:00) for the years present in the data."""
    hol = set()
    for y in all_years:
        for mmdd in mmdd_list:
            month, day = map(int, mmdd.split("-"))
            hol.add(pd.Timestamp(year=y, month=month, day=day))
    return hol

def split_weekday_weekend(days_index: pd.Index,
                          include_holidays_as_weekend: bool,
                          holiday_dates: set[pd.Timestamp]):
    """Return (weekday_days, weekend_days) DatetimeIndex based on Mon..Sun (+holidays if flagged)."""
    wd, we = [], []
    for d in pd.Index(days_index).sort_values():
        dow = d.dayofweek  # Mon=0 .. Sun=6
        is_weekend = (dow >= 5)
        is_holiday = (d.normalize() in holiday_dates)
        if include_holidays_as_weekend and is_holiday:
            we.append(d)
        elif is_weekend:
            we.append(d)
        else:
            wd.append(d)
    return pd.DatetimeIndex(wd), pd.DatetimeIndex(we)

# =========================
# Matrices (full day)
# =========================
def build_daily_matrix_full_day(
    s_30: pd.Series,
    days: pd.Index,
    shape_mode: str = "none",
):
    """
    Build a day-by-row matrix for the full day [00:00, 24:00).

    Returns
    -------
    dates : pd.DatetimeIndex
        Dates for which a complete full-day window exists with no NaNs.
    X : np.ndarray, shape (n_days, n_slots)
        One row per day, columns are 30-min slots (48 columns for 24h at 30-min).
    """
    rows = []
    keep_dates = []
    nslots_target = None

    for d in days:
        # Select the day [d, d+1)
        day_slice = s_30.loc[(s_30.index >= d) & (s_30.index < d + pd.Timedelta(days=1))]
        if day_slice.empty or day_slice.isna().all():
            continue

        if nslots_target is None:
            nslots_target = day_slice.shape[0]

        if len(day_slice) == nslots_target and not day_slice.isna().any():
            x = day_slice.values.astype(float)
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

def _time_axis_ticks_full_day(nslots: int):
    """
    Build x-ticks & labels for full day given slot count (30-min slots expected).
    If nslots is multiple of 2, tick per hour; else ~12 ticks.
    """
    if nslots % 2 == 0:
        slots_per_hour = 2
        ticks = np.arange(0, nslots, slots_per_hour)
        labels = [f"{h:02d}:00" for h in range(len(ticks))]
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
        return True
    except Exception as e:
        print(">> DTW requested but tslearn is not available. Install with: pip install tslearn")
        print(f"   Detail: {e}")
        return False

def evaluate_dtw_for_K(X: np.ndarray, k_list, random_state=RANDOM_STATE):
    """
    Evaluate DTW clustering across K. Returns (inertia_list, models_dict).
    Silhouette for DTW is omitted; inertia_ (within-cluster sum) is used.
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
def plot_elbow_and_sil(k_list, inertia, sil, title, savepath_prefix, algo_tag=""):
    tag = "_dtw" if algo_tag == "dtw" else ""
    # Elbow
    plt.figure()
    plt.plot(list(k_list), inertia, marker="o")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.title(f"Elbow - {title}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{savepath_prefix}{tag}_elbow.png", dpi=140)
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
        plt.savefig(f"{savepath_prefix}_silhouette.png", dpi=140)
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

def plot_cluster_prototypes(X, labels, title, savepath_prefix):
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
    ticks, labels_x = _time_axis_ticks_full_day(nslots)
    plt.xticks(ticks=ticks, labels=labels_x, rotation=0)
    plt.xlabel("Hour of day (00–24)")
    plt.ylabel("p_norm (30-min)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{savepath_prefix}_prototypes.png", dpi=140)
    plt.close()

def plot_cluster_counts(labels: np.ndarray, title: str, savepath: str):
    """Save a bar chart with number of samples per cluster."""
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

def save_day_labels_csv(dates, labels, split_name, group_tag, algo_tag=""):
    tag = "_dtw" if algo_tag == "dtw" else ""
    out = pd.DataFrame({"date": dates, "cluster": labels})
    out["group"] = f"LOAD_{group_tag.upper()}"
    out["split"] = split_name
    out.to_csv(os.path.join(OUT_DIR, f"labels_load{tag}_{group_tag}_{split_name}.csv"), index=False)

def save_prototypes_csv(X, labels, split_name, group_tag, algo_tag=""):
    tag = "_dtw" if algo_tag == "dtw" else ""
    K = int(labels.max() + 1)
    rows = []
    for k in range(K):
        mask = labels == k
        if mask.sum() == 0:
            continue
        proto = X[mask].mean(axis=0)
        for h in range(X.shape[1]):
            rows.append({
                "group": f"LOAD_{group_tag.upper()}",
                "split": split_name,
                "cluster": k,
                "slot": h,
                "value": float(proto[h])
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_DIR, f"prototypes_load{tag}_{group_tag}_{split_name}.csv"), index=False
    )

# =========================
# Pipelines
# =========================
def run_clustering_on_matrix(X_tr, dates_tr, X_va, dates_va, group_tag: str):
    """Run either Euclidean or DTW clustering for a given group and save artifacts."""
    if X_tr.shape[0] < 4:
        print(f"!! [{group_tag}] Not enough days for clustering (min 4 recommended). Skipping.")
        return

    if CLUSTER_ALGO == "euclid":
        # ---- Elbow & silhouette in TRAIN (PCA space) ----
        inertia, sil = evaluate_kmeans_for_K(X_tr, K_RANGE, random_state=RANDOM_STATE)
        elbow_k = suggest_k_elbow(inertia, list(K_RANGE))
        print(f"   [{group_tag}] Suggested K (elbow heuristic): {elbow_k}")

        prefix = os.path.join(OUT_DIR, f"load_{group_tag}_train")
        plot_elbow_and_sil(K_RANGE, inertia, sil,
                           f"LOAD {group_tag.upper()} (TRAIN)",
                           savepath_prefix=prefix, algo_tag="")

        # ---- Fit final model with suggested K ----
        model = fit_kmeans(X_tr, k=elbow_k, random_state=RANDOM_STATE)
        labels_tr = model["kmeans"].labels_
        save_day_labels_csv(dates_tr, labels_tr, "train", group_tag, algo_tag="")
        save_prototypes_csv(X_tr, labels_tr, "train", group_tag, algo_tag="")

        # Visualizations
        plot_pca_scatter(X_tr, model, labels_tr,
                         f"LOAD TRAIN - PCA by cluster ({group_tag}, K={elbow_k})",
                         os.path.join(OUT_DIR, f"load_{group_tag}_train_pca.png"))
        plot_cluster_prototypes(X_tr, labels_tr,
                                f"LOAD TRAIN - Prototypes ({group_tag}, K={elbow_k})",
                                os.path.join(OUT_DIR, f"load_{group_tag}_train"))

        # ---- Predict clusters in VAL ----
        if X_va.shape[0] > 0:
            labels_va = predict_kmeans(model, X_va)
            save_day_labels_csv(dates_va, labels_va, "val", group_tag, algo_tag="")
            save_prototypes_csv(X_va, labels_va, "val", group_tag, algo_tag="")
        
        return model # return the model for test set prediction

    elif CLUSTER_ALGO == "dtw":
        # Ensure tslearn is present
        if not _require_tslearn():
            return

        # ---- Elbow using DTW inertia ----
        inertia, models_by_k = evaluate_dtw_for_K(X_tr, K_RANGE, random_state=RANDOM_STATE)
        elbow_k = suggest_k_elbow(inertia, list(K_RANGE))
        print(f"   [{group_tag}] Suggested K (DTW elbow heuristic): {elbow_k}")

        prefix = os.path.join(OUT_DIR, f"load_{group_tag}_train")
        plot_elbow_and_sil(K_RANGE, inertia, sil=None,
                           title=f"LOAD {group_tag.upper()} (TRAIN, DTW)",
                           savepath_prefix=prefix, algo_tag="dtw")

        # Use the already-fitted model at elbow_k for efficiency
        model = models_by_k[elbow_k]
        labels_tr = model.labels_
        save_day_labels_csv(dates_tr, labels_tr, "train", group_tag, algo_tag="dtw")
        save_prototypes_csv(X_tr, labels_tr, "train", group_tag, algo_tag="dtw")

        # PCA scatter just for visualization (embedding), using labels from DTW
        plot_pca_scatter(X_tr, {}, labels_tr,
                         f"LOAD TRAIN (DTW) - PCA visualization ({group_tag}, K={elbow_k})",
                         os.path.join(OUT_DIR, f"load_{group_tag}_train_dtw_pca.png"))
        plot_cluster_prototypes(X_tr, labels_tr,
                                f"LOAD TRAIN (DTW) - Prototypes ({group_tag}, K={elbow_k})",
                                os.path.join(OUT_DIR, f"load_{group_tag}_train_dtw"))

        # ---- Predict clusters in VAL (DTW) ----
        if X_va.shape[0] > 0:
            labels_va = predict_dtw(model, X_va)
            save_day_labels_csv(dates_va, labels_va, "val", group_tag, algo_tag="dtw")
            save_prototypes_csv(X_va, labels_va, "val", group_tag, algo_tag="dtw")
        
        return model # return the model for test set prediction

    else:
        print(f"!! Unknown CLUSTER_ALGO '{CLUSTER_ALGO}'. Use 'euclid' or 'dtw'.")

def predict_on_test_series(model,
                           s30_test: pd.Series,
                           group_tag: str,
                           days_subset: pd.Index | None = None):
    """Apply fitted model to an external test series. Saves labels/prototypes/counts plot."""
    if model is None:
        print(f"[TEST {group_tag}] No model provided, skipping.")
        return

    # Si se pide un subconjunto (weekday/weekend), úsalo; si no, usa todos los días presentes.
    if days_subset is None:
        days_test = pd.Index(s30_test.index.normalize().unique()).sort_values()
    else:
        # Asegura que existan en el índice
        all_days = pd.Index(s30_test.index.normalize().unique())
        days_test = pd.DatetimeIndex([d for d in days_subset if d in set(all_days)])

    if len(days_test) == 0:
        print(f"[TEST {group_tag}] No days available after filtering.")
        return

    dates_te, X_te = build_daily_matrix_full_day(s30_test, days_test, shape_mode=SHAPE_MODE)
    if X_te.shape[0] == 0:
        print(f"[TEST {group_tag}] No complete test days with full 30-min window.")
        return

    # Predicción según algoritmo
    if CLUSTER_ALGO == "euclid":
        labels_te = predict_kmeans(model, X_te)
        algo_tag = ""
    elif CLUSTER_ALGO == "dtw":
        labels_te = predict_dtw(model, X_te)
        algo_tag = "dtw"
    else:
        print(f"[TEST {group_tag}] Unknown CLUSTER_ALGO '{CLUSTER_ALGO}'.")
        return

    # Guardar CSVs y figura
    save_day_labels_csv(dates_te, labels_te, "test", group_tag, algo_tag=algo_tag)
    save_prototypes_csv(X_te, labels_te, "test", group_tag, algo_tag=algo_tag)

    counts_path = os.path.join(OUT_DIR, f"load_{group_tag}_test_counts.png")
    plot_cluster_counts(labels_te,
                        title=f"LOAD TEST - counts ({group_tag})",
                        savepath=counts_path)
    print(f"[TEST {group_tag}] Saved: labels/prototypes and {counts_path}")

# =========================
# Main
# =========================
def main():
    print(">> Loading LOAD series and resampling to 30-min...")
    s5 = load_series(CSV_PATH, VALUE_COL)
    s30 = to_30min(s5)

    # Days present in the TRAIN/VAL data
    all_days = pd.Index(s30.index.normalize().unique()).sort_values()
    total_days = len(all_days)
    years = sorted(pd.Index(s30.index.year).unique())
    holiday_dates = build_holiday_dates(years, HOLIDAYS_MM_DD)

    print(f"Total days: {total_days}")
    print(f"Clustering algorithm: {CLUSTER_ALGO}")
    print(f"Grouping mode: {GROUPING_MODE}")

    # --- (A) Entrenar y guardar modelos por grupo ---
    models_by_group = {}

    if GROUPING_MODE == "all":
        days_train, days_val = split_train_val_by_days(all_days, train_ratio=0.7)
        dates_tr, X_tr = build_daily_matrix_full_day(s30, days_train, shape_mode=SHAPE_MODE)
        dates_va, X_va = build_daily_matrix_full_day(s30, days_val,   shape_mode=SHAPE_MODE)
        print(f"[LOAD_ALL] complete days -> train={len(dates_tr)}, val={len(dates_va)}")
        model_all = run_clustering_on_matrix(X_tr, dates_tr, X_va, dates_va, group_tag="all")
        models_by_group["all"] = model_all

    elif GROUPING_MODE == "split":
        wd_days, we_days = split_weekday_weekend(all_days, INCLUDE_HOLIDAYS_AS_WEEKEND, holiday_dates)

        # WEEKDAY
        if len(wd_days) > 0:
            days_train, days_val = split_train_val_by_days(wd_days, train_ratio=0.7)
            dates_tr, X_tr = build_daily_matrix_full_day(s30, days_train, shape_mode=SHAPE_MODE)
            dates_va, X_va = build_daily_matrix_full_day(s30, days_val,   shape_mode=SHAPE_MODE)
            print(f"[LOAD_WEEKDAY] complete days -> train={len(dates_tr)}, val={len(dates_va)}")
            models_by_group["weekday"] = run_clustering_on_matrix(X_tr, dates_tr, X_va, dates_va, group_tag="weekday")
        else:
            print("[LOAD_WEEKDAY] No weekday days available after filtering.")
            models_by_group["weekday"] = None

        # WEEKEND (+ holidays if flagged)
        if len(we_days) > 0:
            days_train, days_val = split_train_val_by_days(we_days, train_ratio=0.7)
            dates_tr, X_tr = build_daily_matrix_full_day(s30, days_train, shape_mode=SHAPE_MODE)
            dates_va, X_va = build_daily_matrix_full_day(s30, days_val,   shape_mode=SHAPE_MODE)
            suffix = "weekend_holiday" if INCLUDE_HOLIDAYS_AS_WEEKEND else "weekend"
            print(f"[LOAD_{suffix.upper()}] complete days -> train={len(dates_tr)}, val={len(dates_va)}")
            models_by_group[suffix] = run_clustering_on_matrix(X_tr, dates_tr, X_va, dates_va, group_tag=suffix)
        else:
            print("[LOAD_WEEKEND] No weekend/holiday days available after filtering.")
            suffix = "weekend_holiday" if INCLUDE_HOLIDAYS_AS_WEEKEND else "weekend"
            models_by_group[suffix] = None

    else:
        print(f"!! Unknown GROUPING_MODE '{GROUPING_MODE}'. Use 'all' or 'split'.")
        return

    # --- (B) Si existe TEST_CSV_PATH, cargar y predecir ---
    test_s30 = None
    if TEST_CSV_PATH is not None and os.path.exists(TEST_CSV_PATH):
        print(f">> Loading TEST series from {TEST_CSV_PATH} ...")
        s5_test = load_series(TEST_CSV_PATH, VALUE_COL)
        test_s30 = to_30min(s5_test)

        test_years = sorted(pd.Index(test_s30.index.year).unique())
        holiday_dates_test = build_holiday_dates(test_years, HOLIDAYS_MM_DD)

        if GROUPING_MODE == "all":
            predict_on_test_series(models_by_group.get("all"), test_s30, group_tag="all")
        else:
            # split: calcular subconjuntos para el TEST también
            test_days_all = pd.Index(test_s30.index.normalize().unique()).sort_values()
            wd_test, we_test = split_weekday_weekend(test_days_all,
                                                     INCLUDE_HOLIDAYS_AS_WEEKEND,
                                                     holiday_dates_test)

            predict_on_test_series(models_by_group.get("weekday"),
                                   test_s30, group_tag="weekday", days_subset=wd_test)

            suffix = "weekend_holiday" if INCLUDE_HOLIDAYS_AS_WEEKEND else "weekend"
            predict_on_test_series(models_by_group.get(suffix),
                                   test_s30, group_tag=suffix, days_subset=we_test)
    else:
        print(">> TEST_CSV_PATH not provided or file not found. Skipping TEST prediction.")

    print(f">> Done. Check ./{OUT_DIR} for plots and CSVs (train/val/test).")

if __name__ == "__main__":
    main()
