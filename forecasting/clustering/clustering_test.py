# --- ADD THESE IMPORTS NEAR YOUR OTHER IMPORTS ---
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def load_series(csv_path: str) -> pd.Series:
    """Load CSV into a pandas Series indexed by timestamp with float values."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return df

# ==========================
# PV: CLUSTER → CENTROIDS → 1-MIN PROFILES
# ==========================
def generate_pv_profiles(
    pv_5min_test: pd.DataFrame,
    value_col: str = None,          # set to your column (e.g., 'power'); if None, it tries to auto-detect
    out_dir: str = "PV Profiles",
    n_clusters: int = 5,
    resample_rule: str = "30T",     # 30 minutes for clustering
    base_freq: str = "5T",          # expected base freq (5 minutes)
    agg_for_30min: str = "mean",    # 'mean' if input is power; 'sum' if input is energy per 5 min
    convert_to_kWh_per_min: bool = False,  # if True and input is kW, output 'mult' in kWh/min
    clip_at_zero: bool = True,
    random_state: int = 42,
):
    """
    1) Take 5-min PV series and resample to 30-min.
    2) Cluster daily 30-min profiles into n_clusters (KMeans).
    3) For each cluster centroid (48 pts), upsample to 1-minute (1440 pts) via time interpolation.
    4) Save CSVs as 'PV_profile_1..n.csv' with columns: time,mult (00:01:00..24:00:00).

    Notes:
      - If your input is power in kW, keep agg_for_30min='mean'.
      - If your input is energy per 5 minutes (kWh/5min), use agg_for_30min='sum'.
      - Set convert_to_kWh_per_min=True if you want 'mult' in kWh/min (from kW).
    """
    if pv_5min_test is None or len(pv_5min_test) == 0:
        print("[PV] Empty DataFrame provided.")
        return

    # Ensure a datetime index
    df = pv_5min_test.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        # try common column names
        for cand in ["datetime", "time", "timestamp", "date"]:
            if cand in df.columns:
                df[cand] = pd.to_datetime(df[cand], errors="coerce")
                df = df.set_index(cand)
                break
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("[PV] Could not find a datetime index or a datetime-like column.")

    # Pick value column (auto-detect if not provided)
    if value_col is None:
        candidates = ["power", "pv", "pv_kW", "P", "value", "p_norm"]
        value_col = next((c for c in candidates if c in df.columns), None)
        if value_col is None:
            # fall back to the first numeric column
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                raise ValueError("[PV] No numeric column found to use as PV value.")
            value_col = num_cols[0]

    s = df[value_col].dropna().sort_index()

    # Optional: normalize to a strict base frequency (5T) if needed, then resample to 30T
    # For power, use mean; for energy windows, use sum.
    if agg_for_30min == "sum":
        s_30 = s.resample(resample_rule).sum()
    else:
        s_30 = s.resample(resample_rule).mean()

    # Build daily matrix with 48 columns (00:00..23:30)
    df30 = s_30.to_frame("val")
    df30["date"] = df30.index.normalize()
    df30["slot"] = df30.index.hour * 2 + (df30.index.minute // 30)  # 0..47
    mat = df30.pivot(index="date", columns="slot", values="val").sort_index()

    # Drop days with any missing slots
    mat = mat.dropna()
    if mat.empty:
        raise ValueError("[PV] No complete days after resampling to 30 minutes.")

    # KMeans clustering
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    km.fit(mat.values)  # shape: (n_days, 48)
    centers = km.cluster_centers_                            # shape: (n_clusters, 48)
    cluster_sizes = np.bincount(km.labels_, minlength=n_clusters)

    # Order profiles by total daily energy/power descending (nice, stable order 1..n)
    order = np.argsort(-centers.sum(axis=1))

    os.makedirs(out_dir, exist_ok=True)

    # Helper: upsample one centroid (48 @ 30min) to 1440 @ 1min
    def _upsample_centroid_to_1min(center_48: np.ndarray) -> pd.DataFrame:
        ref0 = pd.Timestamp("2000-01-01 00:00:00")
        # 30-min index with an extra point at 24:00 to allow interpolation to end-of-day
        idx_30 = pd.date_range(ref0, ref0 + pd.Timedelta(days=1), freq="30T")  # 49 stamps (incl. 24:00)
        vals_49 = np.concatenate([center_48, center_48[:1]])  # repeat first point at 24:00
        s30 = pd.Series(vals_49, index=idx_30)

        # 1-min index for full day including 00:00 and 24:00 (1441 stamps)
        idx_1 = pd.date_range(ref0, ref0 + pd.Timedelta(days=1), freq="T")
        s1 = s30.reindex(idx_1).interpolate(method="time")

        if clip_at_zero:
            s1 = s1.clip(lower=0.0)

        # If converting kW → kWh/min, do it now
        vals = s1.values
        if convert_to_kWh_per_min:
            vals = vals / 60.0

        # Build final dataframe: 00:01:00..24:00:00 (skip 00:00)
        times = [f"{m//60:02d}:{m%60:02d}:00" for m in range(1, 1441)]
        df_out = pd.DataFrame({"time": times, "mult": vals[1:]})  # drop 00:00
        return df_out

    # Save PV_profile_1..n.csv (1 = highest “energy” centroid)
    for rank, idx in enumerate(order, start=1):
        centroid = centers[idx]
        prof_df = _upsample_centroid_to_1min(centroid)
        out_csv = os.path.join(out_dir, f"PV_profile_{rank}.csv")
        prof_df.to_csv(out_csv, index=False, float_format="%.3f")
        print(f"[PV] Saved {out_csv}  (cluster size = {cluster_sizes[idx]})")

pv_5min_test = load_series("data/pv_5min_test.csv")
# ======= CALL IT (example) =======
# If your value column is, say, 'power', set value_col='power'.
# If you want 'mult' to be energy per minute (kWh/min) from kW input, set convert_to_kWh_per_min=True.
generate_pv_profiles(
    pv_5min_test,
    value_col=None,                 # auto-detect; change to your column if you prefer
    out_dir="PV Profiles",
    n_clusters=5,
    resample_rule="30T",
    agg_for_30min="mean",           # use 'sum' if your input is energy per 5-min
    convert_to_kWh_per_min=False,   # set True if you want kWh/min like tus EV perfiles
    clip_at_zero=True,
    random_state=42,
)

# ==========================
# PLOT: PV_profile_1..5 en una sola figura y guardar PNG
# ==========================
pv_dir = "PV Profiles"
profile_paths = [os.path.join(pv_dir, f"PV_profile_{i}.csv") for i in range(1, 6)]
profiles = []

for i, p in enumerate(profile_paths, start=1):
    if not os.path.exists(p):
        continue
    dfp = pd.read_csv(p)

    # Convert "time" (HH:MM:SS) → minuto del día (1..1440)
    tmin = (pd.to_timedelta(dfp["time"]).dt.total_seconds() // 60).astype(int).values

    # 'mult' según tus perfiles (puede ser kWh/min si activaste la conversión)
    y = dfp["mult"].astype(float).values

    # OPCIONAL: si tus 'mult' son potencias en kW y quieres graficar energía por minuto:
    # y = y / 60.0

    profiles.append((i, tmin, y))

if profiles:
    plt.figure(figsize=(10, 5))
    for i, tmin, y in profiles:
        plt.plot(tmin, y, label=f"PV_profile_{i}")
    plt.xlabel("Minute of day")
    plt.ylabel("mult")  # cambia a 'kWh per minute' si corresponde a tu configuración
    plt.title("PV Profiles (cluster centroids, 1-min)")
    plt.xlim(1, 1440)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()

    out_png = os.path.join(pv_dir, "PV_profiles_overlay.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[PV] Plot saved: {out_png}")
else:
    print("[PV] No PV_profile_X.csv files found to plot.")
