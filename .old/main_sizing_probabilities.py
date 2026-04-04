from pathlib import Path

import pandas as pd


def _load_labels(path: Path, cluster_col_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "cluster", "group", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out["cluster"] = out["cluster"].astype(int)
    out = out.rename(columns={"cluster": cluster_col_name})
    return out


def _marginal_probabilities(
    df: pd.DataFrame,
    cluster_col: str,
    group_col: str,
    split_col: str = "split",
) -> pd.DataFrame:
    counts = (
        df.groupby([group_col, split_col, cluster_col], as_index=False)
        .size()
        .rename(columns={"size": "n_days"})
    )
    totals = counts.groupby([group_col, split_col], as_index=False)["n_days"].sum()
    totals = totals.rename(columns={"n_days": "n_total"})
    out = counts.merge(totals, on=[group_col, split_col], how="left")
    out["probability"] = out["n_days"] / out["n_total"]
    return out.sort_values([group_col, split_col, cluster_col]).reset_index(drop=True)


def _joint_probabilities(
    load_labels: pd.DataFrame,
    pv_labels: pd.DataFrame,
) -> pd.DataFrame:
    merged = load_labels.merge(
        pv_labels,
        on=["date", "split"],
        how="inner",
        suffixes=("_load", "_pv"),
    )
    if merged.empty:
        raise ValueError("No overlapping dates between load and PV labels.")

    counts = (
        merged.groupby(
            ["split", "group_load", "group_pv", "cluster_load", "cluster_pv"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_days"})
    )

    base = (
        merged[["split", "group_load", "group_pv"]]
        .drop_duplicates()
        .sort_values(["split", "group_load", "group_pv"])
        .reset_index(drop=True)
    )
    load_clusters = (
        merged[["split", "group_load", "cluster_load"]]
        .drop_duplicates()
        .sort_values(["split", "group_load", "cluster_load"])
        .reset_index(drop=True)
    )
    pv_clusters = (
        merged[["split", "group_pv", "cluster_pv"]]
        .drop_duplicates()
        .sort_values(["split", "group_pv", "cluster_pv"])
        .reset_index(drop=True)
    )

    full_pairs = (
        base.merge(load_clusters, on=["split", "group_load"], how="left")
        .merge(pv_clusters, on=["split", "group_pv"], how="left")
    )

    out = full_pairs.merge(
        counts,
        on=["split", "group_load", "group_pv", "cluster_load", "cluster_pv"],
        how="left",
    )
    out["n_days"] = out["n_days"].fillna(0).astype(int)

    totals = (
        merged.groupby(["split", "group_load", "group_pv"], as_index=False)
        .size()
        .rename(columns={"size": "n_total"})
    )
    out = out.merge(totals, on=["split", "group_load", "group_pv"], how="left")
    out["probability"] = out["n_days"] / out["n_total"]
    return out.sort_values(
        ["split", "group_load", "group_pv", "cluster_load", "cluster_pv"]
    ).reset_index(drop=True)


if __name__ == "__main__":
    labels_pv_path = Path("data/sizing/labels_pv_dtw_train.csv")
    labels_load_path = Path("data/sizing/labels_load_dtw_all_train.csv")
    out_dir = Path("data/sizing")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_pv = _load_labels(labels_pv_path, "cluster_pv")
    labels_load = _load_labels(labels_load_path, "cluster_load")

    prob_load = _marginal_probabilities(labels_load, "cluster_load", "group")
    prob_pv = _marginal_probabilities(labels_pv, "cluster_pv", "group")
    prob_joint = _joint_probabilities(labels_load, labels_pv)

    prob_load.to_csv(out_dir / "prob_load.csv", index=False)
    prob_pv.to_csv(out_dir / "prob_pv.csv", index=False)
    prob_joint.to_csv(out_dir / "prob_joint_load_pv.csv", index=False)

    print(f"Saved: {out_dir / 'prob_load.csv'}")
    print(f"Saved: {out_dir / 'prob_pv.csv'}")
    print(f"Saved: {out_dir / 'prob_joint_load_pv.csv'}")
