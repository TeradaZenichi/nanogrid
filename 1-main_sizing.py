import json
from pathlib import Path

import pandas as pd
from sizing import (
    MicrogridDesign,
    interpolate_pv_scenarios,
    normalize_scenario_values,
)


if __name__ == "__main__":
    pv_csv_path = Path("data/sizing/pv_scenarios.csv")
    load_csv_path = Path("data/sizing/load_scenarios.csv")

    df_pv_raw = pd.read_csv(pv_csv_path)
    df_load = pd.read_csv(load_csv_path)

    # Normalize directly in source CSV files.
    df_pv_raw = normalize_scenario_values(df_pv_raw)
    df_load = normalize_scenario_values(df_load)
    df_pv_raw.to_csv(pv_csv_path, index=False)
    df_load.to_csv(load_csv_path, index=False)

    load_slots = int(df_load["slot"].nunique())
    df_pv = interpolate_pv_scenarios(df_pv_raw, target_slots=load_slots)
    df_pv.to_csv(pv_csv_path, index=False)

    out_dir = Path("outputs/sizing")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_pv.to_csv(out_dir / "pv_scenarios_interpolated.csv", index=False)

    with open("data/parameters.json", "r", encoding="utf-8") as f:
        params = json.load(f)

    design = MicrogridDesign(params, df_pv, df_load)
    design.build()
    design.optimize()
