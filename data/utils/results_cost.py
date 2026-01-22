import sys
import numpy as np
import pandas as pd

# === EDIT HERE ===
# TYPE = "Ideal_36h"  # "ideal" ou "real"
TYPE = "Real_36h"  # "ideal" ou "real"
CSV_PATH = f"outputs/{TYPE}/operation_real.csv"  # <-- put your CSV path here
NDIGITS = 6                                      # rounding for printed numbers
# ================

COST_COLS = ["cost_grid", "cost_shed", "cost_curt", "cost_total", "obj"]
COMPONENTS = ["cost_grid", "cost_shed", "cost_curt"]

def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=None, engine="python", encoding="latin-1")

def coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series, errors="coerce")

def main():
    try:
        df = load_csv(CSV_PATH)
    except Exception as e:
        print(f"Error reading CSV at '{CSV_PATH}': {e}", file=sys.stderr)
        sys.exit(1)

    totals = {}
    present, missing = [], []
    for col in COST_COLS:
        if col in df.columns:
            s = coerce_numeric(df[col])
            totals[col] = float(np.nansum(s.values))
            present.append(col)
        else:
            missing.append(col)

    comp_total = float(sum(totals[c] for c in COMPONENTS if c in totals))

    total_from_col = totals.get("cost_total", None)
    if total_from_col is not None:
        final_total = total_from_col
        total_source = "cost_total column"
    else:
        final_total = comp_total
        total_source = "sum of components (cost_grid + cost_shed + cost_curt)"

    # Print
    rd = NDIGITS
    print("=== TOTALS (sum over rows) ===")
    for col in COST_COLS:
        if col in totals:
            print(f"- {col}: {round(totals[col], rd)}")
        else:
            print(f"- {col}: (missing)")

    print(f"\nComponents total (grid + shed + curt): {round(comp_total, rd)}")
    if total_from_col is not None:
        print(f"Total (from cost_total): {round(total_from_col, rd)}")
        print(f"Discrepancy (cost_total - components): {round(total_from_col - comp_total, rd)}")

    print(f"\nFINAL TOTAL = {round(final_total, rd)}  [source: {total_source}]")

if __name__ == "__main__":
    main()
