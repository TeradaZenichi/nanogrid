"""Stochastic plan sensitivity to the re-solve interval.

Sweeps resolve_every_h over four outage realizations, including the two
seeds where the 24h plan blew up. Outputs go to Results/10-recourse/.
"""

import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opt import simulate_stochastic

N_ITERS = 2880
OUT_ROOT = "Results/10-recourse"
PARAMS_JSON = "data/parameters.json"

FREQS_H = [24, 12, 6, 3, 1]

# (name, window start, EDS.seed); may/42 and may/44 are the blow-up cases.
WINDOWS = [
    ("may_seed42", pd.Timestamp("2009-05-01 00:00:00"), 42),
    ("may_seed44", pd.Timestamp("2009-05-01 00:00:00"), 44),
    ("aug_seed42", pd.Timestamp("2009-08-01 00:00:00"), 42),
    ("nov_seed42", pd.Timestamp("2009-11-01 00:00:00"), 42),
]

if __name__ == "__main__":
    base = json.load(open(PARAMS_JSON, encoding="utf-8"))

    results = []
    for name, start, seed in WINDOWS:
        params = deepcopy(base)
        params["EDS"]["seed"] = seed
        for freq in FREQS_H:
            case = f"{name}_r{freq}h"
            print(f"===== {case} =====")
            m = simulate_stochastic(params, start, N_ITERS, f"{OUT_ROOT}/{case}",
                                    resolve_every_h=float(freq))
            results.append({**m, "window": name, "seed": seed})

    summary = pd.DataFrame(results)
    summary.to_csv(f"{OUT_ROOT}/summary.csv", index=False)
    print(summary.pivot_table(index="window", columns="resolve_every_h",
                              values="operation_total_cost").round(1).to_string())
    print(f"\nsummary: {OUT_ROOT}/summary.csv")
