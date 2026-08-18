"""Generate the paper's booktabs tables from result summaries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "outputs"
RES = SHARED / "sweeps" / "economic"
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def latex_table(caption: str, label: str, colspec: str,
                header_rows: list[str], body_rows: list[list[str]]) -> str:
    lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", f"\\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    lines += header_rows
    lines.append(r"\midrule")
    lines += [" & ".join(r) + r" \\" for r in body_rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def t1_forecast() -> str:
    s = json.load(open(
        SHARED / "forecasting-causal-v2" / "prototype_eval_summary.json",
        encoding="utf-8",
    ))
    mae = s["load_mae_5min_by_method"]
    skill = s["load_skill_vs_seasonal_60min"]
    order = ["lstm", "prototype-calendar", "prototype-prefix", "prototype-knn",
             "seasonal-naive", "persistence"]
    labels = {"lstm": "LSTM", "prototype-calendar": "Analog-day (calendar)",
              "prototype-prefix": "Analog-day (prefix)", "prototype-knn": "Analog-day (kNN)",
              "seasonal-naive": "Seasonal-naive", "persistence": "Persistence"}
    rows = []
    for m in order:
        if m not in mae:
            continue
        sk = skill.get(m, float("nan"))
        rows.append([labels[m], f"{mae[m]:.3f}",
                     "0" if abs(sk) < 1e-9 else f"{sk:+.2f}"])
    return latex_table(
        r"Load forecast quality over the test year (8{,}761 origins). "
        r"MAE at 5-min resolution and skill vs.\ the seasonal-naive baseline at 60-min blocks.",
        "tab:forecast", "lcc",
        [r"Forecaster & MAE [kW] & Skill vs.\ naive \\"], rows)


def t2_controllers() -> str:
    df = pd.read_csv(RES / "03-forecast-operation" / "summary.csv")
    df = df[df["month"] == "2009-05"]
    cost = df.set_index("forecaster")["operation_total_cost"].to_dict()
    ideal = cost["ideal"]
    order = [("ideal", "MPC, perfect forecast (oracle)"),
             ("lstm", "MPC + LSTM"),
             ("prototype-prefix", "MPC + analog-day (prefix)")]
    rows = []
    for key, lbl in order:
        if key not in cost:
            continue
        c = cost[key]
        ratio = "1.00$\\times$ (ref.)" if key == "ideal" else f"{c / ideal:.2f}$\\times$"
        rows.append([lbl, f"{c:.1f}", ratio])
    return latex_table(
        r"Closed-loop operating cost over a 10-day window (May), same outage calendar. "
        r"Ratio is relative to the perfect-forecast oracle; the oracle--LSTM gap is the EVPI.",
        "tab:controllers", "lcc",
        [r"Controller & Cost [USD] & vs.\ oracle \\"], rows)


def t3_monthly() -> str:
    s = pd.read_csv(RES / "03-forecast-operation" / "summary.csv")
    s["ctrl"] = s["forecaster"]
    piv = s.pivot_table(index="month", columns="ctrl",
                        values="operation_total_cost", aggfunc="first").sort_index()
    cols = ["ideal", "lstm", "prototype-prefix"]
    rows = []
    for month in piv.index:
        r = piv.loc[month]
        rows.append([month.replace("-", "--")] + [f"{r[c]:.1f}" for c in cols])
    tot = piv[cols].sum()
    rows.append([r"\textbf{Annual}"] + [f"\\textbf{{{tot[c]:.0f}}}" for c in cols])
    return latex_table(
        r"Operating cost [USD] by controller across twelve monthly 10-day windows (120 days). "
        r"The last row is the 120-day aggregate.",
        "tab:monthly", "lccc",
        [r"Window & Oracle & LSTM & Analog \\"], rows)


def t4_mesh() -> str:
    df = pd.read_csv(RES / "01-mesh" / "summary.csv")
    df = df[df["controller_name"] == "prototype"].copy()
    picks = ["h12_t1_5_t2_30", "h12_t1_15_t2_120", "h36_t1_5_t2_30"]
    note = {"h12_t1_5_t2_30": "selected mesh",
            "h12_t1_15_t2_120": "fastest solve",
            "h36_t1_5_t2_30": "long horizon"}
    rows = []
    for c in picks:
        r = df[df["combo"] == c].iloc[0]
        rows.append([f"{r['h']}", f"{r['t1']}", f"{r['t2']}",
                     f"{r['operation_total_cost']:.1f}",
                     f"{r['avg_solve_time_s']:.2f}", note[c]])
    return latex_table(
        r"Representative time-mesh operating points (analog-day MPC, 10-day window). "
        r"Cost is governed by the horizon $h$, solve time by the mesh resolution.",
        "tab:mesh", "cccccl",
        [r"$h$ [h] & $\Delta t_1$ & $\Delta t_2$ [min] & Cost [USD] & Solve [s] & Note \\"],
        rows)


def t5_robustness() -> str:
    s = pd.read_csv(RES / "05-robustness" / "summary.csv")
    piv = s.pivot_table(index="variant", columns="controller",
                        values="operation_total_cost", aggfunc="mean")

    rows = []
    for v in ["noise_005", "noise_010"]:
        if v in piv.index:
            std = f"0.{v.split('_')[1]}"
            rows.append([f"Actuator noise $\\sigma={std}$",
                         f"{piv.loc[v, 'mpc']:.1f}", f"{piv.loc[v, 'stochastic']:.0f}"])
    for v in ["outage_2pct", "outage_5pct"]:
        if v in piv.index:
            pct = v.replace("outage_", "").replace("pct", "")
            rows.append([f"Outage prob.\\ {pct}\\%/day",
                         f"{piv.loc[v, 'mpc']:.1f}", f"{piv.loc[v, 'stochastic']:.0f}"])
    seeds = piv[piv.index.str.startswith("seed")]
    smpc = seeds["mpc"]
    sst = seeds["stochastic"]
    rows.append([r"\midrule Outage seeds (mean\,$\pm$\,sd)",
                 f"{smpc.mean():.1f}$\\pm${smpc.std():.1f}",
                 f"{sst.mean():.0f}$\\pm${sst.std():.0f}"])
    rows.append([r"Outage seeds (max)", f"{smpc.max():.1f}", f"{sst.max():.0f}"])
    return latex_table(
        r"Robustness of closed-loop MPC vs.\ the receding stochastic plan. "
        r"Reported values are averaged over the available validation windows.",
        "tab:robustness", "lcc",
        [r"Variant & MPC [USD] & Stochastic [USD] \\"], rows)


def t6_mesh_full() -> str:
    df = pd.read_csv(RES / "01-mesh" / "summary.csv")
    df = df[df["t2"].isin([30, 60, 120])]
    methods = [("ideal", "Perfect forecast"), ("lstm", "LSTM"), ("prototype", "Analog-day")]
    t2s = [30, 60, 120]

    cost = {}
    rank = {}
    for key, _ in methods:
        d = df[df["controller_name"] == key]
        for _, r in d.iterrows():
            cost[(key, r["h"], r["t1"], r["t2"])] = r["operation_total_cost"]
        for pos, (_, r) in enumerate(d.nsmallest(3, "operation_total_cost").iterrows(), start=1):
            rank[(key, r["h"], r["t1"], r["t2"])] = pos

    rows = []
    for h in [12, 24, 36]:
        for i, t1 in enumerate([5, 10, 15]):
            cells = [str(h) if i == 0 else "", str(t1)]
            for key, _ in methods:
                for t2 in t2s:
                    v = cost.get((key, h, t1, t2))
                    txt = "--" if v is None else f"{v:.1f}"
                    p = rank.get((key, h, t1, t2))
                    if p == 1:
                        txt = f"\\textbf{{{txt}}}"
                    elif p in (2, 3):
                        txt = f"\\underline{{{txt}}}"
                    cells.append(txt)
            rows.append(cells)

    header = [
        r"& & \multicolumn{3}{c}{Perfect forecast} & \multicolumn{3}{c}{LSTM} & \multicolumn{3}{c}{Analog-day} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"$h$ [h] & $\Delta t_1$ & 30 & 60 & 120 & 30 & 60 & 120 & 30 & 60 & 120 \\",
    ]
    return latex_table(
        r"Operating cost [USD, 10-day May window] over the full time-mesh grid: "
        r"forecaster $\times$ horizon $h$ $\times$ near-term step $\Delta t_1$ [min] "
        r"$\times$ coarse step $\Delta t_2$ [min]. Bold marks each forecaster's best "
        r"mesh, underline the second and third best.",
        "tab:mesh-full", "cc ccc ccc ccc",
        header, rows)


def main() -> None:
    parts = [
        "% Generated by scripts/make_tables.py.",
        "% Requires \\usepackage{booktabs} in the paper preamble.",
        "",
        t1_forecast(), t2_controllers(), t3_monthly(), t4_mesh(), t5_robustness(),
        t6_mesh_full(),
    ]
    (OUT / "tables.tex").write_text("\n".join(parts), encoding="utf-8")
    print(f"[tables] 6 tables written to {(OUT / 'tables.tex').as_posix()}")


if __name__ == "__main__":
    main()
