"""Generate the paper figures from result summaries."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "outputs"
RES = SHARED / "sweeps" / "economic"
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)

_font_path = str(ROOT / "data" / "Gulliver.otf")
font_manager.fontManager.addfont(_font_path)
_prop = font_manager.FontProperties(fname=_font_path)
_font_name = _prop.get_name()
plt.rcParams.update({
    "font.family": [_font_name, "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

CTRL = {
    "ideal": ("#111111", "o", "MPC (perfect forecast)"),
    "lstm": ("#1f77b4", "s", "MPC + LSTM"),
    "prototype": ("#2ca02c", "^", "MPC + analog-day"),
    "prototype-prefix": ("#2ca02c", "^", "MPC + analog-day"),
    "stochastic": ("#d62728", "v", "Stochastic plan"),
}

PLAIN_LOG_TICKS = FuncFormatter(lambda value, _: f"{value:g}")


def _save(fig, name: str) -> None:
    fig.tight_layout(pad=0.45)
    for text in fig.findobj(matplotlib.text.Text):
        text.set_text(text.get_text().replace("\N{MINUS SIGN}", "-"))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[fig] {name}")


def _mesh() -> pd.DataFrame:
    return pd.read_csv(RES / "01-mesh" / "summary.csv")


SEASON = {
    "2009-05": ("#6a51a3", "May"),
    "2009-08": ("#e6550d", "Aug (PV-rich)"),
    "2009-11": ("#31a354", "Nov"),
    "2010-02": ("#3182bd", "Feb (PV-poor)"),
}


def _mesh_seasonal() -> pd.DataFrame:
    df = pd.read_csv(RES / "02-seasonal-mesh" / "summary.csv")
    df = df[df["t2"].isin([30, 60, 120])].copy()
    df["season"] = df["month"]
    df["norm"] = df.groupby(["season", "controller_name"])["operation_total_cost"].transform(
        lambda values: values / values.min()
    )
    return df


def fig_mesh_pareto() -> None:
    df = _mesh()
    p = df[df["controller_name"] == "prototype"].copy()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    hcolors = {12: "#ff7f0e", 24: "#1f77b4", 36: "#2ca02c"}
    for h, c in hcolors.items():
        d = p[p["h"] == h]
        ax.scatter(d["avg_solve_time_s"], d["operation_total_cost"],
                   s=55, c=c, alpha=0.8, label=f"h = {h} h", edgecolor="none")

    pp = p.sort_values("avg_solve_time_s")
    best, front = float("inf"), []
    for _, r in pp.iterrows():
        if r["operation_total_cost"] < best - 1e-9:
            best = r["operation_total_cost"]
            front.append(r)
    fr = pd.DataFrame(front)
    ax.plot(fr["avg_solve_time_s"], fr["operation_total_cost"],
            "-", c="#333", lw=1.5, zorder=1, label="Pareto front")

    knee = fr.loc[fr["operation_total_cost"].idxmin()]
    cheap = fr.loc[fr["avg_solve_time_s"].idxmin()]
    for r, dx, dy, ha in [(knee, 6, -2, "left"), (cheap, 8, 8, "left")]:
        ax.annotate(f"h{r['h']}/{r['t1']}/{r['t2']}",
                    (r["avg_solve_time_s"], r["operation_total_cost"]),
                    fontsize=8, fontweight="bold",
                    xytext=(dx, dy), textcoords="offset points", ha=ha)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(PLAIN_LOG_TICKS)
    ax.yaxis.set_major_formatter(PLAIN_LOG_TICKS)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("avg solve time per step [s, log]  (lower better)")
    ax.set_ylabel("operating cost, 10 days [USD, log]  (lower better)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    _save(fig, "fig_mesh_pareto")


def fig_mesh_methods() -> None:
    df = _mesh()
    df = df[(df["t1"] == 10) & df["t2"].isin([30, 60, 120])]
    methods = [("ideal", "perfect forecast"), ("lstm", "LSTM"), ("prototype", "analog-day")]
    hs = [12, 24, 36]
    t2s = [30, 60, 120]

    grids = {}
    for key, _ in methods:
        d = df[df["controller_name"] == key]
        g = np.full((len(hs), len(t2s)), np.nan)
        for _, r in d.iterrows():
            if r["h"] in hs and r["t2"] in t2s:
                g[hs.index(r["h"]), t2s.index(r["t2"])] = r["operation_total_cost"]
        grids[key] = g

    vals = np.concatenate([g[~np.isnan(g)] for g in grids.values() if not np.isnan(g).all()])
    vmin, vmax = vals.min(), vals.max()

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharey=True)
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#dddddd")
    for ax, (key, lbl) in zip(axes, methods):
        g = grids[key]
        ax.imshow(np.ma.masked_invalid(g), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(len(hs)):
            for j in range(len(t2s)):
                txt = "--" if np.isnan(g[i, j]) else f"{g[i, j]:.1f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(t2s)), [str(t) for t in t2s])
        ax.set_yticks(range(len(hs)), [str(h) for h in hs])
        ax.set_xlabel(r"$\Delta t_2$ [min]")
        ax.text(0.5, 1.04, lbl, transform=ax.transAxes, ha="center", fontsize=10)
    axes[0].set_ylabel("horizon $h$ [h]")
    fig.tight_layout()
    _save(fig, "fig_mesh_methods")


def fig_mesh_cube() -> None:
    from matplotlib.colors import LogNorm

    df = _mesh()
    df = df[df["t2"].isin([30, 60, 120])]
    methods = [("ideal", "perfect forecast"), ("lstm", "LSTM"), ("prototype", "analog-day")]
    hs = [12, 24, 36]
    t1s = [5, 10, 15]
    t2s = [30, 60, 120]

    vals = df[df["controller_name"].isin([m for m, _ in methods])]["operation_total_cost"]
    norm = LogNorm(vmin=vals.min(), vmax=vals.max())
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#dddddd")

    from matplotlib.patches import Rectangle

    top3 = {}
    for key, _ in methods:
        d = df[df["controller_name"] == key].nsmallest(3, "operation_total_cost")
        for pos, (_, r) in enumerate(d.iterrows(), start=1):
            top3[(key, r["h"], r["t1"], r["t2"])] = pos

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.5), sharex=True, sharey=True)
    for row, (key, lbl) in enumerate(methods):
        d = df[df["controller_name"] == key]
        for col, t1 in enumerate(t1s):
            ax = axes[row, col]
            g = np.full((len(hs), len(t2s)), np.nan)
            for _, r in d[d["t1"] == t1].iterrows():
                if r["h"] in hs and r["t2"] in t2s:
                    g[hs.index(r["h"]), t2s.index(r["t2"])] = r["operation_total_cost"]
            ax.imshow(np.ma.masked_invalid(g), cmap=cmap, norm=norm, aspect="auto")
            for i in range(len(hs)):
                for j in range(len(t2s)):
                    if np.isnan(g[i, j]):
                        continue
                    pos = top3.get((key, hs[i], t1, t2s[j]))
                    ax.text(j, i, f"{g[i, j]:.0f}" if g[i, j] >= 100 else f"{g[i, j]:.1f}",
                            ha="center", va="center", fontsize=8,
                            fontweight="bold" if pos == 1 else "normal")
                    if pos is not None:
                        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="black", lw=2.2 if pos == 1 else 1.4))
            if row == 0:
                ax.set_title(rf"$\Delta t_1$ = {t1} min", fontsize=10)
            if row == 2:
                ax.set_xlabel(r"$\Delta t_2$ [min]")
            if col == 0:
                ax.set_ylabel(f"{lbl}\n\nhorizon $h$ [h]")
            ax.set_xticks(range(len(t2s)), [str(t) for t in t2s])
            ax.set_yticks(range(len(hs)), [str(h) for h in hs])
    fig.tight_layout()
    _save(fig, "fig_mesh_cube")


def fig_dt2_seasonal() -> None:
    a = _mesh_seasonal()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for ax, ctrl, title in zip(
        axes, ("prototype", "ideal"),
        ("MPC + analog-day forecast", "MPC + perfect forecast (oracle)"),
    ):
        d = a[(a["controller_name"] == ctrl) & (a["h"] == 36)]
        g = d.groupby(["season", "t2"])["norm"].mean().reset_index()
        for season, (col, lbl) in SEASON.items():
            ds = g[g["season"] == season].sort_values("t2")
            ax.plot(ds["t2"], ds["norm"], "-o", c=col, lw=1.8, ms=6, label=lbl)
        ax.axhline(1.0, color="#999", ls=":", lw=1)
        ax.set_xlabel(r"coarse timestep $\Delta t_2$ [min]")
        ax.text(0.5, 0.97, title, transform=ax.transAxes, ha="center", va="top", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([30, 60, 120])
    axes[0].set_ylabel("cost / season minimum")
    axes[0].legend(frameon=False, fontsize=8, title="season", title_fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_dt2_seasonal")


def fig_horizon_seasonal() -> None:
    a = _mesh_seasonal()
    p = a[a["controller_name"] == "prototype"]
    g = p.groupby(["season", "h"])["norm"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for season, (col, lbl) in SEASON.items():
        d = g[g["season"] == season].sort_values("h")
        ax.plot(d["h"], d["norm"], "-o", c=col, lw=1.9, ms=6, label=lbl)
    ax.axhline(1.0, color="#999", ls=":", lw=1)
    ax.set_xlabel("prediction horizon $h$ [h]")
    ax.set_ylabel("cost / season minimum")
    ax.set_xticks([12, 24, 36])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, title="season", title_fontsize=8)
    _save(fig, "fig_horizon_seasonal")


def fig_pareto_seasonal() -> None:
    a = _mesh_seasonal()
    p = a[a["controller_name"] == "prototype"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for season, (col, lbl) in SEASON.items():
        d = p[p["season"] == season].sort_values("avg_solve_time_s")
        ax.scatter(d["avg_solve_time_s"], d["norm"], s=28, c=col, alpha=0.55)
        best, front = float("inf"), []
        for _, r in d.iterrows():
            if r["norm"] < best - 1e-9:
                best = r["norm"]
                front.append(r)
        fr = pd.DataFrame(front)
        ax.plot(fr["avg_solve_time_s"], fr["norm"], "-o", c=col, lw=1.8, ms=5, label=lbl)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(PLAIN_LOG_TICKS)
    ax.set_xlabel("avg solve time per step [s, log]  (lower better)")
    ax.set_ylabel("cost / season minimum  (lower better)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=8, title="season", title_fontsize=8)
    _save(fig, "fig_pareto_seasonal")


def _monthly() -> pd.DataFrame:
    s = pd.read_csv(RES / "03-forecast-operation" / "summary.csv")
    s["ctrl"] = s["forecaster"]
    piv = s.pivot_table(index="month", columns="ctrl",
                        values="operation_total_cost", aggfunc="first")
    return piv.sort_index()


def fig_evpi_monthly() -> None:
    piv = _monthly()
    x = np.arange(len(piv))
    fig, (ax, axe) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    for ctrl in ("ideal", "lstm", "prototype-prefix"):
        c, mk, lbl = CTRL[ctrl]
        ax.plot(x, piv[ctrl], f"-{mk}", c=c, lw=1.8, ms=5, label=lbl)
    ax.set_ylabel("operating cost, 10 days [USD]")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    evpi = piv["lstm"] - piv["ideal"]
    axe.bar(x, evpi, color="#1f77b4", alpha=0.75)
    axe.set_ylabel("EVPI [abs]\nlstm - ideal")
    axe.set_xticks(x)
    axe.set_xticklabels(piv.index, rotation=45, ha="right", fontsize=8)
    axe.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "fig_evpi_monthly")


def fig_stochastic_tail() -> None:
    piv = _monthly()
    seeds = pd.read_csv(RES / "05-robustness" / "summary.csv")
    seeds = seeds[seeds["variant"].str.startswith("seed")]
    seed_stoch = seeds[seeds["controller"] == "stochastic"]["operation_total_cost"].values
    seed_mpc = seeds[seeds["controller"] == "mpc"]["operation_total_cost"].values

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    data = [piv["prototype-prefix"].values, seed_mpc, seed_stoch]
    labels = ["MPC\n(12 months)", "MPC\n(seed variants)",
              "Stochastic\n(seed variants)"]
    bp = ax.boxplot(data, tick_labels=labels, showfliers=True, widths=0.6, patch_artist=True)
    for patch, col in zip(bp["boxes"], ["#2ca02c", "#2ca02c", "#d62728"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.35)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(PLAIN_LOG_TICKS)
    ax.set_ylabel("operating cost, 10 days [USD, log]")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "fig_stochastic_tail")


def fig_recourse() -> None:
    r = pd.read_csv(RES / "04-recourse" / "summary.csv")
    piv = r.pivot_table(index="window", columns="resolve_every_h",
                        values="operation_total_cost")
    piv = piv[sorted(piv.columns)]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for w in piv.index:
        ax.plot(piv.columns, piv.loc[w], "-o", lw=1.6, ms=5, label=w)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(PLAIN_LOG_TICKS)
    ax.set_xlabel("re-solve interval [h]  (smaller = more feedback)")
    ax.set_ylabel("stochastic cost, 10 days [USD, log]")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "fig_recourse")


def main() -> None:
    fig_mesh_pareto()
    fig_mesh_methods()
    fig_mesh_cube()
    fig_dt2_seasonal()
    fig_horizon_seasonal()
    fig_pareto_seasonal()
    fig_evpi_monthly()
    fig_stochastic_tail()
    fig_recourse()
    print(f"\n9 figures written to {OUT.as_posix()}")


if __name__ == "__main__":
    main()
