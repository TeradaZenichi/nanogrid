"""Generate publication artifacts for the annual temporal-mesh sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opt.campaigns import campaign_layout, load_campaign_provenance, validate_summary_provenance


DEFAULT_CAMPAIGN_ROOT = ROOT / "outputs" / "sweeps" / "economic"
GULLIVER_FONT = ROOT / "data" / "Gulliver.otf"

CONTROLLERS = ("ideal", "lstm", "prototype")
IMPLEMENTABLE = ("lstm", "prototype")
COLORS = {"ideal": "#5B6573", "lstm": "#0072B2", "prototype": "#D55E00"}
LABELS = {"ideal": "Ideal target", "lstm": "LSTM", "prototype": "Prototype"}
MARKERS = {12: "o", 24: "s", 36: "^"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_and_validate(path: Path, provenance) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_summary_provenance(df, provenance, "annual mesh")
    required = {
        "month", "combo", "h", "t1", "t2", "controller_name", "audit_pass",
        "operation_total_cost", "operation_grid_reliability_cost",
        "operation_wear_cost", "operation_bess_throughput_kwh",
        "avg_solve_time_s", "total_solve_time_s", "total_time_s",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError("Missing annual-mesh columns: " + ", ".join(missing))
    if len(df) != 972:
        raise RuntimeError(f"Expected 972 annual-mesh rows, found {len(df)}")
    if set(df["controller_name"]) != set(CONTROLLERS):
        raise RuntimeError("Annual mesh must contain ideal, LSTM, and prototype")
    if df["month"].nunique() != 12 or df["combo"].nunique() != 27:
        raise RuntimeError("Annual mesh must contain 12 months and 27 configurations")
    if df.duplicated(["month", "combo", "controller_name"]).any():
        raise RuntimeError("Duplicate month/configuration/controller records detected")
    if not df["audit_pass"].fillna(False).all():
        rejected = int((~df["audit_pass"].fillna(False)).sum())
        raise RuntimeError(f"Cannot report annual mesh: {rejected} rows failed audit")
    numeric = [
        "operation_total_cost", "operation_grid_reliability_cost",
        "operation_wear_cost", "operation_bess_throughput_kwh",
        "avg_solve_time_s", "total_solve_time_s", "total_time_s",
    ]
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        raise RuntimeError("Annual mesh contains non-finite reporting metrics")
    return df


def configuration_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["combo", "h", "t1", "t2", "controller_name"], as_index=False)
        .agg(
            windows=("month", "nunique"),
            cost_120d=("operation_total_cost", "sum"),
            cost_mean_10d=("operation_total_cost", "mean"),
            grid_reliability_cost_120d=("operation_grid_reliability_cost", "sum"),
            wear_cost_120d=("operation_wear_cost", "sum"),
            bess_throughput_120d_kwh=("operation_bess_throughput_kwh", "sum"),
            mean_solve_time_s=("avg_solve_time_s", "mean"),
            total_solve_time_h=("total_solve_time_s", lambda x: x.sum() / 3600.0),
            total_runtime_h=("total_time_s", lambda x: x.sum() / 3600.0),
        )
    )
    best = summary.groupby("controller_name")["cost_120d"].transform("min")
    summary["cost_regret_pct"] = 100.0 * (summary["cost_120d"] / best - 1.0)
    summary["cost_regret_bp"] = 100.0 * summary["cost_regret_pct"]
    return summary.sort_values(["controller_name", "h", "t1", "t2"])


def _style() -> None:
    if not GULLIVER_FONT.exists():
        raise FileNotFoundError(f"Required publication font not found: {GULLIVER_FONT}")
    font_manager.fontManager.addfont(GULLIVER_FONT)
    font_name = font_manager.FontProperties(fname=GULLIVER_FONT).get_name()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "axes.titleweight": "normal",
            "figure.titlesize": 9,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "mathtext.fontset": "custom",
            "mathtext.rm": font_name,
            "mathtext.it": f"{font_name}:italic",
            "mathtext.bf": f"{font_name}:bold",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def _save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(summary: pd.DataFrame, champion: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.55), sharey=True)
    for ax, controller in zip(axes, IMPLEMENTABLE):
        sub = summary[summary.controller_name == controller]
        for horizon in sorted(sub.h.unique()):
            part = sub[sub.h == horizon]
            scatter = ax.scatter(
                part.mean_solve_time_s,
                part.cost_regret_bp,
                c=part.t2,
                cmap="viridis",
                vmin=30,
                vmax=120,
                marker=MARKERS[int(horizon)],
                s=30 + 3 * part.t1,
                edgecolor="white",
                linewidth=0.5,
                label=f"H={int(horizon)} h",
            )
        selected = sub[
            (sub.h == champion["h"])
            & (sub.t1 == champion["t1"])
            & (sub.t2 == champion["t2"])
        ].iloc[0]
        ax.scatter(
            [selected.mean_solve_time_s], [selected.cost_regret_bp],
            marker="*", s=160, facecolor="#CC0000", edgecolor="black",
            linewidth=0.6, zorder=5, label="Selected mesh",
        )
        ax.annotate(
            f"({champion['h']}, {champion['t1']}, {champion['t2']})",
            (selected.mean_solve_time_s, selected.cost_regret_bp),
            xytext=(5, 7), textcoords="offset points", fontsize=6.5,
        )
        ax.set_title(LABELS[controller])
        ax.set_xlabel("Mean optimization time per solve (s)")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Cost regret relative to controller optimum (bp)")
    cbar = fig.colorbar(scatter, ax=axes, pad=0.02, fraction=0.04)
    cbar.set_label(r"Coarse step $\Delta t_2$ (min)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center",
               bbox_to_anchor=(0.46, 0.90))
    fig.suptitle("Annual temporal-mesh cost-computation trade-off", y=0.995)
    fig.subplots_adjust(top=0.76)
    _save(fig, output, "mesh_tradeoff")


def plot_factor_effects(summary: pd.DataFrame, output: Path) -> None:
    factors = [("h", r"H (h)"), ("t1", r"$\Delta t_1$ (min)"), ("t2", r"$\Delta t_2$ (min)")]
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.25))
    for col, (factor, label) in enumerate(factors):
        for controller in CONTROLLERS:
            sub = summary[summary.controller_name == controller]
            effect = sub.groupby(factor, as_index=False).agg(
                cost=("cost_120d", "mean"), solve=("mean_solve_time_s", "mean")
            )
            effect["regret_bp"] = 10_000.0 * (effect.cost / sub.cost_120d.min() - 1.0)
            axes[0, col].plot(
                effect[factor], effect.regret_bp, marker="o",
                color=COLORS[controller], label=LABELS[controller],
            )
            axes[1, col].plot(
                effect[factor], effect.solve, marker="o",
                color=COLORS[controller], label=LABELS[controller],
            )
        axes[0, col].set_title(label)
        axes[1, col].set_xlabel(label)
        axes[0, col].set_xticks(sorted(summary[factor].unique()))
        axes[1, col].set_xticks(sorted(summary[factor].unique()))
    axes[0, 0].set_ylabel("Mean cost regret (bp)")
    axes[1, 0].set_ylabel("Mean solve time (s)")
    axes[0, 2].legend(frameon=False, loc="best")
    fig.suptitle("Main effects of temporal-mesh parameters", y=1.01)
    fig.tight_layout()
    _save(fig, output, "mesh_factor_effects")


def plot_heatmaps(summary: pd.DataFrame, champion: dict, output: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(8.4, 7.6), sharex=True, sharey=True)
    image = None
    vmax = max(0.01, float(summary.cost_regret_bp.quantile(0.98)))
    for row, controller in enumerate(CONTROLLERS):
        for col, horizon in enumerate((12, 24, 36)):
            ax = axes[row, col]
            sub = summary[(summary.controller_name == controller) & (summary.h == horizon)]
            pivot = sub.pivot(index="t1", columns="t2", values="cost_regret_bp").sort_index().sort_index(axis=1)
            image = ax.imshow(
                pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax,
                origin="lower",
            )
            for i, t1 in enumerate(pivot.index):
                for j, t2 in enumerate(pivot.columns):
                    value = pivot.loc[t1, t2]
                    text = f"{value:.3f}"
                    selected = (
                        int(horizon) == champion["h"]
                        and int(t1) == champion["t1"]
                        and int(t2) == champion["t2"]
                    )
                    ax.text(j, i, text, ha="center", va="center", fontsize=6,
                            fontweight="bold" if selected else "normal",
                            color="white" if value > 0.55 * vmax else "black")
                    if selected:
                        ax.scatter(j - 0.36, i, marker="*", s=18, color="black", zorder=4)
            ax.set_xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
            if row == 0:
                ax.set_title(f"H={horizon} h")
            if col == 0:
                ax.set_ylabel(LABELS[controller] + "\n" + r"$\Delta t_1$ (min)")
            if row == 2:
                ax.set_xlabel(r"$\Delta t_2$ (min)")
            ax.grid(False)
    cbar = fig.colorbar(image, ax=axes, pad=0.02, fraction=0.025)
    cbar.set_label("Cost regret (bp)")
    fig.suptitle("Cost sensitivity across the full temporal-mesh design", y=0.99)
    _save(fig, output, "mesh_cost_heatmaps")


def plot_champion_monthly(df: pd.DataFrame, champion: dict, output: Path) -> None:
    tag = f"h{champion['h']}_t1_{champion['t1']}_t2_{champion['t2']}"
    sub = df[df.combo == tag].copy()
    pivot = sub.pivot(index="month", columns="controller_name", values="operation_total_cost")
    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [1.8, 1.0]},
    )
    ax, gap_ax = axes
    x = np.arange(len(pivot))
    for controller in CONTROLLERS:
        ax.plot(x, pivot[controller], marker="o", linewidth=1.5,
                color=COLORS[controller], label=LABELS[controller])
    ax.set_ylabel("Operating cost (10-day window)")
    ax.legend(frameon=False, ncol=3)
    ax.set_title(
        rf"Selected mesh: $H={champion['h']}$ h, $\Delta t_1={champion['t1']}$ min, "
        rf"$\Delta t_2={champion['t2']}$ min"
    )
    for controller in IMPLEMENTABLE:
        gap_bp = 10_000.0 * (pivot[controller] / pivot["ideal"] - 1.0)
        gap_ax.plot(x, gap_bp, marker="o", linewidth=1.5,
                    color=COLORS[controller], label=LABELS[controller])
    gap_ax.axhline(0.0, color=COLORS["ideal"], linewidth=0.8, linestyle="--")
    gap_ax.set_ylabel("Gap to ideal (bp)")
    gap_ax.set_xlabel("Validation window")
    gap_ax.set_xticks(x, pivot.index, rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, output, "mesh_champion_monthly_cost")


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def write_full_table(summary: pd.DataFrame, _champion: dict, path: Path) -> None:
    controller_order = ("lstm", "ideal", "prototype")
    cost = summary.pivot(
        index=["h", "t1", "t2"], columns="controller_name", values="cost_mean_10d"
    )
    solve = summary.pivot(index=["h", "t1", "t2"], columns="controller_name", values="mean_solve_time_s")
    cost_min = cost.min(axis=0)
    solve_min = solve.min(axis=0)

    def minimum_cell(value: float, minimum: float, digits: int) -> str:
        rendered = _fmt(value, digits)
        if np.isclose(value, minimum, rtol=0.0, atol=1e-12):
            return rf"\textbf{{{rendered}}}"
        return rendered

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Operational cost and mean solution time for each controller and temporal-mesh configuration. Values are averaged over twelve 10-day monthly windows. The ideal controller represents the perfect-information target and is not considered an operational candidate. Bold values indicate the minimum of each column.}",
        r"\label{tab:controller_mesh_comparison}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ccc rr rr rr}",
        r"\toprule",
        r"& & & \multicolumn{2}{c}{LSTM} & \multicolumn{2}{c}{Ideal} & \multicolumn{2}{c}{Prototype} \\",
        r"\cmidrule(lr){4-5}",
        r"\cmidrule(lr){6-7}",
        r"\cmidrule(lr){8-9}",
        r"$H$ & $\Delta t_1$ & $\Delta t_2$ & $C^{\mathrm{op}}$ & $\overline{\tau}^{\mathrm{solve}}$ & $C^{\mathrm{op}}$ & $\overline{\tau}^{\mathrm{solve}}$ & $C^{\mathrm{op}}$ & $\overline{\tau}^{\mathrm{solve}}$ \\",
        r"(h) & (min) & (min) & (USD) & (s) & (USD) & (s) & (USD) & (s) \\",
        r"\midrule",
    ]
    for h, t1, t2 in cost.index:
        values = []
        for controller in controller_order:
            c = minimum_cell(
                cost.loc[(h, t1, t2), controller], cost_min[controller], 4
            )
            s = minimum_cell(
                solve.loc[(h, t1, t2), controller], solve_min[controller], 3
            )
            values.extend([c, s])
        lines.append(f"{int(h)} & {int(t1)} & {int(t2)} & " + " & ".join(values) + r" \\")
        if t2 == 120 and not (h == cost.index[-1][0] and t1 == cost.index[-1][1]):
            lines.append(r"\addlinespace[1pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ranked_table(selection_path: Path, path: Path) -> None:
    ranked = pd.read_csv(selection_path).head(10)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Top temporal meshes ranked using implementable-controller regret and computational time.}",
        r"\label{tab:annual_mesh_ranked}",
        r"\small",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"Rank & $H$ & $\Delta t_1$ & $\Delta t_2$ & Mean regret & Max. regret & Time (s) \\",
        r"\midrule",
    ]
    for rank, row in enumerate(ranked.itertuples(index=False), start=1):
        vals = [
            str(rank), str(int(row.h)), str(int(row.t1)), str(int(row.t2)),
            rf"{100*row.mean_regret:.4f}\%", rf"{100*row.max_regret:.4f}\%",
            _fmt(row.mean_solve_time_s, 3),
        ]
        if rank == 1:
            vals = [rf"\textbf{{{x}}}" for x in vals]
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_champion_comparison(df: pd.DataFrame, champion: dict, path: Path) -> None:
    tag = f"h{champion['h']}_t1_{champion['t1']}_t2_{champion['t2']}"
    sub = df[df.combo == tag]
    pivot = sub.pivot(index="month", columns="controller_name", values="operation_total_cost")
    totals = pivot.sum()
    delta = pivot["lstm"] - pivot["prototype"]
    ci = stats.t.interval(0.95, len(delta) - 1, loc=delta.mean(), scale=stats.sem(delta))
    metrics = sub.groupby("controller_name").agg(
        grid_reliability=("operation_grid_reliability_cost", "sum"),
        wear=("operation_wear_cost", "sum"),
        total=("operation_total_cost", "sum"),
        throughput=("operation_bess_throughput_kwh", "sum"),
        solve=("avg_solve_time_s", "mean"),
        runtime=("total_time_s", lambda x: x.sum() / 3600.0),
    )
    metrics["gap_ideal"] = 100.0 * (metrics.total / metrics.loc["ideal", "total"] - 1.0)
    implementable_min = metrics.loc[list(IMPLEMENTABLE)].min()

    def cell(controller: str, metric: str, digits: int, emphasize: bool = True) -> str:
        value = metrics.loc[controller, metric]
        rendered = _fmt(value, digits)
        if emphasize and controller in IMPLEMENTABLE and np.isclose(value, implementable_min[metric]):
            return rf"\textbf{{{rendered}}}"
        return rendered
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Operational comparison for the selected temporal mesh ($H={champion['h']}$~h, $\Delta t_1={champion['t1']}$~min, and $\Delta t_2={champion['t2']}$~min) over twelve paired 10-day validation windows.}}",
        r"\label{tab:selected_mesh_controllers}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Controller & \multicolumn{4}{c}{Operating cost} & BESS & \multicolumn{2}{c}{Computation} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){7-8}",
        r" & Grid and reliability & Wear & Total & Gap to ideal & Throughput & Mean solve & Runtime \\",
        r" & (USD) & (USD) & (USD) & (\%) & (kWh) & (s) & (h) \\",
        r"\midrule",
    ]
    for controller in CONTROLLERS:
        lines.append(
            f"{LABELS[controller]} & {cell(controller, 'grid_reliability', 3)} & "
            f"{cell(controller, 'wear', 3)} & {cell(controller, 'total', 3)} & "
            f"{cell(controller, 'gap_ideal', 3)} & {cell(controller, 'throughput', 3, False)} & "
            f"{cell(controller, 'solve', 3)} & {cell(controller, 'runtime', 2)} " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\par\smallskip",
            r"\parbox{0.96\textwidth}{\scriptsize The ideal controller is a nonimplementable target and is not considered when highlighting the best implementable result. Bold entries indicate the lower cost or computation metric between LSTM and prototype. Costs and throughput are accumulated over 120 validation days; mean solve time is calculated per MPC optimization. The paired LSTM--prototype monthly total-cost difference is "
            + rf"{_fmt(delta.sum(), 4)} ({_fmt(100*delta.sum()/totals['prototype'], 4)}\%); "
            + rf"the 95\% confidence interval of the mean difference per 10-day window is [{_fmt(ci[0], 4)}, {_fmt(ci[1], 4)}] USD." + r"}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(
    df: pd.DataFrame,
    champion: dict,
    path: Path,
    campaign_root: Path,
    campaign_id: str,
) -> None:
    tag = f"h{champion['h']}_t1_{champion['t1']}_t2_{champion['t2']}"
    selected = df[df.combo == tag]
    totals = selected.groupby("controller_name")["operation_total_cost"].sum()
    solve = selected.groupby("controller_name")["avg_solve_time_s"].mean()
    delta = totals["lstm"] - totals["prototype"]
    text = f"""# Annual temporal-mesh reporting package

This directory summarizes the audited annual temporal-mesh campaign: 27 mesh
configurations, three controllers, and twelve paired 10-day validation windows
(972 simulations and 120 evaluated days per configuration/controller).

The selected trade-off mesh is $H={champion['h']}$ h,
$\\Delta t_1={champion['t1']}$ min, and $\\Delta t_2={champion['t2']}$ min.
Its 120-day costs are {totals['ideal']:.4f} (ideal target),
{totals['lstm']:.4f} (LSTM), and {totals['prototype']:.4f} (prototype). The
LSTM-prototype difference is {delta:.4f}; their mean solution times are
{solve['lstm']:.3f} s and {solve['prototype']:.3f} s, respectively.

## Files

- `mesh_tradeoff.pdf`: implementable-controller cost/computation trade-off.
- `mesh_factor_effects.pdf`: marginal effects of $H$, $\\Delta t_1$, and
  $\\Delta t_2$ on cost regret and solution time.
- `mesh_cost_heatmaps.pdf`: all 81 controller/mesh combinations, including the
  ideal target used only as a reference.
- `mesh_champion_monthly_cost.pdf`: monthly costs and gaps to the ideal target
  for the selected mesh.
- `mesh_full_table.tex`: complete 27-row comparison.
- `mesh_ranked_table.tex`: ten highest-ranked implementable meshes.
- `mesh_champion_comparison.tex`: selected-mesh operational cost, BESS-use, and
  computation comparison.
- `mesh_configuration_summary.csv`: aggregated data behind all artifacts.
- `manifest.json`: source hash and campaign-integrity metadata.

PNG versions are provided for inspection; PDF versions should be used in the
paper. All charts use `data/Gulliver.otf`, with 7 pt axis labels and 8 pt panel
titles. The selected mesh is the joint cost/computation compromise for LSTM
and prototype, not the minimum-cost mesh of the ideal target.

## Reproduction

```powershell
.\\.venv\\Scripts\\python.exe experiments\\13_generate_publication_artifacts.py --campaign-root {campaign_root.relative_to(ROOT).as_posix()} --campaign-id {campaign_id}
```
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--campaign-id")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--champion", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    args.campaign_root = args.campaign_root.resolve()
    layout = campaign_layout(args.campaign_root)
    args.input = (args.input or layout["summary"]).resolve()
    args.selection = (args.selection or (layout["seasonal"] / "champion_mesh.csv")).resolve()
    args.champion = (args.champion or layout["champion"]).resolve()
    args.output = (args.output or layout["publication"]).resolve()
    provenance = load_campaign_provenance(args.campaign_root, args.campaign_id)
    df = _load_and_validate(args.input, provenance)
    champion = json.loads(args.champion.read_text(encoding="utf-8"))
    mesh = {key: int(champion[key]) for key in ("h", "t1", "t2")}
    args.output.mkdir(parents=True, exist_ok=True)
    _style()

    summary = configuration_summary(df)
    summary.to_csv(args.output / "mesh_configuration_summary.csv", index=False)
    plot_tradeoff(summary, mesh, args.output)
    plot_factor_effects(summary, args.output)
    plot_heatmaps(summary, mesh, args.output)
    plot_champion_monthly(df, mesh, args.output)
    write_full_table(summary, mesh, args.output / "mesh_full_table.tex")
    write_ranked_table(args.selection, args.output / "mesh_ranked_table.tex")
    write_champion_comparison(df, mesh, args.output / "mesh_champion_comparison.tex")
    write_readme(
        df, mesh, args.output / "README.md",
        args.campaign_root, provenance.campaign_id,
    )

    manifest = {
        "campaign_id": provenance.campaign_id,
        "sizing_source_sha256": provenance.sizing_source_sha256,
        "source": str(args.input),
        "source_sha256": _sha256(args.input),
        "rows": len(df),
        "controllers": list(CONTROLLERS),
        "months": sorted(df.month.unique().tolist()),
        "configurations": int(df.combo.nunique()),
        "audit_passed": int(df.audit_pass.sum()),
        "selected_mesh": mesh,
        "typography": {
            "family": "Gulliver",
            "font_path": str(GULLIVER_FONT),
            "font_sha256": _sha256(GULLIVER_FONT),
            "axis_label_size_pt": 7,
            "panel_title_size_pt": 8,
            "figure_title_size_pt": 9,
        },
        "cost_scope": "sum of twelve paired 10-day validation windows (120 days)",
        "regret_definition": "relative to the minimum 120-day cost of each controller",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
