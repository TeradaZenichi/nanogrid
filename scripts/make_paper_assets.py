"""Build publication-ready LaTeX tables and vector figures.

The assets are generated exclusively from the consolidated, audited files in
``outputs/paper-analysis`` and the annual temporal-mesh campaign.  Figures are
designed for a single journal column (3.45 in) unless their content genuinely
requires two columns.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "paper-analysis"
OUT = SOURCE / "paper-assets"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
PREVIEWS = OUT / "previews"
for directory in (FIGURES, TABLES, PREVIEWS):
    directory.mkdir(parents=True, exist_ok=True)

FONT_PATH = ROOT / "data" / "Gulliver.otf"
if not FONT_PATH.is_file():
    raise FileNotFoundError(f"Required paper font is missing: {FONT_PATH}")
font_manager.fontManager.addfont(str(FONT_PATH))
FONT_NAME = font_manager.FontProperties(fname=FONT_PATH).get_name()

plt.rcParams.update(
    {
        # Gulliver is primary; DejaVu only supplies symbols absent from the OTF
        # (notably the Unicode minus used by Matplotlib tick formatters).
        "font.family": [FONT_NAME, "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.65,
        "lines.linewidth": 1.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "savefig.transparent": False,
    }
)

COLORS = {
    "critical_50": "#3378b5",
    "full_100": "#d45b3f",
    "with": "#225ea8",
    "without": "#9ecae1",
    "prototype": "#3b7f52",
    "stochastic": "#7553a6",
}


def _save(fig: plt.Figure, stem: str) -> None:
    """Save vector paper output and a raster preview for visual QA."""
    fig.tight_layout(pad=0.45)
    for label in fig.findobj(matplotlib.text.Text):
        label.set_text(label.get_text().replace("\N{MINUS SIGN}", "-"))
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(PREVIEWS / f"{stem}.png", bbox_inches="tight", pad_inches=0.025, dpi=220)
    plt.close(fig)


def _style(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis=grid_axis, color="#d6d6d6", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)


def _write_table(name: str, caption: str, label: str, colspec: str,
                 header: list[str], rows: list[list[str]], *, wide: bool = False,
                 note: str | None = None) -> None:
    environment = "table*" if wide else "table"
    lines = [
        f"\\begin{{{environment}}}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        *header,
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if note:
        lines.append(r"\par\vspace{1mm}{\footnotesize " + note + "}")
    lines.extend([f"\\end{{{environment}}}", ""])
    (TABLES / name).write_text("\n".join(lines), encoding="utf-8")


def sizing_assets() -> None:
    sizing = pd.read_csv(SOURCE / "sizing_degradation_comparison.csv").set_index("sizing_case")
    cases = ["critical_50", "full_100"]
    labels = [r"Critical 50\%", r"Full 100\%"]

    rows = []
    for case, label in zip(cases, labels):
        row = sizing.loc[case]
        rows.append(
            [
                label,
                f"{row.pv_size_kw_with_degradation:.2f}",
                f"{row.bess_size_kwh_with_degradation:.2f}",
                f"{row.bess_power_kw_with_degradation:.2f}",
                f"{row.objective_usd_with_degradation:,.0f}",
                f"{row.bess_size_kwh_degradation_delta_pct:+.1f}",
            ]
        )
    _write_table(
        "tab_sizing.tex",
        "Degradation-aware PV--BESS sizing for the two resilience requirements.",
        "tab:sizing",
        "lrrrr",
        [r"Case & PV & BESS $E/P$ & Cost & $\Delta E$ \\",
         r" & [kW] & [kWh/kW] & [USD] & [\%] \\"],
        [[row[0], row[1], f"{row[2]}/{row[3]}", row[4], row[5]] for row in rows],
        note=r"$\Delta E$ is the BESS-capacity increase relative to sizing without degradation.",
    )

    x = np.arange(len(cases))
    width = 0.34
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    without = sizing.loc[cases, "bess_size_kwh_without_degradation"].to_numpy()
    with_deg = sizing.loc[cases, "bess_size_kwh_with_degradation"].to_numpy()
    ax.bar(x - width / 2, without, width, color=COLORS["without"], label="Without degradation")
    ax.bar(x + width / 2, with_deg, width, color=COLORS["with"], label="With degradation")
    for xpos, values in ((x - width / 2, without), (x + width / 2, with_deg)):
        for xx, value in zip(xpos, values):
            ax.text(xx, value + 0.25, f"{value:.2f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x, ["Critical 50%", "Full 100%"])
    ax.set_ylabel("Optimal BESS capacity [kWh]")
    ax.set_ylim(0, max(with_deg) * 1.22)
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    _save(fig, "fig_sizing_degradation")


def autonomy_assets() -> None:
    frame = pd.read_csv(SOURCE / "sizing_autonomy_processed.csv")
    frame = frame[frame.sizing_case.isin(["critical_50", "full_100"])].copy()
    rows = []
    for case, label in (("critical_50", r"Critical 50\%"), ("full_100", r"Full 100\%")):
        subset = frame[frame.sizing_case == case].sort_values("outage_duration_h")
        for idx, row in subset.iterrows():
            duration = int(row.outage_duration_h)
            if row.with_degradation_status == "optimal":
                bess = f"{row.bess_size_kwh_with_degradation:.2f}"
                pv = f"{row.pv_size_kw_with_degradation:.2f}"
                objective = f"{row.objective_usd_with_degradation:,.0f}"
            else:
                bess = f"$>{row.original_capacity_cap_kwh:.0f}$"
                pv = "--"
                objective = "infeasible"
            rows.append([label if duration == 1 else "", str(duration), pv, bess, objective])
    _write_table(
        "tab_autonomy.tex",
        "Sensitivity of degradation-aware sizing to the outage-autonomy requirement.",
        "tab:autonomy",
        "lrrrr",
        [r"Case & Autonomy [h] & PV [kW] & BESS [kWh] & Cost [USD] \\"],
        rows,
        note=(r"The 6-h full-service design is infeasible under the 25-kWh cap; "
              r"an expanded-bound solve certified a minimum of 25.63 kWh."),
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 4.45), sharex=True)
    for case, label in (("critical_50", "Critical 50%"), ("full_100", "Full 100%")):
        d = frame[(frame.sizing_case == case) & (frame.with_degradation_status == "optimal")]
        color = COLORS[case]
        ax1.plot(d.outage_duration_h, d.bess_size_kwh_with_degradation, "o-", color=color, label=label)
        ax2.plot(d.outage_duration_h, d.objective_usd_with_degradation / 1000, "o-", color=color)
    cert = frame[(frame.sizing_case == "full_100") & frame.capacity_limit_certified.astype(bool)]
    if not cert.empty:
        row = cert.iloc[0]
        ax1.scatter([row.outage_duration_h], [row.minimum_feasible_bess_kwh_with_degradation],
                    marker="x", s=38, linewidth=1.5, color=COLORS["full_100"], zorder=5)
        ax1.axhline(row.original_capacity_cap_kwh, color="#555555", linestyle="--", linewidth=0.8)
        ax1.annotate(
            "minimum 25.63 kWh",
            (row.outage_duration_h, row.minimum_feasible_bess_kwh_with_degradation),
            xytext=(-8, 12), textcoords="offset points", ha="right", va="bottom",
            fontsize=6.8, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
            arrowprops={"arrowstyle": "-", "lw": 0.6},
        )
    ax1.set_ylabel("BESS capacity [kWh]")
    ax1.set_ylim(top=27.2)
    ax1.legend(
        frameon=True, facecolor="white", edgecolor="none", framealpha=0.92,
        loc="center left", bbox_to_anchor=(0.01, 0.72), borderpad=0.25,
    )
    ax2.set_ylabel("Lifetime cost [thousand USD]")
    ax2.set_xlabel("Required outage autonomy [h]")
    ax2.set_xticks([1, 2, 4, 6])
    for ax in (ax1, ax2):
        _style(ax)
    fig.subplots_adjust(hspace=0.13)
    _save(fig, "fig_autonomy_sensitivity")


def mesh_assets() -> None:
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.5), sharex=False)
    for ax, (case, label) in zip(axes, (("critical_50", "Critical 50%"), ("full_100", "Full 100%"))):
        path = ROOT / "outputs" / "operation-sweep" / "with-degradation" / case
        grouped = pd.read_csv(path / "02-seasonal-mesh" / "champion_mesh.csv")
        champion = json.loads((path / "champion_mesh.json").read_text(encoding="utf-8"))
        mask = ((grouped.h == champion["h"]) & (grouped.t1 == champion["t1"]) &
                (grouped.t2 == champion["t2"]))
        chosen = grouped[mask].iloc[0]
        rows.append([
            label.replace("%", r"\%"),
            f"{champion['h']}/{champion['t1']}/{champion['t2']}",
            f"{100 * chosen.mean_regret:.3f}",
            f"{100 * chosen.max_regret:.3f}",
            f"{chosen.mean_solve_time_s:.3f}",
            f"{chosen.selection_score:.4f}",
        ])
        dominated = grouped[~grouped.pareto.astype(bool)]
        pareto = grouped[grouped.pareto.astype(bool)]
        ax.scatter(dominated.mean_solve_time_s, 100 * dominated.mean_regret,
                   s=13, alpha=0.28, color=COLORS[case], edgecolors="none",
                   label="Dominated")
        ax.scatter(pareto.mean_solve_time_s, 100 * pareto.mean_regret,
                   s=18, alpha=0.75, color=COLORS[case], edgecolors="none",
                   label="Pareto set")
        ax.scatter([chosen.mean_solve_time_s], [100 * chosen.mean_regret], s=48, marker="*",
                   color="#111111", label="Selected mesh", zorder=5)
        ax.set_xscale("log")
        # Matplotlib's default LogFormatter emits U+2212 in negative powers.
        # Gulliver has an ASCII hyphen but no U+2212, so use plain numeric
        # labels and suppress minor labels instead of accepting a missing glyph.
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([0.2, 0.5, 1.0] if case == "critical_50" else [0.2, 0.4, 0.8])
        ax.set_ylabel("Mean forecaster regret [%]")
        ax.set_title(label, loc="left", fontweight="bold")
        ax.yaxis.set_major_locator(MaxNLocator(5))
        _style(ax, "both")
    axes[-1].set_xlabel("Mean solution time per MPC step [s]")
    axes[0].legend(frameon=False, loc="upper right", fontsize=6.5)
    fig.subplots_adjust(hspace=0.32)
    _save(fig, "fig_mesh_tradeoff")
    _write_table(
        "tab_mesh.tex",
        "Selected nonuniform MPC meshes from the annual two-forecaster campaign.",
        "tab:mesh",
        "lrrrrr",
        [r"Case & $H/t_1/t_2$ & Mean & Max. & Solve & Score \\",
         r" & [h/min/min] & [\%] & [\%] & [s] & \\"],
        rows,
        note=(r"Regrets use each forecaster's own minimum-cost mesh. The score is "
              r"$0.4z(\bar r)+0.4z(r_{\max})+0.2z(T)$ after Pareto filtering."),
    )


def controller_assets() -> None:
    stats = pd.read_csv(SOURCE / "controller_paired_statistics.csv")
    stats = stats[(stats.metric == "operation_total_cost_inventory_adjusted") &
                  stats.sizing_case.isin(["critical_50", "full_100"])]
    order = ["ideal", "lstm", "reserve_only", "load_shifting", "peak_shaving", "self_consumption"]
    names = {
        "ideal": "Perfect forecast",
        "lstm": "LSTM",
        "reserve_only": "Reserve-only",
        "load_shifting": "Load shifting",
        "peak_shaving": "Peak shaving",
        "self_consumption": "Self-consumption",
    }
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 5.25), sharex=True)
    for ax, (case, label) in zip(axes, (("critical_50", "Critical 50%"), ("full_100", "Full 100%"))):
        d = stats[stats.sizing_case == case].set_index("comparator_controller").loc[order]
        values = d.mean_paired_difference_pct.to_numpy()
        colors = ["#4c78a8" if value < 0 else "#e07a5f" for value in values]
        y = np.arange(len(order))
        ax.barh(y, values, color=colors, height=0.64)
        ax.axvline(0, color="#222222", linewidth=0.7)
        ax.set_yticks(y, [names[key] for key in order])
        ax.invert_yaxis()
        ax.set_title(label, loc="left", fontweight="bold")
        for yy, value in zip(y, values):
            # Put small negative labels just to the right of zero so they do not
            # collide with long method names at single-column width.
            xpos = value + 0.8 if value >= 0 else 0.55
            ax.text(xpos, yy, f"{value:+.1f}%", va="center", ha="left", fontsize=6.8)
        _style(ax, "x")
        for key in order:
            r = d.loc[key]
            rows.append([
                label.replace("%", r"\%") if key == order[0] else "", names[key],
                f"{r.mean_paired_difference_pct:+.1f}",
                f"[{r.bootstrap_ci95_low:+.2f}, {r.bootstrap_ci95_high:+.2f}]",
                f"{r.wilcoxon_holm_p_value:.4f}",
            ])
    axes[-1].set_xlabel("Adjusted cost difference vs. prototype MPC [%]")
    axes[-1].set_xlim(min(-5, stats.mean_paired_difference_pct.min() - 3),
                      stats.mean_paired_difference_pct.max() + 8)
    fig.subplots_adjust(hspace=0.3)
    _save(fig, "fig_controller_comparison")
    _write_table(
        "tab_controller_comparison.tex",
        "Paired comparison against prototype-forecast MPC over twelve monthly windows.",
        "tab:controller-comparison",
        "llrrr",
        [r"Case & Comparator & $\Delta C$ [\%] & 95\% CI [USD] & $p_{\rm Holm}$ \\"],
        rows,
        wide=True,
        note=r"Negative differences favor the comparator. Confidence intervals are paired bootstrap intervals ($20{,}000$ resamples).",
    )


def forecast_decomposition_assets() -> None:
    data = pd.read_csv(SOURCE / "controller_monthly_paired.csv")
    cases = ["critical_50", "full_100"]
    labels = ["Critical 50%", "Full 100%"]
    metrics = [
        ("operation_total_cost", "Raw cost"),
        ("operation_wear_cost", "BESS wear"),
        ("terminal_adjustment", "Terminal adjustment"),
        ("operation_total_cost_inventory_adjusted", "Adjusted cost"),
    ]
    differences: dict[str, dict[str, float]] = {}
    rows = []
    for case, label in zip(cases, labels):
        d = data[data.sizing_case == case]
        pivots = {column: d.pivot(index="month", columns="controller_name", values=column)
                  for column in ("operation_total_cost", "operation_wear_cost",
                                 "operation_total_cost_inventory_adjusted")}
        terminal = pivots["operation_total_cost_inventory_adjusted"] - pivots["operation_total_cost"]
        differences[case] = {
            "operation_total_cost": float((pivots["operation_total_cost"].lstm - pivots["operation_total_cost"].ideal).mean()),
            "operation_wear_cost": float((pivots["operation_wear_cost"].lstm - pivots["operation_wear_cost"].ideal).mean()),
            "terminal_adjustment": float((terminal.lstm - terminal.ideal).mean()),
            "operation_total_cost_inventory_adjusted": float((pivots["operation_total_cost_inventory_adjusted"].lstm - pivots["operation_total_cost_inventory_adjusted"].ideal).mean()),
        }
        ideal = d[d.controller_name == "ideal"]
        lstm = d[d.controller_name == "lstm"]
        rows.append([
            label.replace("%", r"\%"),
            f"{differences[case]['operation_total_cost']:+.3f}",
            f"{differences[case]['operation_wear_cost']:+.3f}",
            f"{differences[case]['terminal_adjustment']:+.3f}",
            f"{differences[case]['operation_total_cost_inventory_adjusted']:+.3f}",
            f"{lstm.terminal_energy_delta_kwh.mean() - ideal.terminal_energy_delta_kwh.mean():+.3f}",
        ])

    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    for offset, case, label in ((-width / 2, cases[0], labels[0]), (width / 2, cases[1], labels[1])):
        values = [differences[case][key] for key, _ in metrics]
        ax.bar(x + offset, values, width, color=COLORS[case], label=label)
    ax.axhline(0, color="#222222", linewidth=0.7)
    ax.set_xticks(x, [label.replace(" ", "\n") for _, label in metrics])
    ax.set_ylabel(r"Mean LSTM - ideal difference [USD]")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _style(ax)
    _save(fig, "fig_lstm_ideal_decomposition")
    _write_table(
        "tab_forecast_decomposition.tex",
        "Decomposition of the mean LSTM-minus-perfect-forecast cost difference.",
        "tab:forecast-decomposition",
        "lrrrrr",
        [r"Case & Raw & Wear & Terminal & Adjusted & $\Delta E_T$ \\",
         r" & [USD] & [USD] & [USD] & [USD] & [kWh] \\"],
        rows,
        note=(r"Negative values favor LSTM. Wear is a component of raw cost; terminal adjustment is applied ex post. "
              r"$\Delta E_T$ is the LSTM-minus-perfect terminal-energy difference."),
    )


def robustness_assets() -> None:
    frame = pd.read_csv(SOURCE / "robustness_summary.csv")
    frame = frame[(frame.sizing_case.isin(["critical_50", "full_100"])) &
                  (frame.variant.isin(["noise_005", "noise_010"]))]
    rows = []
    for case, label in (("critical_50", r"Critical 50\%"), ("full_100", r"Full 100\%")):
        for noise, noise_label in (("noise_005", "5"), ("noise_010", "10")):
            d = frame[(frame.sizing_case == case) & (frame.variant == noise)].set_index("controller_name")
            rows.append([label if noise == "noise_005" else "", noise_label,
                         f"{d.loc['prototype', 'operation_total_cost_delta_pct_vs_base']:+.2f}",
                         f"{d.loc['stochastic', 'operation_total_cost_delta_pct_vs_base']:+.2f}"])
    _write_table(
        "tab_robustness.tex",
        "Sensitivity to BESS actuator noise over four seasonal windows.",
        "tab:robustness",
        "lrrr",
        [r"Case & Noise [\%] & Prototype [\%] & Stochastic [\%] \\"],
        rows,
        note=r"Entries are mean operating-cost changes relative to each controller's noise-free base case.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.45), sharey=True)
    for ax, (case, label) in zip(axes, (("critical_50", "Critical\n50%"), ("full_100", "Full\n100%"))):
        d = frame[frame.sizing_case == case]
        x = np.arange(2)
        width = 0.34
        for offset, ctrl, ctrl_label, color in (
            (-width / 2, "prototype", "Prototype", COLORS["prototype"]),
            (width / 2, "stochastic", "Stochastic", COLORS["stochastic"]),
        ):
            values = [float(d[(d.variant == variant) & (d.controller_name == ctrl)].operation_total_cost_delta_pct_vs_base.iloc[0])
                      for variant in ("noise_005", "noise_010")]
            ax.bar(x + offset, values, width, color=color, label=ctrl_label)
        ax.set_xticks(x, ["5%", "10%"])
        ax.set_xlabel(label)
        _style(ax)
    axes[0].set_ylabel("Cost increase from base [%]")
    axes[1].legend(frameon=False, loc="upper left", fontsize=6.5)
    _save(fig, "fig_bess_noise_robustness")


def write_manifest() -> None:
    table_names = [
        "tab_sizing.tex", "tab_autonomy.tex", "tab_mesh.tex",
        "tab_controller_comparison.tex", "tab_forecast_decomposition.tex",
        "tab_robustness.tex",
    ]
    combined = [
        "% Generated by scripts/make_paper_assets.py",
        "% Preamble requirements: \\usepackage{booktabs}",
        "",
    ]
    for name in table_names:
        combined.append((TABLES / name).read_text(encoding="utf-8"))
    (TABLES / "tables_all.tex").write_text("\n".join(combined), encoding="utf-8")

    manifest = {
        "font_file": str(FONT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "font_family_detected": FONT_NAME,
        "figure_width_in": 3.45,
        "figures": sorted(path.name for path in FIGURES.glob("*.pdf")),
        "tables": table_names,
        "source_files": [
            "outputs/paper-analysis/sizing_degradation_comparison.csv",
            "outputs/paper-analysis/sizing_autonomy_processed.csv",
            "outputs/paper-analysis/controller_monthly_paired.csv",
            "outputs/paper-analysis/controller_paired_statistics.csv",
            "outputs/paper-analysis/robustness_summary.csv",
            "outputs/operation-sweep/with-degradation/{case}/02-seasonal-mesh/champion_mesh.csv",
            "outputs/operation-sweep/with-degradation/{case}/02-seasonal-mesh/champion_mesh.json",
        ],
        "mesh_selection_decision_model": "docs/mesh_selection_decision_model.md",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Paper assets\n\n"
        "Generated with `python scripts/make_paper_assets.py`. Figures are vector PDFs "
        "set to 3.45 inches (single-column width) and use the Gulliver font. PNG files "
        "under `previews/` are for visual inspection only.\n\n"
        "LaTeX tables require `\\usepackage{booktabs}`. Each table is available separately; "
        "`tables/tables_all.tex` contains the complete set. The controller-comparison table "
        "uses `table*`; all other assets are single-column.\n\n"
        "The exact temporal-mesh decision rule is recorded in "
        "`docs/mesh_selection_decision_model.md`.\n",
        encoding="utf-8",
    )


def main() -> None:
    sizing_assets()
    autonomy_assets()
    mesh_assets()
    controller_assets()
    forecast_decomposition_assets()
    robustness_assets()
    write_manifest()
    print(f"Paper assets written to {OUT}")


if __name__ == "__main__":
    main()
