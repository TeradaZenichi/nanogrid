"""Build paper-ready sizing, controller, and robustness analyses."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "paper-analysis"
SIZING_ROOT = ROOT / "outputs" / "sizing-sweep"
OPERATION_ROOT = ROOT / "outputs" / "operation-sweep" / "with-degradation"
BASELINE_ROOT = ROOT / "outputs" / "baseline-sweep" / "with-degradation"
BASELINE_SCRIPT = ROOT / "3-baselines.py"
CAPACITY_DIAGNOSTIC = (
    OUTPUT / "sizing-autonomy" / "capacity_limit_diagnostic.json"
)
CASES = ("economic", "critical_50", "full_100")
CONTROLLER_ORDER = (
    "ideal",
    "lstm",
    "prototype",
    "load_shifting",
    "reserve_only",
    "peak_shaving",
    "self_consumption",
)
PRIMARY_REFERENCE = "prototype"
PAIRED_METRICS = (
    "operation_total_cost_inventory_adjusted",
    "operation_total_cost",
    "operation_wear_cost",
    "operation_bess_throughput_kwh",
    "load_shed_kwh",
    "pv_curtailed_kwh",
)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260908
GULLIVER_PATH = ROOT / "data" / "Gulliver.otf"


def _configure_paper_style() -> str:
    """Use the manuscript font consistently in every generated figure."""
    if not GULLIVER_PATH.is_file():
        raise FileNotFoundError(f"Gulliver font not found: {GULLIVER_PATH}")
    font_manager.fontManager.addfont(str(GULLIVER_PATH))
    family = font_manager.FontProperties(fname=GULLIVER_PATH).get_name()
    plt.rcParams.update(
        {
            "font.family": [family, "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return family


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _sizing_comparison() -> pd.DataFrame:
    rows = []
    for variant in ("with-degradation", "without-degradation"):
        path = SIZING_ROOT / variant / "summary.csv"
        frame = pd.read_csv(path)
        if set(frame["case_id"]) != set(CASES):
            raise RuntimeError(f"Incomplete sizing summary: {path}")
        frame.insert(0, "degradation_variant", variant)
        rows.append(frame)
    combined = pd.concat(rows, ignore_index=True, sort=False)
    pivot = combined.pivot(index="case_id", columns="degradation_variant")
    output_rows = []
    for case in CASES:
        row = {"sizing_case": case}
        for metric in (
            "pv_size_kw",
            "bess_size_kwh",
            "bess_power_kw",
            "capex_usd",
            "npv_opex_usd",
            "objective_usd",
        ):
            with_value = float(pivot.loc[case, (metric, "with-degradation")])
            without_value = float(
                pivot.loc[case, (metric, "without-degradation")]
            )
            row[f"{metric}_with_degradation"] = with_value
            row[f"{metric}_without_degradation"] = without_value
            row[f"{metric}_degradation_delta"] = with_value - without_value
            row[f"{metric}_degradation_delta_pct"] = (
                100.0 * (with_value - without_value) / abs(without_value)
                if abs(without_value) > 1e-6
                else np.nan
            )
        output_rows.append(row)
    return pd.DataFrame(output_rows)


def _controller_monthly(baselines: ModuleType) -> pd.DataFrame:
    outputs = []
    for case in CASES:
        case_data = baselines._load_case(case)
        references, warnings = baselines._load_available_references(case_data)
        if warnings:
            raise RuntimeError(f"{case}: reference warnings: {'; '.join(warnings)}")
        baseline = pd.read_csv(BASELINE_ROOT / case / "summary.csv")
        prototype = references[references["controller_name"] == PRIMARY_REFERENCE]
        paired_baseline, _, months = baselines._paired_selection(
            baseline, prototype
        )
        references = references[references["month"].isin(months)].copy()
        expected = len(baselines.MONTHS)
        coverage = references.groupby("controller_name")["month"].nunique()
        if len(months) != expected or not (coverage == expected).all():
            raise RuntimeError(f"{case}: incomplete paired controller coverage")
        paired_baseline["pairing_pass"] = True
        combined = pd.concat(
            [references, paired_baseline], ignore_index=True, sort=False
        )
        combined["sizing_case"] = case
        outputs.append(combined)

    frame = pd.concat(outputs, ignore_index=True, sort=False)
    keep = [
        "sizing_case",
        "month",
        "controller_name",
        "controller_family",
        "exogenous_sha256",
        "audit_pass",
        *PAIRED_METRICS,
        "grid_import_kwh",
        "grid_export_kwh",
        "final_soc_pct",
        "terminal_energy_delta_kwh",
        "outage_events",
        "outage_hours",
    ]
    missing = sorted(set(keep).difference(frame.columns))
    if missing:
        raise RuntimeError("Controller monthly data lacks: " + ", ".join(missing))
    frame = frame[keep].copy()
    for metric in (*PAIRED_METRICS, "grid_import_kwh", "grid_export_kwh"):
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")
    frame["controller_name"] = pd.Categorical(
        frame["controller_name"], CONTROLLER_ORDER, ordered=True
    )
    return frame.sort_values(["sizing_case", "month", "controller_name"])


def _bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _holm_adjust(p_values: Iterable[float]) -> list[float]:
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    count = len(p)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _paired_statistics(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case in CASES:
        case_frame = monthly[monthly["sizing_case"] == case]
        for metric in PAIRED_METRICS:
            pivot = case_frame.pivot(
                index="month", columns="controller_name", values=metric
            )
            metric_rows = []
            for controller in CONTROLLER_ORDER:
                if controller == PRIMARY_REFERENCE:
                    continue
                paired = pivot[[PRIMARY_REFERENCE, controller]].dropna()
                difference = (
                    paired[controller] - paired[PRIMARY_REFERENCE]
                ).to_numpy(dtype=float)
                if len(difference) != 12:
                    raise RuntimeError(
                        f"{case}/{metric}/{controller}: expected 12 paired months"
                    )
                digest = hashlib.sha256(
                    f"{case}|{metric}|{controller}".encode()
                ).digest()
                seed = BOOTSTRAP_SEED + int.from_bytes(digest[:4], "little")
                ci_low, ci_high = _bootstrap_mean_ci(difference, seed)
                if np.allclose(difference, 0.0, atol=1e-12):
                    p_value = 1.0
                else:
                    try:
                        p_value = float(
                            wilcoxon(
                                difference,
                                zero_method="wilcox",
                                alternative="two-sided",
                                method="auto",
                            ).pvalue
                        )
                    except ValueError:
                        p_value = 1.0
                reference_mean = float(paired[PRIMARY_REFERENCE].mean())
                comparator_mean = float(paired[controller].mean())
                metric_rows.append(
                    {
                        "sizing_case": case,
                        "metric": metric,
                        "reference_controller": PRIMARY_REFERENCE,
                        "comparator_controller": controller,
                        "paired_months": len(difference),
                        "reference_mean": reference_mean,
                        "comparator_mean": comparator_mean,
                        "mean_paired_difference": float(difference.mean()),
                        "median_paired_difference": float(np.median(difference)),
                        "mean_paired_difference_pct": (
                            100.0 * float(difference.mean()) / abs(reference_mean)
                            if abs(reference_mean) > 1e-12
                            else np.nan
                        ),
                        "bootstrap_ci95_low": ci_low,
                        "bootstrap_ci95_high": ci_high,
                        "months_comparator_lower": int((difference < -1e-12).sum()),
                        "months_equal": int(np.isclose(difference, 0.0, atol=1e-12).sum()),
                        "months_comparator_higher": int((difference > 1e-12).sum()),
                        "wilcoxon_p_value": p_value,
                    }
                )
            adjusted = _holm_adjust(row["wilcoxon_p_value"] for row in metric_rows)
            for row, p_adjusted in zip(metric_rows, adjusted):
                row["wilcoxon_holm_p_value"] = p_adjusted
                row["ci_excludes_zero"] = bool(
                    row["bootstrap_ci95_low"] > 0.0
                    or row["bootstrap_ci95_high"] < 0.0
                )
            rows.extend(metric_rows)
    return pd.DataFrame(rows)


def _robustness_summary() -> pd.DataFrame:
    rows = []
    metrics = (
        "operation_total_cost",
        "operation_wear_cost",
        "operation_load_shedding_cost",
        "operation_pv_curtailment_cost",
        "operation_bess_throughput_kwh",
    )
    for case in CASES:
        frame = pd.read_csv(OPERATION_ROOT / case / "05-robustness" / "summary.csv")
        valid = frame[
            frame["status"].astype(str).ne("error")
            & frame["audit_pass"].astype(str).str.lower().isin(("true", "1"))
        ].copy()
        for (controller, variant), group in valid.groupby(
            ["controller_name", "variant"], sort=True
        ):
            row = {
                "sizing_case": case,
                "controller_name": controller,
                "variant": variant,
                "seasonal_windows": int(group["window"].nunique()),
                "trajectories": len(group),
            }
            for metric in metrics:
                values = pd.to_numeric(group[metric], errors="raise")
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1))
                row[f"{metric}_sum"] = float(values.sum())
            rows.append(row)
    result = pd.DataFrame(rows)
    for metric in ("operation_total_cost", "operation_wear_cost"):
        column = f"{metric}_mean"
        base = (
            result[result["variant"] == "base"]
            .set_index(["sizing_case", "controller_name"])[column]
            .to_dict()
        )
        result[f"{metric}_delta_vs_base"] = result.apply(
            lambda row, value_column=column, base_values=base: row[value_column]
            - base_values[(row["sizing_case"], row["controller_name"])],
            axis=1,
        )
        result[f"{metric}_delta_pct_vs_base"] = result.apply(
            lambda row, value_column=column, base_values=base: (
                100.0
                * (
                    row[value_column]
                    - base_values[(row["sizing_case"], row["controller_name"])]
                )
                / abs(base_values[(row["sizing_case"], row["controller_name"])])
                if abs(
                    base_values[(row["sizing_case"], row["controller_name"])]
                )
                > 1e-12
                else np.nan
            ),
            axis=1,
        )
    return result


def _autonomy_processed() -> pd.DataFrame:
    path = OUTPUT / "sizing-autonomy" / "summary.csv"
    frame = pd.read_csv(path)
    expected = len(CASES[1:]) * 4 * 2
    if len(frame) != expected:
        raise RuntimeError(f"Autonomy summary has {len(frame)}/{expected} rows")
    certificate = json.loads(CAPACITY_DIAGNOSTIC.read_text(encoding="utf-8"))
    if not (
        certificate.get("has_loaded_solution")
        and certificate.get("original_cap_is_infeasible")
    ):
        raise RuntimeError("The 6 h capacity-limit diagnostic is not certified")
    pivot = frame.pivot(
        index=["case_id", "outage_duration_h"],
        columns="degradation_variant",
    )
    rows = []
    for case in CASES[1:]:
        for duration in (1.0, 2.0, 4.0, 6.0):
            with_status = str(
                pivot.loc[(case, duration), ("termination_condition", "with-degradation")]
            ).lower()
            without_status = str(
                pivot.loc[
                    (case, duration),
                    ("termination_condition", "without-degradation"),
                ]
            ).lower()
            capacity_limited = case == "full_100" and duration == 6.0
            if capacity_limited:
                with_status = "capacity_limit_infeasible"
            row = {
                "sizing_case": case,
                "outage_duration_h": duration,
                "with_degradation_status": with_status,
                "without_degradation_status": without_status,
                "capacity_limit_certified": capacity_limited,
                "original_capacity_cap_kwh": (
                    float(certificate["original_capacity_cap_kwh"])
                    if capacity_limited
                    else np.nan
                ),
                "minimum_feasible_bess_kwh_with_degradation": (
                    float(certificate["minimum_feasible_bess_kwh"])
                    if capacity_limited
                    else np.nan
                ),
                "capacity_shortfall_kwh": (
                    float(certificate["original_cap_shortfall_kwh"])
                    if capacity_limited
                    else np.nan
                ),
            }
            for metric in (
                "pv_size_kw",
                "bess_size_kwh",
                "bess_power_kw",
                "capex_usd",
                "npv_opex_usd",
                "objective_usd",
                "test_energy_margin_kwh_last_year",
            ):
                with_value = float(
                    pivot.loc[(case, duration), (metric, "with-degradation")]
                )
                without_value = float(
                    pivot.loc[(case, duration), (metric, "without-degradation")]
                )
                row[f"{metric}_with_degradation"] = with_value
                row[f"{metric}_without_degradation"] = without_value
                row[f"{metric}_degradation_delta"] = with_value - without_value
                row[f"{metric}_degradation_delta_pct"] = (
                    100.0 * (with_value - without_value) / abs(without_value)
                    if np.isfinite(with_value)
                    and np.isfinite(without_value)
                    and abs(without_value) > 1e-12
                    else np.nan
                )
            row["bess_required_kwh_with_degradation"] = (
                row["minimum_feasible_bess_kwh_with_degradation"]
                if capacity_limited
                else row["bess_size_kwh_with_degradation"]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _plot_paired_cost(statistics: pd.DataFrame) -> None:
    data = statistics[
        statistics["metric"] == "operation_total_cost_inventory_adjusted"
    ].copy()
    controllers = [c for c in CONTROLLER_ORDER if c != PRIMARY_REFERENCE]
    labels = {
        "ideal": "Perfect forecast MPC",
        "lstm": "LSTM MPC",
        "load_shifting": "Load shifting",
        "reserve_only": "Reserve only",
        "peak_shaving": "Peak shaving",
        "self_consumption": "Self-consumption",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharey=True)
    for ax, case in zip(axes, CASES):
        subset = data[data["sizing_case"] == case].set_index(
            "comparator_controller"
        ).reindex(controllers)
        y = np.arange(len(controllers))
        mean = subset["mean_paired_difference"].to_numpy(float)
        low = subset["bootstrap_ci95_low"].to_numpy(float)
        high = subset["bootstrap_ci95_high"].to_numpy(float)
        ax.errorbar(
            mean,
            y,
            xerr=np.vstack((mean - low, high - mean)),
            fmt="o",
            color="#245B78",
            ecolor="#7CA6BD",
            capsize=3,
        )
        ax.axvline(0.0, color="#555555", linewidth=0.9, linestyle="--")
        ax.set_title(case.replace("_", " ").title())
        ax.grid(axis="x", alpha=0.2)
        ax.set_yticks(y, [labels[c] for c in controllers])
        ax.invert_yaxis()
        if case == "economic" and np.nanmax(np.abs(np.r_[low, high])) < 1e-6:
            ax.set_xlim(-0.01, 0.01)
            ax.set_xticks((-0.01, 0.0, 0.01), ("-0.01", "0", "0.01"))
            ax.text(
                0.5,
                0.04,
                "No material controller difference",
                transform=ax.transAxes,
                ha="center",
                fontsize=7.5,
                color="#555555",
            )
    fig.suptitle("Paired monthly controller comparison (mean and bootstrap 95% CI)")
    fig.supxlabel("Cost difference vs prototype MPC (USD/10-day window)", y=0.01)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUTPUT / "paired_controller_cost.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "paired_controller_cost.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_autonomy(autonomy: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True)
    colors = {"critical_50": "#3F7CAC", "full_100": "#B65745"}
    for case in CASES[1:]:
        subset = autonomy[autonomy["sizing_case"] == case].sort_values(
            "outage_duration_h"
        )
        x = subset["outage_duration_h"]
        label = case.replace("_", " ").title()
        axes[0].plot(
            x,
            subset["bess_size_kwh_with_degradation"],
            marker="o",
            color=colors[case],
            label=label,
        )
        axes[0].plot(
            x,
            subset["bess_size_kwh_without_degradation"],
            marker="o",
            linestyle="--",
            color=colors[case],
            alpha=0.75,
        )
        axes[1].plot(
            x,
            subset["objective_usd_with_degradation"],
            marker="o",
            color=colors[case],
            label=label,
        )
        axes[1].plot(
            x,
            subset["objective_usd_without_degradation"],
            marker="o",
            linestyle="--",
            color=colors[case],
            alpha=0.75,
        )
    certified = autonomy[autonomy["capacity_limit_certified"]].iloc[0]
    axes[0].scatter(
        [certified["outage_duration_h"]],
        [certified["minimum_feasible_bess_kwh_with_degradation"]],
        marker="X",
        s=70,
        color=colors["full_100"],
        zorder=5,
    )
    axes[0].axhline(
        certified["original_capacity_cap_kwh"],
        color="#555555",
        linestyle=":",
        linewidth=1.0,
    )
    axes[0].annotate(
        f"minimum {certified['minimum_feasible_bess_kwh_with_degradation']:.2f} kWh\n"
        f"> {certified['original_capacity_cap_kwh']:.0f} kWh cap",
        xy=(
            certified["outage_duration_h"],
            certified["minimum_feasible_bess_kwh_with_degradation"],
        ),
        xytext=(-6, 15),
        textcoords="offset points",
        ha="right",
        fontsize=7.5,
    )
    axes[0].set_ylabel("BESS energy capacity (kWh)")
    axes[1].set_ylabel("Lifetime objective (USD)")
    for ax in axes:
        ax.set_xlabel("Outage duration (h)")
        ax.set_xticks((1, 2, 4, 6))
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].text(
        0.98,
        0.03,
        "Solid: with degradation\nDashed: without degradation",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    axes[1].text(
        0.53,
        0.94,
        "6 h with degradation:\ncapacity-limit infeasible",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color=colors["full_100"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "sizing_autonomy_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "sizing_autonomy_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_markdown(
    sizing: pd.DataFrame,
    autonomy: pd.DataFrame,
    statistics: pd.DataFrame,
    robustness: pd.DataFrame,
) -> None:
    lines = [
        "# Paper result summary",
        "",
        "## Degradation-aware sizing",
        "",
        "| Sizing case | BESS without degradation (kWh) | BESS with degradation (kWh) | BESS change | Lifetime-cost change |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in ("critical_50", "full_100"):
        row = sizing[sizing["sizing_case"] == case].iloc[0]
        lines.append(
            f"| {case} | {row['bess_size_kwh_without_degradation']:.3f} | "
            f"{row['bess_size_kwh_with_degradation']:.3f} | "
            f"{row['bess_size_kwh_degradation_delta_pct']:+.2f}% | "
            f"{row['objective_usd_degradation_delta_pct']:+.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Outage-autonomy sensitivity",
            "",
            "| Sizing case | Outage duration (h) | BESS with degradation (kWh) | PV with degradation (kW) | Lifetime objective (USD) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in autonomy.itertuples(index=False):
        if row.capacity_limit_certified:
            lines.append(
                f"| {row.sizing_case} | {row.outage_duration_h:g} | "
                f"infeasible; minimum {row.minimum_feasible_bess_kwh_with_degradation:.3f} "
                f"> cap {row.original_capacity_cap_kwh:.1f} | -- | -- |"
            )
        else:
            lines.append(
                f"| {row.sizing_case} | {row.outage_duration_h:g} | "
                f"{row.bess_size_kwh_with_degradation:.3f} | "
                f"{row.pv_size_kw_with_degradation:.3f} | "
                f"{row.objective_usd_with_degradation:.2f} |"
            )

    lines.extend(
        [
            "",
            (
                "The 6 h full-service point with degradation is infeasible under "
                "the 25 kWh design cap. An expanded-bound minimum-capacity solve "
                "certified 25.6276 kWh, a 0.6276 kWh shortfall."
            ),
        ]
    )

    cost = statistics[
        (statistics["metric"] == "operation_total_cost_inventory_adjusted")
        & statistics["sizing_case"].isin(("critical_50", "full_100"))
    ]
    lines.extend(
        [
            "",
            "## Paired controller comparison",
            "",
            "Differences are comparator minus prototype MPC for one 10-day window. Negative differences favor the comparator.",
            "",
            "| Sizing case | Comparator | Mean difference (USD) | Bootstrap 95% CI | Holm-adjusted p-value |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in cost.itertuples(index=False):
        lines.append(
            f"| {row.sizing_case} | {row.comparator_controller} | "
            f"{row.mean_paired_difference:+.3f} | "
            f"[{row.bootstrap_ci95_low:+.3f}, {row.bootstrap_ci95_high:+.3f}] | "
            f"{row.wilcoxon_holm_p_value:.4f} |"
        )

    robustness_focus = robustness[
        robustness["variant"].isin(("noise_005", "noise_010", "outage_2pct", "outage_5pct"))
        & robustness["sizing_case"].isin(("critical_50", "full_100"))
    ]
    lines.extend(
        [
            "",
            "## Robustness",
            "",
            "| Sizing case | Controller | Variant | Mean-cost change from base |",
            "|---|---|---|---:|",
        ]
    )
    for row in robustness_focus.itertuples(index=False):
        lines.append(
            f"| {row.sizing_case} | {row.controller_name} | {row.variant} | "
            f"{row.operation_total_cost_delta_pct_vs_base:+.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Statistical protocol",
            "",
            f"- {BOOTSTRAP_SAMPLES:,} paired bootstrap resamples with seed {BOOTSTRAP_SEED}.",
            "- Two-sided Wilcoxon signed-rank tests.",
            "- Holm correction within each sizing case and metric.",
            "- Twelve paired seasonal windows per controller comparison.",
        ]
    )
    (OUTPUT / "paper_result_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _configure_paper_style()
    baselines = _load_module(BASELINE_SCRIPT, "baseline_campaign_analysis")
    sizing = _sizing_comparison()
    monthly = _controller_monthly(baselines)
    statistics = _paired_statistics(monthly)
    robustness = _robustness_summary()
    autonomy = _autonomy_processed()

    _atomic_csv(sizing, OUTPUT / "sizing_degradation_comparison.csv")
    _atomic_csv(monthly, OUTPUT / "controller_monthly_paired.csv")
    _atomic_csv(statistics, OUTPUT / "controller_paired_statistics.csv")
    _atomic_csv(robustness, OUTPUT / "robustness_summary.csv")
    _atomic_csv(autonomy, OUTPUT / "sizing_autonomy_processed.csv")
    _plot_paired_cost(statistics)
    _plot_autonomy(autonomy)
    _write_markdown(sizing, autonomy, statistics, robustness)

    manifest = {
        "analysis_version": "paper-analysis-v2",
        "primary_controller_reference": PRIMARY_REFERENCE,
        "paired_months_per_comparison": 12,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "paired_difference_definition": "comparator minus prototype MPC",
        "wilcoxon_alternative": "two-sided",
        "multiple_comparison_correction": "Holm within sizing case and metric",
        "autonomy_sizing_points": 16,
        "autonomy_feasible_points": 15,
        "capacity_limit_certificate": {
            "path": "sizing-autonomy/capacity_limit_diagnostic.json",
            "case": "full_100",
            "outage_duration_h": 6.0,
            "original_cap_kwh": 25.0,
            "minimum_feasible_kwh": float(
                autonomy.loc[
                    autonomy["capacity_limit_certified"],
                    "minimum_feasible_bess_kwh_with_degradation",
                ].iloc[0]
            ),
        },
        "outputs": [
            "sizing_degradation_comparison.csv",
            "controller_monthly_paired.csv",
            "controller_paired_statistics.csv",
            "robustness_summary.csv",
            "sizing_autonomy_processed.csv",
            "paper_result_summary.md",
            "paired_controller_cost.png",
            "paired_controller_cost.pdf",
            "sizing_autonomy_sensitivity.png",
            "sizing_autonomy_sensitivity.pdf",
        ],
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Paper analyses written to {OUTPUT}")


if __name__ == "__main__":
    main()
