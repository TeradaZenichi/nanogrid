"""Build portable input bundles for the three operational sizing campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opt.utils import load_sized_parameters_from_artifact


PARAMETERS = ROOT / "data" / "parameters.json"
OUTPUT_ROOT = ROOT / "operation-campaigns"
CAMPAIGNS = {
    "economic": ROOT / "paper" / "sizing" / "economic" / "degradation" / "sizing_decision_variables.json",
    "critical_50": ROOT / "paper" / "sizing" / "resilience" / "critical" / "sizing_decision_variables.json",
    "full_100": ROOT / "paper" / "sizing" / "resilience" / "full" / "sizing_decision_variables.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_payload(source: Path, campaign_id: str) -> dict:
    result = json.loads(source.read_text(encoding="utf-8"))
    metadata = result.get("metadata", {}) or {}
    audit = result.get("model_audit", {}) or {}
    return {
        "schema_version": 1,
        "artifact_type": "portable_sizing_solution",
        "case_id": campaign_id,
        "source_result_sha256": _sha256(source),
        "source_result": source.relative_to(ROOT).as_posix(),
        "result_created_at_utc": result.get("created_at_utc"),
        "solver_status": result.get("solver_status"),
        "termination_condition": result.get("termination_condition"),
        "has_loaded_solution": result.get("has_loaded_solution"),
        "minimum_outage_load_served_fraction": result.get(
            "minimum_outage_load_served_fraction",
            metadata.get("minimum_outage_load_served_fraction", 0.0),
        ),
        "replacement_policy": result.get("replacement_policy"),
        "decision_variables": result.get("decision_variables", {}),
        "costs": result.get("costs", result.get("objective_breakdown", {})),
        "metadata": {
            "cyclic_daily_soc": metadata.get("cyclic_daily_soc"),
            "optimized_cyclic_initial_soc": metadata.get("optimized_cyclic_initial_soc"),
            "sizing_model_version": metadata.get("sizing_model_version"),
            "minimum_outage_load_served_fraction": metadata.get(
                "minimum_outage_load_served_fraction", 0.0
            ),
            "resilience_peak_load_kw": metadata.get("resilience_peak_load_kw", 0.0),
            "resilience_outage_energy_kwh": metadata.get(
                "resilience_outage_energy_kwh", 0.0
            ),
        },
        "model_audit": {
            "cycle_closure_max_abs_kwh": audit.get("cycle_closure_max_abs_kwh"),
            "simultaneous_charge_discharge_max_kw": audit.get(
                "simultaneous_charge_discharge_max_kw"
            ),
            "resilience_power_margin_by_year_kw": audit.get(
                "resilience_power_margin_by_year_kw"
            ),
            "resilience_energy_margin_by_year_kwh": audit.get(
                "resilience_energy_margin_by_year_kwh"
            ),
        },
    }


def _write_or_reuse_artifact(source: Path, artifact: Path, campaign_id: str) -> None:
    executed = (
        ROOT / "outputs" / "sweeps" / campaign_id
        / "campaign-inputs" / "sizing_artifact.json"
    )
    if executed.exists():
        artifact.write_bytes(executed.read_bytes())
        return
    if source.exists():
        artifact.write_text(
            json.dumps(_portable_payload(source, campaign_id), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return
    if not artifact.exists():
        raise FileNotFoundError(
            f"Neither the original result nor its canonical artifact exists: {source}"
        )


def _economic_coverage() -> dict:
    summary_path = ROOT / "paper" / "operation" / "economic" / "mesh" / "summary.csv"
    if not summary_path.exists():
        return {"complete": False, "summary": summary_path.relative_to(ROOT).as_posix()}
    summary = pd.read_csv(summary_path)
    controllers = sorted(summary["controller_name"].dropna().unique().tolist())
    combinations = int(summary["combo"].dropna().nunique())
    months = int(summary["month"].dropna().nunique())
    audit_pass = int(
        summary["audit_pass"].fillna(False).astype(str).str.lower().eq("true").sum()
    )
    expected = 27 * 3 * 12
    coverage = {
        "complete": len(summary) == expected and audit_pass == expected,
        "summary": summary_path.relative_to(ROOT).as_posix(),
        "rows": int(len(summary)),
        "expected_rows": expected,
        "controllers": controllers,
        "temporal_meshes": combinations,
        "monthly_windows": months,
        "audit_pass_rows": audit_pass,
    }
    current_path = CAMPAIGNS["economic"]
    resilience_path = (
        ROOT / "paper" / "sizing" / "resilience" / "economic" / "sizing_decision_variables.json"
    )
    if current_path.exists() and resilience_path.exists():
        current = json.loads(current_path.read_text(encoding="utf-8"))["decision_variables"]
        resilience = json.loads(resilience_path.read_text(encoding="utf-8"))["decision_variables"]
        deltas = {
            "pv_size_kw": abs(float(current["P_hat_PV_kw"]) - float(resilience["P_hat_PV_kw"])),
            "bess_size_kwh": abs(
                float(current["E_hat_BESS_kwh"]) - float(resilience["E_hat_BESS_kwh"])
            ),
        }
        coverage["new_economic_equivalence"] = {
            "equivalent_within_1e-6": max(deltas.values()) <= 1e-6,
            "absolute_differences": deltas,
            "new_sizing_artifact": resilience_path.relative_to(ROOT).as_posix(),
        }
    return coverage


def _commands(campaign_id: str, artifact: Path) -> list[str]:
    common = (
        "python experiments/12_corrected_pipeline.py --workers 4 --n-iters 2880 "
        f"--parameters data/parameters.json --sizing-artifact {artifact.as_posix()} "
        f"--campaign-id {campaign_id} --out-root outputs/sweeps/{campaign_id}"
    )
    campaign_root = f"outputs/sweeps/{campaign_id}"
    return [
        f"{common} --stage mesh",
        f"{common} --stage annual-mesh",
        (
            "python experiments/14_consolidate_reference_results.py "
            f"--campaign-root {campaign_root} --campaign-id {campaign_id}"
        ),
        (
            "python experiments/15_baseline_comparison.py "
            f"--campaign-root {campaign_root} --campaign-id {campaign_id} --workers 4"
        ),
        (
            "python experiments/13_generate_publication_artifacts.py "
            f"--campaign-root {campaign_root} --campaign-id {campaign_id}"
        ),
    ]


def build_bundles(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    economic_coverage = _economic_coverage()
    index = {"schema_version": 1, "campaigns": {}}
    for campaign_id, source in CAMPAIGNS.items():
        campaign_dir = output_root / campaign_id
        campaign_dir.mkdir(parents=True, exist_ok=True)
        artifact = campaign_dir / "sizing_artifact.json"
        _write_or_reuse_artifact(source, artifact, campaign_id)
        params = load_sized_parameters_from_artifact(PARAMETERS, artifact)
        applied = params["sizing_case_applied"]
        commands = _commands(campaign_id, artifact.relative_to(ROOT))
        payload = {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "status": "complete" if campaign_id == "economic" and economic_coverage["complete"] else "pending",
            "sizing_artifact": artifact.relative_to(ROOT).as_posix(),
            "sizing_artifact_sha256": _sha256(artifact),
            "parameters": PARAMETERS.relative_to(ROOT).as_posix(),
            "parameters_sha256": _sha256(PARAMETERS),
            "pv_size_kw": applied["P_hat_PV_kw"],
            "bess_size_kwh": applied["E_hat_BESS_kwh"],
            "bess_power_kw": applied["BESS_Pmax_kw"],
            "operation_days": 120,
            "mesh_forecaster_configurations": 81,
            "monthly_runs": 972,
            "commands": commands,
        }
        if campaign_id == "economic":
            payload["existing_coverage"] = economic_coverage
        (campaign_dir / "campaign.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        script = ["$ErrorActionPreference = 'Stop'"]
        for command in commands:
            script.extend((command, "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"))
        (campaign_dir / "run_full_sweep.ps1").write_text(
            "\n".join(script) + "\n", encoding="utf-8"
        )
        index["campaigns"][campaign_id] = payload

    (output_root / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_root / "README.md").write_text(
        "# Distributed operational sweeps\n\n"
        "Each directory contains a solved sizing artifact, the two-stage 972-run "
        "operational sweep, reference consolidation, paired baselines, and publication "
        "artifacts. No sizing optimization is executed on the worker machine.\n\n"
        "The economic sweep is published under `paper/operation/economic/mesh`. "
        "Run `critical_50/run_full_sweep.ps1` and `full_100/run_full_sweep.ps1` on "
        "separate repository copies. Return the corresponding directory under "
        "`outputs/sweeps/` without renaming it, then promote audited summaries to `paper/`.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_bundles(args.output_root.resolve())
    print(args.output_root.resolve())
