"""Build the compact, paper-facing result package."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MANIFEST = PAPER / "manifest.json"
PACKAGE_VERSION = "paper-results-v3"
TEXT_SUFFIXES = {".csv", ".json", ".md", ".tex", ".txt"}
GITHUB_FILE_LIMIT_BYTES = 100 * 1024 * 1024
OPERATION_CAMPAIGNS = {"economic": "economic", "critical": "critical_50", "full": "full_100"}

FILES = {
    "sizing/economic/sizing_report.pdf": "outputs/sizing/sizing_report.pdf",
    "sizing/economic/sizing_report_manifest.json": "outputs/sizing/sizing_report_manifest.json",
    "sizing/economic/comparison_alpha_cases.json": "outputs/sizing/comparison_alpha_cases.json",
    "sizing/economic/sizing_comparison_alpha.png": "outputs/sizing/sizing_comparison_alpha.png",
    "sizing/economic/sizing_trajectories.png": "outputs/sizing/sizing_trajectories.png",
    "sizing/economic/degradation/sizing_decision_variables.json": "outputs/sizing/alpha_gt_0/sizing_decision_variables.json",
    "sizing/economic/degradation/bess_capacity_by_year.pdf": "outputs/sizing/alpha_gt_0/bess_capacity_by_year.pdf",
    "sizing/economic/degradation/discounted_opex_by_year.pdf": "outputs/sizing/alpha_gt_0/discounted_opex_by_year.pdf",
    "sizing/economic/no_degradation/sizing_decision_variables.json": "outputs/sizing/alpha_eq_0/sizing_decision_variables.json",
    "sizing/economic/no_degradation/bess_capacity_by_year.pdf": "outputs/sizing/alpha_eq_0/bess_capacity_by_year.pdf",
    "sizing/economic/no_degradation/discounted_opex_by_year.pdf": "outputs/sizing/alpha_eq_0/discounted_opex_by_year.pdf",
    "sizing/sensitivity/summary.csv": "outputs/sizing-sensitivity/sizing_sensitivity_summary.csv",
    "sizing/sensitivity/results.json": "outputs/sizing-sensitivity/sizing_sensitivity_results.json",
    "sizing/sensitivity/report.pdf": "outputs/sizing-sensitivity/sizing_sensitivity_report.pdf",
    "sizing/sensitivity/manifest.json": "outputs/sizing-sensitivity/manifest.json",
    "sizing/sensitivity/case_catalog.json": "outputs/sizing-sensitivity/case_catalog.json",
    "sizing/resilience/summary.csv": "outputs/sizing-resilience/summary.csv",
    "sizing/resilience/report.pdf": "outputs/sizing-resilience/sizing_resilience_report.pdf",
    "sizing/resilience/manifest.json": "outputs/sizing-resilience/manifest.json",
    "sizing/resilience/economic/effective_parameters.json": "outputs/sizing-resilience/economic/effective_parameters.json",
    "sizing/resilience/economic/sizing_decision_variables.json": "outputs/sizing-resilience/economic/sizing_decision_variables.json",
    "sizing/resilience/critical/effective_parameters.json": "outputs/sizing-resilience/critical_50/effective_parameters.json",
    "sizing/resilience/critical/sizing_decision_variables.json": "outputs/sizing-resilience/critical_50/sizing_decision_variables.json",
    "sizing/resilience/full/effective_parameters.json": "outputs/sizing-resilience/full_100/effective_parameters.json",
    "sizing/resilience/full/sizing_decision_variables.json": "outputs/sizing-resilience/full_100/sizing_decision_variables.json",
    "forecasting/evaluation_manifest.json": "outputs/forecasting-causal-v2/evaluation_manifest.json",
    "forecasting/parameters_used.json": "outputs/forecasting-causal-v2/parameters_used.json",
    "forecasting/summary.json": "outputs/forecasting-causal-v2/prototype_eval_summary.json",
    "forecasting/metrics.csv": "outputs/forecasting-causal-v2/prototype_eval_metrics.csv",
    "forecasting/cluster_accuracy.csv": "outputs/forecasting-causal-v2/prototype_eval_cluster_accuracy.csv",
    "forecasting/error_by_lead_load.png": "outputs/forecasting-causal-v2/prototype_eval_error_by_lead_load.png",
    "forecasting/error_by_lead_pv.png": "outputs/forecasting-causal-v2/prototype_eval_error_by_lead_pv.png",
    "forecasting/cluster_accuracy.png": "outputs/forecasting-causal-v2/prototype_eval_cluster_accuracy.png",
    "noise/pipeline_state.json": "outputs/sweeps/economic/05-robustness/pipeline_state.json",
    "operation/index.json": "operation-campaigns/index.json",
    "operation/economic/provenance/campaign.json": "operation-campaigns/economic/campaign.json",
    "operation/economic/provenance/sizing_artifact.json": "operation-campaigns/economic/sizing_artifact.json",
    "operation/economic/provenance/parameters.json": "outputs/sweeps/economic/campaign-inputs/parameters.json",
    "operation/economic/provenance/operational_parameters.json": "outputs/sweeps/economic/campaign-inputs/operational_parameters.json",
    "operation/economic/campaign_manifest.json": "outputs/sweeps/economic/campaign_manifest.json",
    "operation/economic/champion_mesh.json": "outputs/sweeps/economic/champion_mesh.json",
    "operation/economic/mesh/summary.csv": "outputs/sweeps/economic/02-seasonal-mesh/summary.csv",
    "operation/economic/mesh/pipeline_state.json": "outputs/sweeps/economic/02-seasonal-mesh/pipeline_state.json",
    "operation/economic/mesh/champion_mesh.csv": "outputs/sweeps/economic/02-seasonal-mesh/champion_mesh.csv",
    "operation/economic/mesh/champion_mesh.json": "outputs/sweeps/economic/02-seasonal-mesh/champion_mesh.json",
    "operation/economic/mesh/factor_effects.csv": "outputs/sweeps/economic/02-seasonal-mesh/factor_effects.csv",
    "operation/economic/forecasting/summary.csv": "outputs/sweeps/economic/03-forecast-operation/summary.csv",
    "operation/economic/forecasting/pipeline_state.json": "outputs/sweeps/economic/03-forecast-operation/pipeline_state.json",
    "operation/economic/recourse/summary.csv": "outputs/sweeps/economic/04-recourse/summary.csv",
    "operation/economic/recourse/pipeline_state.json": "outputs/sweeps/economic/04-recourse/pipeline_state.json",
    "operation/economic/robustness/summary.csv": "outputs/sweeps/economic/05-robustness/summary.csv",
    "operation/economic/robustness/pipeline_state.json": "outputs/sweeps/economic/05-robustness/pipeline_state.json",
    "operation/economic/baselines/summary.csv": "outputs/sweeps/economic/06-baselines/summary.csv",
    "operation/economic/baselines/baseline_summary.csv": "outputs/sweeps/economic/06-baselines/baseline_summary.csv",
    "operation/economic/baselines/controller_comparison_120d.csv": "outputs/sweeps/economic/06-baselines/controller_comparison_120d.csv",
    "operation/economic/baselines/manifest.json": "outputs/sweeps/economic/06-baselines/manifest.json",
    "operation/economic/baselines/pipeline_state.json": "outputs/sweeps/economic/06-baselines/pipeline_state.json",
    "operation/critical/provenance/campaign.json": "operation-campaigns/critical_50/campaign.json",
    "operation/critical/provenance/sizing_artifact.json": "operation-campaigns/critical_50/sizing_artifact.json",
    "operation/full/provenance/campaign.json": "operation-campaigns/full_100/campaign.json",
    "operation/full/provenance/sizing_artifact.json": "operation-campaigns/full_100/sizing_artifact.json",
}

TREES = {
    "operation/economic/mesh/reference": "outputs/sweeps/economic/02-seasonal-mesh/reference",
    "operation/economic/mesh/publication": "outputs/sweeps/economic/02-seasonal-mesh/publication",
    "operation/economic/figures": "outputs/sweeps/economic/figures",
}

READMES = {
    "README.md": """# Definitive paper results

This directory is the canonical, versioned result package. It contains final
summaries, sizing decisions, tables, figures, and provenance. Local raw outputs
remain under the ignored `outputs/` directory and are not duplicated here.

- `sizing/`: economic, sensitivity, and resilience-oriented sizing results.
- `forecasting/`: corrected causal forecast evaluation.
- `noise/`: paired actuator-noise robustness results.
- `operation/`: operational results grouped by sizing policy.

Promote audited local outputs with `python scripts/build_paper_results.py`. The
root `manifest.json` records the source and SHA-256 hash of every artifact.
""",
    "sizing/README.md": """# Sizing results

`economic/` contains the degradation comparison used by the original economic
operation campaign. `sensitivity/` contains the eleven-case parameter study.
`resilience/` contains the completed economic, critical-load, and full-load
resilience sizing cases.
""",
    "forecasting/README.md": """# Forecasting results

These are the causal completed-hour v2 results. They supersede the earlier
pre-causal forecast evaluation. The origin-level Parquet checkpoint remains in
`outputs/forecasting-causal-v2/`; this package keeps the aggregate metrics,
figures, parameters, and evaluation manifest.
""",
    "noise/README.md": """# BESS actuator-noise robustness

The current experiment compares the base case with zero-mean Gaussian actuator
noise having standard deviations equal to 5% and 10% of rated BESS power. It
covers prototype and stochastic controllers over four seasonal 10-day windows,
for 24 paired cases in total.

`summary.csv` contains the audited source rows. `effects.csv` reports paired
changes relative to the corresponding base controller and seasonal window.
This is sufficient as an initial robustness check, but the experiment should be
expanded for the new critical and full sizing policies before drawing general
conclusions about larger batteries.
""",
    "operation/README.md": """# Operational results

Operational results are grouped by the sizing policy applied to the plant:

- `economic/`: complete 120-day mesh campaign and downstream analyses.
- `critical/`: operation sweep pending; provenance inputs are already frozen.
- `full/`: operation sweep pending; provenance inputs are already frozen.

The perfect-information controller is retained only as a target. Prototype and
LSTM are the implementable MPC candidates.
""",
    "operation/economic/README.md": """# Economic operation campaign

This campaign is complete. It includes 972 mesh--controller--month runs over 12
paired 10-day windows, the audited selected-mesh package, forecast comparison,
recourse study, robustness study, rule-based baselines, and publication
artifacts. `provenance/sizing_artifact.json` is the exact sizing file
used by these runs; `operational_parameters.json` is its materialized plant
configuration. Raw trajectories and stochastic plans remain under `outputs/`.
""",
    "operation/critical/README.md": """# Critical-load operation campaign

Status: pending. No operational result is reported here yet. The frozen sizing
artifact and campaign definition are available under `provenance/`; they target
the `critical_50` sizing case. Populate this directory only after the complete
paired sweep passes its audits.
""",
    "operation/full/README.md": """# Full-resilience operation campaign

Status: pending. No operational result is reported here yet. The frozen sizing
artifact and campaign definition are available under `provenance/`; they target
the `full_100` sizing case. Populate this directory only after the complete
paired sweep passes its audits.
""",
}


def csv_truth(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def campaign_completion(campaign_id: str) -> tuple[bool, str]:
    root = ROOT / "outputs" / "sweeps" / campaign_id
    if not root.exists():
        return False, "operational sweep not run"
    definition_path = ROOT / "operation-campaigns" / campaign_id / "campaign.json"
    sizing_definition = ROOT / "operation-campaigns" / campaign_id / "sizing_artifact.json"
    sizing_input = root / "campaign-inputs" / "sizing_artifact.json"
    parameter_input = root / "campaign-inputs" / "parameters.json"
    operational_input = root / "campaign-inputs" / "operational_parameters.json"
    required = (
        root / "campaign_manifest.json",
        root / "champion_mesh.json",
        root / "02-seasonal-mesh" / "summary.csv",
        root / "02-seasonal-mesh" / "reference" / "selected_mesh_120d_summary.csv",
        root / "02-seasonal-mesh" / "publication" / "manifest.json",
        root / "06-baselines" / "summary.csv",
        root / "06-baselines" / "controller_comparison_120d.csv",
        root / "06-baselines" / "manifest.json",
        sizing_input,
        parameter_input,
        operational_input,
        definition_path,
        sizing_definition,
    )
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
    if missing:
        return False, "missing " + ", ".join(missing)

    manifest = json.loads(required[0].read_text(encoding="utf-8"))
    if manifest.get("campaign_id") != campaign_id:
        return False, "campaign manifest id mismatch"
    portable = manifest.get("portable_inputs", {}) or {}
    sizing_hash = str(portable.get("sizing_artifact_sha256", ""))
    parameter_hash = str(portable.get("parameters_sha256", ""))
    if not sizing_hash or not parameter_hash:
        return False, "campaign manifest has incomplete input hashes"
    if sha256(sizing_input) != sizing_hash or sha256(parameter_input) != parameter_hash:
        return False, "frozen campaign input hash mismatch"
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    if definition.get("campaign_id") != campaign_id:
        return False, "versioned campaign definition id mismatch"
    if definition.get("sizing_artifact_sha256") != sizing_hash:
        return False, "versioned and executed sizing artifacts differ"
    if sha256(sizing_definition) != sizing_hash:
        return False, "versioned sizing artifact hash mismatch"

    with required[2].open(encoding="utf-8-sig", newline="") as stream:
        mesh = list(csv.DictReader(stream))
    if len(mesh) != 972:
        return False, f"expected 972 mesh rows, found {len(mesh)}"
    if any(row.get("campaign_id") != campaign_id for row in mesh):
        return False, "mesh rows have mixed campaign ids"
    if any(row.get("sizing_source_sha256") != sizing_hash for row in mesh):
        return False, "mesh rows have mixed sizing artifacts"
    if any(not csv_truth(row.get("audit_pass", "")) for row in mesh):
        return False, "at least one mesh row failed audit"

    baseline_path = root / "06-baselines" / "summary.csv"
    with baseline_path.open(encoding="utf-8-sig", newline="") as stream:
        baselines = list(csv.DictReader(stream))
    if len(baselines) != 48:
        return False, f"expected 48 baseline rows, found {len(baselines)}"
    if any(row.get("campaign_id") != campaign_id for row in baselines):
        return False, "baseline rows have mixed campaign ids"
    if any(row.get("sizing_source_sha256") != sizing_hash for row in baselines):
        return False, "baseline rows have mixed sizing artifacts"
    if any(not csv_truth(row.get("audit_pass", "")) for row in baselines):
        return False, "at least one baseline row failed audit"
    return True, "972 mesh cases and 48 paired baselines audited"


def add_completed_campaign(
    label: str,
    campaign_id: str,
    files: dict[str, str],
    trees: dict[str, str],
) -> None:
    source = f"outputs/sweeps/{campaign_id}"
    destination = f"operation/{label}"
    entries = {
        "campaign_manifest.json": "campaign_manifest.json",
        "champion_mesh.json": "champion_mesh.json",
        "mesh/summary.csv": "02-seasonal-mesh/summary.csv",
        "mesh/pipeline_state.json": "02-seasonal-mesh/pipeline_state.json",
        "mesh/champion_mesh.csv": "02-seasonal-mesh/champion_mesh.csv",
        "mesh/champion_mesh.json": "02-seasonal-mesh/champion_mesh.json",
        "mesh/factor_effects.csv": "02-seasonal-mesh/factor_effects.csv",
        "baselines/summary.csv": "06-baselines/summary.csv",
        "baselines/baseline_summary.csv": "06-baselines/baseline_summary.csv",
        "baselines/controller_comparison_120d.csv": "06-baselines/controller_comparison_120d.csv",
        "baselines/manifest.json": "06-baselines/manifest.json",
        "baselines/pipeline_state.json": "06-baselines/pipeline_state.json",
        "provenance/parameters.json": "campaign-inputs/parameters.json",
        "provenance/operational_parameters.json": "campaign-inputs/operational_parameters.json",
    }
    files.update({f"{destination}/{dst}": f"{source}/{src}" for dst, src in entries.items()})
    trees[f"{destination}/mesh/reference"] = f"{source}/02-seasonal-mesh/reference"
    trees[f"{destination}/mesh/publication"] = f"{source}/02-seasonal-mesh/publication"


def campaign_readme(label: str, campaign_id: str, complete: bool, detail: str) -> str:
    title = {"critical": "Critical-load", "full": "Full-resilience"}[label]
    if complete:
        status = (
            f"Status: complete. Campaign `{campaign_id}` contains {detail}. Its exact "
            "executed sizing artifact and parameter file are stored under `provenance/`."
        )
    else:
        status = (
            f"Status: pending. Campaign `{campaign_id}` has not been promoted because it is "
            f"incomplete ({detail}). Frozen inputs remain under `provenance/`."
        )
    return f"# {title} operation campaign\n\n{status}\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_path(relative: str) -> Path:
    path = (PAPER / relative).resolve()
    if path != PAPER.resolve() and PAPER.resolve() not in path.parents:
        raise RuntimeError(f"Refusing path outside paper package: {path}")
    return path


def normalize_path_text(value: str) -> str:
    for prefix in (f"{ROOT.as_posix()}/", f"{ROOT}\\"):
        if value.startswith(prefix):
            value = value[len(prefix):].replace("\\", "/")
            break
    return value.replace("Results\\", "outputs/").replace("Results/", "outputs/")


def normalize_json_paths(value):
    if isinstance(value, dict):
        return {key: normalize_json_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_json_paths(item) for item in value]
    if isinstance(value, str):
        return normalize_path_text(value)
    return value


def copy_with_normalized_provenance(source: Path, destination: Path) -> None:
    if source.suffix.lower() not in TEXT_SUFFIXES:
        shutil.copy2(source, destination)
        return
    if source.suffix.lower() == ".json":
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        else:
            destination.write_text(
                json.dumps(normalize_json_paths(payload), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            return
    text = source.read_text(encoding="utf-8")
    absolute_results = (ROOT / "Results").as_posix()
    absolute_outputs = (ROOT / "outputs").as_posix()
    text = text.replace(f"{ROOT.as_posix()}/", "")
    text = text.replace(f"{ROOT}\\", "")
    text = text.replace(f"{absolute_results}/", "outputs/")
    text = text.replace(f"{absolute_outputs}/", "outputs/")
    text = text.replace("Results/", "outputs/").replace("Results\\", "outputs/")
    destination.write_text(text, encoding="utf-8")


def remove_previous_managed_files() -> None:
    if not MANIFEST.exists():
        return
    previous = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for item in previous.get("artifacts", []):
        path = package_path(item["destination"])
        if path.is_file():
            path.unlink()
    for path in sorted(PAPER.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def copy_artifact(destination: str, source: str, artifacts: list[dict]) -> None:
    source_path = ROOT / source
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination_path = package_path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    copy_with_normalized_provenance(source_path, destination_path)
    artifacts.append({
        "destination": destination,
        "source": source,
        "source_sha256": sha256(source_path),
        "packaged_sha256": sha256(destination_path),
    })


def copy_tree(destination: str, source: str, artifacts: list[dict]) -> None:
    source_root = ROOT / source
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    for source_path in sorted(path for path in source_root.rglob("*") if path.is_file()):
        relative = source_path.relative_to(source_root).as_posix()
        copy_artifact(f"{destination}/{relative}", f"{source}/{relative}", artifacts)


def write_generated(destination: str, content: str, artifacts: list[dict], source: str | None = None) -> None:
    path = package_path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    artifacts.append({
        "destination": destination,
        "source": source,
        "source_sha256": sha256(ROOT / source) if source else None,
        "packaged_sha256": sha256(path),
    })


def write_noise_tables(artifacts: list[dict]) -> None:
    source_label = "outputs/sweeps/economic/05-robustness/summary.csv"
    source = ROOT / source_label
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row["variant"] in {"base", "noise_005", "noise_010"}]
    if len(selected) != 24 or any(row["audit_pass"].lower() != "true" for row in selected):
        raise RuntimeError("Expected 24 audited base/noise robustness rows")

    summary_path = package_path("noise/summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    artifacts.append({
        "destination": "noise/summary.csv",
        "source": source_label,
        "source_sha256": sha256(source),
        "packaged_sha256": sha256(summary_path),
    })

    indexed = {(row["window"], row["controller_name"], row["variant"]): row for row in selected}
    effects = []
    metrics = (
        "operation_total_cost", "operation_grid_reliability_cost",
        "operation_wear_cost", "operation_bess_throughput_kwh",
    )
    for row in selected:
        if row["variant"] == "base":
            continue
        base = indexed[(row["window"], row["controller_name"], "base")]
        effect = {
            "variant": row["variant"],
            "window": row["window"],
            "controller": row["controller_name"],
        }
        for metric in metrics:
            base_value = float(base[metric])
            noisy_value = float(row[metric])
            effect[f"base_{metric}"] = base_value
            effect[f"noisy_{metric}"] = noisy_value
            effect[f"delta_{metric}"] = noisy_value - base_value
            effect[f"delta_pct_{metric}"] = (
                100.0 * (noisy_value - base_value) / base_value if base_value else ""
            )
        effects.append(effect)

    effects_path = package_path("noise/effects.csv")
    with effects_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(effects[0]))
        writer.writeheader()
        writer.writerows(effects)
    artifacts.append({
        "destination": "noise/effects.csv",
        "source": source_label,
        "source_sha256": sha256(source),
        "packaged_sha256": sha256(effects_path),
    })


def verify_package() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    if manifest["artifact_count"] != len(artifacts):
        raise RuntimeError("Paper manifest artifact count is inconsistent")
    for item in artifacts:
        destination = package_path(item["destination"])
        if not destination.is_file() or sha256(destination) != item["packaged_sha256"]:
            raise RuntimeError(f"Invalid packaged artifact: {destination}")
        if item["source"]:
            source = ROOT / item["source"]
            if not source.is_file() or sha256(source) != item["source_sha256"]:
                raise RuntimeError(f"Invalid source artifact: {source}")


def main() -> None:
    files = dict(FILES)
    trees = dict(TREES)
    readmes = dict(READMES)
    operation_status = {}
    operation_details = {}
    for label, campaign_id in OPERATION_CAMPAIGNS.items():
        complete, detail = campaign_completion(campaign_id)
        operation_status[label] = "complete" if complete else "pending"
        operation_details[label] = detail
        if label == "economic" and not complete:
            raise RuntimeError(f"Economic operation campaign is incomplete: {detail}")
        if label != "economic" and complete:
            add_completed_campaign(label, campaign_id, files, trees)
            readmes[f"operation/{label}/README.md"] = campaign_readme(
                label, campaign_id, complete, detail
            )
        elif label != "economic":
            readmes[f"operation/{label}/README.md"] = campaign_readme(
                label, campaign_id, complete, detail
            )

    readmes["operation/README.md"] = (
        "# Operational results\n\nOperational results are isolated by sizing policy. "
        "Each completed campaign contains its own mesh sweep, selected reference, "
        "baselines, publication artifacts, and exact executed inputs.\n\n"
        + "\n".join(
            f"- `{label}/`: {operation_status[label]} ({operation_details[label]})."
            for label in OPERATION_CAMPAIGNS
        )
        + "\n"
    )

    missing_files = [source for source in files.values() if not (ROOT / source).is_file()]
    missing_trees = [source for source in trees.values() if not (ROOT / source).is_dir()]
    if missing_files or missing_trees:
        missing = ", ".join((*missing_files, *missing_trees))
        raise FileNotFoundError(f"Paper package sources are missing: {missing}")

    remove_previous_managed_files()
    artifacts: list[dict] = []
    for destination, source in files.items():
        copy_artifact(destination, source, artifacts)
    for destination, source in trees.items():
        copy_tree(destination, source, artifacts)
    write_noise_tables(artifacts)
    for destination, content in readmes.items():
        write_generated(destination, content, artifacts)

    oversized = [item for item in artifacts if package_path(item["destination"]).stat().st_size >= GITHUB_FILE_LIMIT_BYTES]
    if oversized:
        names = ", ".join(item["destination"] for item in oversized)
        raise RuntimeError(f"Paper artifacts exceed the GitHub 100 MiB limit: {names}")
    artifact_bytes = sum(package_path(item["destination"]).stat().st_size for item in artifacts)
    largest_bytes = max(package_path(item["destination"]).stat().st_size for item in artifacts)
    manifest = {
        "package_version": PACKAGE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(artifacts),
        "artifact_bytes": artifact_bytes,
        "largest_artifact_bytes": largest_bytes,
        "storage_policy": {
            "canonical_results": "paper/",
            "local_raw_workspace": "outputs/",
            "raw_workspace_tracked_by_git": False,
        },
        "operation_status": operation_status,
        "artifacts": sorted(artifacts, key=lambda item: item["destination"]),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    verify_package()
    print(f"Packaged and verified {len(artifacts)} artifacts under {PAPER}")


if __name__ == "__main__":
    main()
