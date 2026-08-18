"""Migrate the completed economic operation campaign into its isolated layout."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opt.campaigns import sha256_file
from opt.operation import parameter_fingerprint
from opt.utils import load_sized_parameters_from_artifact


LEGACY_ITEMS = (
    "01-mesh",
    "02-seasonal-mesh",
    "03-forecast-operation",
    "04-recourse",
    "05-robustness",
    "06-baselines",
    "campaign_manifest.json",
    "champion_mesh.json",
)


def migrate_paths(outputs: Path, campaign_root: Path) -> None:
    campaign_root.mkdir(parents=True, exist_ok=True)
    for name in LEGACY_ITEMS:
        source = outputs / name
        destination = campaign_root / name
        if source.exists() and destination.exists():
            raise RuntimeError(f"Both legacy and campaign paths exist for {name}")
        if source.exists():
            shutil.move(str(source), str(destination))


def summary_sizing_hash(campaign_root: Path) -> str:
    summary = campaign_root / "02-seasonal-mesh" / "summary.csv"
    with summary.open(encoding="utf-8-sig", newline="") as stream:
        hashes = {
            row["sizing_source_sha256"].strip()
            for row in csv.DictReader(stream)
            if row.get("sizing_source_sha256", "").strip()
        }
    if len(hashes) != 1:
        raise RuntimeError(f"Expected one sizing hash in {summary}, found {sorted(hashes)}")
    return hashes.pop()


def find_sizing_artifact(outputs: Path, expected_hash: str) -> Path:
    candidates = sorted(outputs.glob("sizing/**/sizing_decision_variables.json"))
    matches = [path for path in candidates if sha256_file(path) == expected_hash]
    if len(matches) != 1:
        labels = [path.relative_to(ROOT).as_posix() for path in matches]
        raise RuntimeError(
            f"Expected one sizing artifact with hash {expected_hash}, found {labels}"
        )
    return matches[0]


def add_campaign_column(path: Path, campaign_id: str) -> bool:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    if "sizing_source_sha256" not in fieldnames or "campaign_id" in fieldnames:
        return False
    fieldnames.append("campaign_id")
    for row in rows:
        row["campaign_id"] = campaign_id
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return True


def update_summaries(campaign_root: Path, campaign_id: str) -> list[str]:
    changed = []
    for path in sorted(campaign_root.rglob("*.csv")):
        if add_campaign_column(path, campaign_id):
            changed.append(path.relative_to(campaign_root).as_posix())
    return changed


def freeze_inputs(
    campaign_root: Path,
    parameters: Path,
    sizing_artifact: Path,
) -> tuple[dict, Path, Path]:
    inputs = campaign_root / "campaign-inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    parameter_copy = inputs / "parameters.json"
    sizing_copy = inputs / "sizing_artifact.json"
    shutil.copy2(parameters, parameter_copy)
    shutil.copy2(sizing_artifact, sizing_copy)
    operational = load_sized_parameters_from_artifact(parameter_copy, sizing_copy)
    (inputs / "operational_parameters.json").write_text(
        json.dumps(operational, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return operational, parameter_copy, sizing_copy


def update_manifest(
    campaign_root: Path,
    campaign_id: str,
    operational: dict,
    parameter_copy: Path,
    sizing_copy: Path,
    changed: list[str],
) -> None:
    manifest_path = campaign_root / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    applied = operational["sizing_case_applied"]
    manifest.update(
        schema_version=2,
        campaign_id=campaign_id,
        output_root=campaign_root.relative_to(ROOT).as_posix(),
        base_parameter_fingerprint_sha256=parameter_fingerprint(operational),
        portable_inputs={
            "parameters": parameter_copy.relative_to(ROOT).as_posix(),
            "parameters_sha256": sha256_file(parameter_copy),
            "sizing_artifact": sizing_copy.relative_to(ROOT).as_posix(),
            "sizing_artifact_sha256": sha256_file(sizing_copy),
            "sizing_case": applied["case"],
            "pv_size_kw": applied["P_hat_PV_kw"],
            "bess_size_kwh": applied["E_hat_BESS_kwh"],
            "bess_power_kw": applied["BESS_Pmax_kw"],
        },
        layout_migration={
            "migrated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_layout": "outputs/{01-mesh,...,06-baselines}",
            "destination_layout": campaign_root.relative_to(ROOT).as_posix(),
            "campaign_columns_added": changed,
            "scientific_results_recomputed": False,
        },
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default="economic")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--parameters", type=Path, default=ROOT / "data" / "parameters.json")
    args = parser.parse_args()

    outputs = args.outputs.resolve()
    campaign_root = outputs / "sweeps" / args.campaign_id
    migrate_paths(outputs, campaign_root)
    expected_hash = summary_sizing_hash(campaign_root)
    sizing_artifact = find_sizing_artifact(outputs, expected_hash)
    changed = update_summaries(campaign_root, args.campaign_id)
    operational, parameter_copy, sizing_copy = freeze_inputs(
        campaign_root, args.parameters.resolve(), sizing_artifact
    )
    if operational["sizing_case_applied"]["source_sha256"] != expected_hash:
        raise RuntimeError("Frozen sizing artifact does not match the operation summaries")
    update_manifest(
        campaign_root, args.campaign_id, operational,
        parameter_copy, sizing_copy, changed,
    )
    print(campaign_root)


if __name__ == "__main__":
    main()
