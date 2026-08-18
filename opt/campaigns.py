"""Campaign layout and provenance validation utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CampaignProvenance:
    campaign_id: str
    root: Path
    manifest_path: Path
    sizing_sha256: str
    parameters_sha256: str
    manifest: dict[str, Any]

    @property
    def sizing_source_sha256(self) -> str:
        return self.sizing_sha256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def campaign_layout(root: Path) -> dict[str, Path]:
    root = root.resolve()
    seasonal = root / "02-seasonal-mesh"
    return {
        "root": root,
        "may": root / "01-mesh",
        "seasonal": seasonal,
        "summary": seasonal / "summary.csv",
        "champion": root / "champion_mesh.json",
        "reference": seasonal / "reference",
        "publication": seasonal / "publication",
        "baselines": root / "06-baselines",
        "manifest": root / "campaign_manifest.json",
    }


def load_campaign_provenance(
    campaign_root: Path, expected_campaign_id: str | None = None,
) -> CampaignProvenance:
    layout = campaign_layout(campaign_root)
    manifest_path = layout["manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Campaign manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    campaign_id = str(manifest.get("campaign_id", "")).strip()
    if not campaign_id:
        raise RuntimeError(f"Campaign manifest has no campaign_id: {manifest_path}")
    if expected_campaign_id and campaign_id != expected_campaign_id:
        raise RuntimeError(
            f"Campaign id mismatch: expected {expected_campaign_id!r}, found {campaign_id!r}"
        )

    portable = manifest.get("portable_inputs", {}) or {}
    sizing_sha256 = str(portable.get("sizing_artifact_sha256", "")).strip()
    parameters_sha256 = str(portable.get("parameters_sha256", "")).strip()
    if not sizing_sha256 or not parameters_sha256:
        raise RuntimeError(f"Campaign manifest has incomplete portable inputs: {manifest_path}")

    for name, expected in (
        ("sizing_artifact.json", sizing_sha256),
        ("parameters.json", parameters_sha256),
    ):
        path = layout["root"] / "campaign-inputs" / name
        if not path.is_file():
            raise FileNotFoundError(f"Portable campaign input not found: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"Portable input hash mismatch for {path}: expected {expected}, found {observed}"
            )

    return CampaignProvenance(
        campaign_id=campaign_id,
        root=layout["root"],
        manifest_path=manifest_path,
        sizing_sha256=sizing_sha256,
        parameters_sha256=parameters_sha256,
        manifest=manifest,
    )


def validate_summary_provenance(
    summary: pd.DataFrame, provenance: CampaignProvenance, label: str,
) -> None:
    required = {"campaign_id", "sizing_source_sha256"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise RuntimeError(f"{label} is missing provenance columns: {', '.join(missing)}")

    campaign_ids = set(summary["campaign_id"].dropna().astype(str))
    if campaign_ids != {provenance.campaign_id}:
        raise RuntimeError(
            f"{label} mixes campaign ids: expected {provenance.campaign_id!r}, "
            f"found {sorted(campaign_ids)}"
        )
    sizing_hashes = set(summary["sizing_source_sha256"].dropna().astype(str))
    if sizing_hashes != {provenance.sizing_sha256}:
        raise RuntimeError(
            f"{label} mixes sizing artifacts: expected {provenance.sizing_sha256}, "
            f"found {sorted(sizing_hashes)}"
        )


def validate_parameter_provenance(
    params: dict[str, Any], provenance: CampaignProvenance, label: str,
) -> None:
    applied = params.get("sizing_case_applied", {}) or {}
    observed = str(applied.get("source_sha256", ""))
    if observed != provenance.sizing_sha256:
        raise RuntimeError(
            f"{label} uses sizing artifact {observed or '<missing>'}; "
            f"campaign {provenance.campaign_id!r} requires {provenance.sizing_sha256}"
        )
