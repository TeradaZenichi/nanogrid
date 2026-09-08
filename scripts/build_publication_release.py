"""Assemble compact GitHub artifacts and inventory optional Zenodo datasets.

The default command is non-destructive: it copies the compact, publication-
facing artifacts into ``paper/artifacts`` and regenerates CSV inventories for
the ``core`` and ``full`` Zenodo tiers. Large ZIP packages are created only
when ``--package`` is explicitly requested.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "artifacts"
ZENODO = ROOT / "release" / "zenodo"
PACKAGES = ZENODO / "packages"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _files_under(path: Path) -> list[Path]:
    return sorted(item for item in path.rglob("*") if item.is_file())


def _paper_sources() -> list[tuple[Path, Path]]:
    analysis = ROOT / "outputs" / "paper-analysis"
    assets = analysis / "paper-assets"
    pairs: list[tuple[Path, Path]] = []
    for source in sorted((assets / "figures").glob("*.pdf")):
        pairs.append((source, Path("figures") / source.name))
    for source in sorted((assets / "tables").glob("*.tex")):
        pairs.append((source, Path("tables") / source.name))
    for name in (
        "sizing_degradation_comparison.csv",
        "sizing_autonomy_processed.csv",
        "controller_monthly_paired.csv",
        "controller_paired_statistics.csv",
        "robustness_summary.csv",
        "paper_result_summary.md",
        "manifest.json",
    ):
        source = analysis / name
        if source.is_file():
            pairs.append((source, Path("data") / "consolidated" / name))
    for case in ("economic", "critical_50", "full_100"):
        source_root = (
            ROOT / "outputs" / "operation-sweep" / "with-degradation" /
            case / "02-seasonal-mesh"
        )
        for name in ("champion_mesh.csv", "champion_mesh.json", "factor_effects.csv"):
            source = source_root / name
            if source.is_file():
                pairs.append((source, Path("data") / "mesh" / case / name))
    return pairs


def build_paper_artifacts() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for source, destination_relative in _paper_sources():
        destination = PAPER / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "path": destination_relative.as_posix(),
                "source": _relative(source),
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
    manifest = {
        "bundle": "github-paper-artifacts",
        "generated_by": "scripts/build_publication_release.py",
        "files": records,
    }
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return records


def _excluded(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith(".tmp")
        or ".tmp." in name
        or name.startswith(".pipeline_state")
        or "__pycache__" in path.parts
    )


def _add_tree(target: dict[str, Path], source_root: Path) -> None:
    if not source_root.is_dir():
        return
    for path in _files_under(source_root):
        if not _excluded(path):
            target[_relative(path)] = path


def _add_file(target: dict[str, Path], path: Path) -> None:
    if path.is_file() and not _excluded(path):
        target[_relative(path)] = path


def zenodo_files(tier: str) -> list[Path]:
    selected: dict[str, Path] = {}
    _add_tree(selected, ROOT / "outputs" / "sizing-sweep")

    analysis = ROOT / "outputs" / "paper-analysis"
    for source, _ in _paper_sources():
        _add_file(selected, source)

    operation = ROOT / "outputs" / "operation-sweep" / "with-degradation"
    baseline = ROOT / "outputs" / "baseline-sweep" / "with-degradation"
    if tier == "full":
        _add_tree(selected, operation)
        _add_tree(selected, baseline)
    elif tier == "core":
        _add_tree(selected, baseline)
        for case in ("economic", "critical_50", "full_100"):
            case_root = operation / case
            _add_tree(selected, case_root / "03-forecast-operation")
            _add_tree(selected, case_root / "campaign-inputs")
            for name in ("campaign_manifest.json", "champion_mesh.json"):
                _add_file(selected, case_root / name)
            seasonal = case_root / "02-seasonal-mesh"
            for name in ("summary.csv", "champion_mesh.csv", "champion_mesh.json", "factor_effects.csv"):
                _add_file(selected, seasonal / name)
        for case in ("critical_50", "full_100"):
            _add_tree(selected, operation / case / "05-robustness")
    else:
        raise ValueError(f"Unknown tier: {tier}")
    return [selected[key] for key in sorted(selected)]


def write_inventory(tier: str, files: list[Path]) -> Path:
    ZENODO.mkdir(parents=True, exist_ok=True)
    destination = ZENODO / f"inventory_{tier}.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("archive_path", "size_bytes"))
        writer.writeheader()
        for path in files:
            writer.writerow({"archive_path": _relative(path), "size_bytes": path.stat().st_size})
    return destination


def package_zenodo(tier: str, files: list[Path]) -> Path:
    PACKAGES.mkdir(parents=True, exist_ok=True)
    destination = PACKAGES / f"nanogrid-results-{tier}.zip"
    checksums: list[tuple[str, str]] = []
    with zipfile.ZipFile(destination, "w", allowZip64=True) as archive:
        for path in files:
            relative = _relative(path)
            compression = zipfile.ZIP_STORED if path.suffix.lower() == ".parquet" else zipfile.ZIP_DEFLATED
            archive.write(path, relative, compress_type=compression)
            checksums.append((_sha256(path), relative))
        checksum_text = "".join(f"{digest}  {name}\n" for digest, name in checksums)
        archive.writestr("checksums.sha256", checksum_text, compress_type=zipfile.ZIP_DEFLATED)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package", choices=("core", "full"),
        help="also build the selected large Zenodo ZIP under release/zenodo/packages",
    )
    args = parser.parse_args()

    paper_records = build_paper_artifacts()
    summaries = {}
    selections = {}
    for tier in ("core", "full"):
        files = zenodo_files(tier)
        inventory = write_inventory(tier, files)
        summaries[tier] = {
            "files": len(files),
            "size_bytes": sum(path.stat().st_size for path in files),
            "inventory": _relative(inventory),
        }
        selections[tier] = [_relative(path) for path in files]

    release_manifest = {
        "generated_by": "scripts/build_publication_release.py",
        "github_paper_bundle": {
            "files": len(paper_records),
            "size_bytes": sum(int(row["size_bytes"]) for row in paper_records),
            "path": "paper/artifacts",
        },
        "zenodo_tiers": summaries,
        "selection_policy": {
            "core": "champion operation, baselines, critical/full robustness, mesh summaries, sizing and paper data",
            "full": "complete audited operation and baseline trees plus sizing and paper data",
            "excluded": ["temporary files", ".pipeline_state temporary files", "Python caches"],
        },
    }
    (ZENODO / "release_manifest.json").write_text(
        json.dumps(release_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(release_manifest, indent=2, ensure_ascii=False))
    if args.package:
        package_path = package_zenodo(args.package, [ROOT / item for item in selections[args.package]])
        print(f"Built {package_path} ({package_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

