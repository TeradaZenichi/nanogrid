"""Audit the versioned paper result package before committing it."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MANIFEST = PAPER / "manifest.json"
GITHUB_FILE_LIMIT_BYTES = 100 * 1024 * 1024
LEGACY_MARKERS = ("Results/", "Results\\", "Results-sweeps", "results-old")
ABSOLUTE_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[/\\]")
TEXT_SUFFIXES = {".csv", ".json", ".md", ".tex", ".txt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit() -> dict:
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", [])
    failures = []
    for item in artifacts:
        path = PAPER / item["destination"]
        if not path.is_file():
            failures.append(f"missing: {item['destination']}")
            continue
        if path.stat().st_size >= GITHUB_FILE_LIMIT_BYTES:
            failures.append(f"at least 100 MiB: {item['destination']}")
        if sha256(path) != item["packaged_sha256"]:
            failures.append(f"hash mismatch: {item['destination']}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            marker = next((value for value in LEGACY_MARKERS if value in text), None)
            if marker:
                failures.append(f"legacy marker {marker!r}: {item['destination']}")
            if ABSOLUTE_WINDOWS_PATH.search(text):
                failures.append(f"absolute Windows path: {item['destination']}")
    unlisted = sorted(
        path.relative_to(PAPER).as_posix()
        for path in PAPER.rglob("*")
        if path.is_file() and path != MANIFEST
        and path.relative_to(PAPER).as_posix() not in {item["destination"] for item in artifacts}
    )
    if unlisted:
        failures.append(f"unlisted artifacts: {', '.join(unlisted)}")
    if failures:
        raise RuntimeError("Paper package audit failed:\n- " + "\n- ".join(failures))
    files = [path for path in PAPER.rglob("*") if path.is_file()]
    return {
        "artifacts": len(artifacts),
        "files_including_manifest": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "largest_file_bytes": max(path.stat().st_size for path in files),
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
