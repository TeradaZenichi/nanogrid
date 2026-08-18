"""Migrate large result artifacts to validated Zstandard-compressed Parquet."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opt.artifacts import (
    read_operation,
    read_stochastic_plan,
    write_operation,
    write_stochastic_plan,
)


SCHEMA_VERSION = 1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_frames_equal(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True), actual.reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=1e-13, atol=1e-13,
    )


def safe_unlink(path: Path, root: Path) -> None:
    resolved = path.resolve()
    if root.resolve() not in resolved.parents:
        raise RuntimeError(f"Refusing to remove file outside migration root: {resolved}")
    resolved.unlink()


def conversion_record(kind: str, source: Path, target: Path, source_data: bytes) -> dict:
    return {
        "kind": kind,
        "source": source.relative_to(ROOT).as_posix(),
        "target": target.relative_to(ROOT).as_posix(),
        "source_size_bytes": len(source_data),
        "target_size_bytes": target.stat().st_size,
        "source_sha256": sha256_bytes(source_data),
        "target_sha256": sha256_file(target),
    }


def migrate_operation(source: Path, root: Path) -> dict:
    source_data = source.read_bytes()
    expected = pd.read_csv(io.BytesIO(source_data))
    target = write_operation(expected, source.parent, remove_legacy=False)
    try:
        assert_frames_equal(expected, read_operation(target))
        record = conversion_record("operation", source, target, source_data)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    safe_unlink(source, root)
    return record


def migrate_plan(source: Path, root: Path) -> dict:
    source_data = source.read_bytes()
    expected = json.loads(source_data)
    target = write_stochastic_plan(expected, source, remove_legacy=False)
    try:
        if read_stochastic_plan(target) != expected:
            raise ValueError(f"Stochastic plan changed during conversion: {source}")
        record = conversion_record("stochastic_plan", source, target, source_data)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    safe_unlink(source, root)
    return record


def migrate_table(source: Path, root: Path, kind: str) -> dict:
    source_data = source.read_bytes()
    expected = pd.read_csv(io.BytesIO(source_data))
    target = source.with_suffix(".parquet")
    temporary = source.parent / f".{target.name}.tmp"
    expected.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
    os.replace(temporary, target)
    try:
        assert_frames_equal(expected, pd.read_parquet(target))
        record = conversion_record(kind, source, target, source_data)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    safe_unlink(source, root)
    return record


def sizing_sort(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [key for key in ("year", "slot", "scenario", "contingency") if key in frame]
    return frame.sort_values(keys).reset_index(drop=True) if keys else frame.reset_index(drop=True)


def migrate_sizing_case(operations: Path, root: Path) -> list[dict]:
    source = operations / "operations_all_years.csv"
    target = operations / "operations_all_years.parquet"
    if not source.exists():
        return []
    source_data = source.read_bytes()
    expected = pd.read_csv(io.BytesIO(source_data))
    temporary = operations / ".operations_all_years.parquet.tmp"
    expected.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
    os.replace(temporary, target)
    try:
        assert_frames_equal(expected, pd.read_parquet(target))
        records = [conversion_record("sizing_operations", source, target, source_data)]
        for annual in sorted(operations.glob("operations_year_*.csv")):
            annual_data = annual.read_bytes()
            values = pd.read_csv(io.BytesIO(annual_data))
            years = pd.to_numeric(values["year"], errors="raise").unique()
            if len(years) != 1:
                raise ValueError(f"Expected one year in {annual}")
            reference = expected[pd.to_numeric(expected["year"]) == years[0]]
            assert_frames_equal(sizing_sort(reference), sizing_sort(values))
            records.append({
                "kind": "removed_redundant_sizing_year",
                "source": annual.relative_to(ROOT).as_posix(),
                "target": target.relative_to(ROOT).as_posix(),
                "source_size_bytes": len(annual_data),
                "target_size_bytes": 0,
                "source_sha256": sha256_bytes(annual_data),
                "target_sha256": sha256_file(target),
            })
    except Exception:
        target.unlink(missing_ok=True)
        raise
    safe_unlink(source, root)
    for annual in sorted(operations.glob("operations_year_*.csv")):
        safe_unlink(annual, root)
    return records


def write_manifest(path: Path, started: str, records: list[dict], status: str) -> None:
    source_bytes = sum(item["source_size_bytes"] for item in records)
    target_bytes = sum(
        item["target_size_bytes"] for item in records
        if item["kind"] != "removed_redundant_sizing_year"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "started_at_utc": started,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "converted_artifacts": len(records),
        "source_size_bytes": source_bytes,
        "target_size_bytes": target_bytes,
        "reduction_ratio": source_bytes / target_bytes if target_bytes else None,
        "records": records,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def inventory(root: Path) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    operations = sorted(root.rglob("operation_final.csv"))
    plans = sorted(root.rglob("plans/plan_*.json"))
    forecasts = sorted(root.rglob("prototype_eval_checkpoint.csv"))
    sizing = sorted(
        path.parent for path in (root / "sizing").glob("*/operations/operations_all_years.csv")
    ) if (root / "sizing").exists() else []
    return operations, plans, forecasts, sizing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    root = args.root.resolve()
    workspace = ROOT.resolve()
    if root != workspace and workspace not in root.parents:
        raise RuntimeError(f"Migration root must be inside the repository: {root}")
    operations, plans, forecasts, sizing = inventory(root)
    jobs = [("operation", path) for path in operations]
    jobs.extend(("plan", path) for path in plans)
    jobs.extend(("forecast_checkpoint", path) for path in forecasts)
    if args.limit is not None:
        jobs = jobs[:args.limit]
        sizing = []
    source_size = sum(path.stat().st_size for _, path in jobs)
    source_size += sum(
        path.stat().st_size for directory in sizing
        for path in directory.glob("operations_*.csv")
    )
    print(
        f"Found {len(operations)} operation CSVs, {len(plans)} plan JSONs, "
        f"{len(forecasts)} forecast checkpoints, and {len(sizing)} sizing cases "
        f"({source_size / 1e9:.2f} GB selected)."
    )
    if not args.execute:
        print("Dry run only. Add --execute to convert and remove validated source files.")
        return

    manifest_path = root / "parquet_migration_manifest.json"
    previous = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists() else {}
    )
    started = previous.get("started_at_utc", datetime.now(timezone.utc).isoformat())
    records: list[dict] = list(previous.get("records", []))
    completed_before = len(records)
    total = len(jobs) + len(sizing)
    t0 = time.perf_counter()
    try:
        for index, (kind, source) in enumerate(jobs, start=1):
            if kind == "operation":
                record = migrate_operation(source, root)
            elif kind == "plan":
                record = migrate_plan(source, root)
            else:
                record = migrate_table(source, root, kind)
            records.append(record)
            if index % 25 == 0 or index == total:
                elapsed = time.perf_counter() - t0
                eta = elapsed / index * (total - index)
                print(
                    f"[{index}/{total} remaining] elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
                write_manifest(manifest_path, started, records, "running")
        for offset, directory in enumerate(sizing, start=len(jobs) + 1):
            records.extend(migrate_sizing_case(directory, root))
            elapsed = time.perf_counter() - t0
            eta = elapsed / offset * (total - offset)
            print(f"[{offset}/{total}] elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
            write_manifest(manifest_path, started, records, "running")
    except Exception:
        write_manifest(manifest_path, started, records, "failed")
        raise
    write_manifest(manifest_path, started, records, "complete")
    elapsed = time.perf_counter() - t0
    print(
        f"Migration complete in {elapsed:.1f}s; {completed_before} prior records "
        f"and {len(records) - completed_before} new records. Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
