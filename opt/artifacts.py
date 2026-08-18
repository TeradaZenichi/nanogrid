"""Read and write compact operational artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


OPERATION_PARQUET = "operation_final.parquet"
OPERATION_CSV = "operation_final.csv"
PLAN_FORMAT = "nanogrid-stochastic-plan-v1"
PLAN_METADATA_KEY = b"nanogrid_plan_payload"
PLAN_FORMAT_KEY = b"nanogrid_plan_format"


def _json_default(value: Any):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def operation_artifact_path(path: str | Path, required: bool = True) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    directory = path if path.suffix == "" else path.parent
    candidates = (directory / OPERATION_PARQUET, directory / OPERATION_CSV)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if required:
        raise FileNotFoundError(
            f"Missing operation artifact in {directory}; expected "
            f"{OPERATION_PARQUET} or {OPERATION_CSV}"
        )
    return candidates[0]


def read_operation(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    artifact = operation_artifact_path(path)
    if artifact.suffix == ".parquet":
        frame = pd.read_parquet(artifact)
        return frame.head(nrows) if nrows is not None else frame
    return pd.read_csv(artifact, nrows=nrows)


def _operation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.copy()
    if "timestamp" not in values.columns:
        index_name = values.index.name or "index"
        values = values.reset_index().rename(columns={index_name: "timestamp"})
    for column in values.select_dtypes(include="object"):
        if values[column].map(lambda item: isinstance(item, (dict, list, tuple))).any():
            values[column] = values[column].map(
                lambda item: json.dumps(
                    item, ensure_ascii=False, separators=(",", ":"), default=_json_default,
                )
                if isinstance(item, (dict, list, tuple)) else item
            )
    return values


def write_operation(
    frame: pd.DataFrame, directory: str | Path, remove_legacy: bool = True,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / OPERATION_PARQUET
    temporary = directory / f".{OPERATION_PARQUET}.tmp"
    _operation_frame(frame).to_parquet(
        temporary, engine="pyarrow", compression="zstd", index=False,
    )
    os.replace(temporary, target)
    legacy = directory / OPERATION_CSV
    if remove_legacy and legacy.exists():
        legacy.unlink()
    return target


def write_stochastic_plan(
    payload: dict[str, Any], path: str | Path, remove_legacy: bool = True,
) -> Path:
    path = Path(path).with_suffix(".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    rows = []
    for scenario, contingencies in payload["scenarios"].items():
        for contingency, actions in contingencies.items():
            rows.extend(
                {"scenario": scenario, "contingency": contingency, **action}
                for action in actions
            )
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[PLAN_FORMAT_KEY] = PLAN_FORMAT.encode("utf-8")
    metadata_payload = {key: value for key, value in payload.items() if key != "scenarios"}
    metadata[PLAN_METADATA_KEY] = json.dumps(
        metadata_payload, ensure_ascii=False, separators=(",", ":"), default=_json_default,
    ).encode("utf-8")
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
    os.replace(temporary, path)
    legacy = path.with_suffix(".json")
    if remove_legacy and legacy.exists():
        legacy.unlink()
    return path


def read_stochastic_plan(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    if metadata.get(PLAN_FORMAT_KEY, b"").decode("utf-8") != PLAN_FORMAT:
        raise ValueError(f"Unsupported stochastic plan format: {path}")
    payload = json.loads(metadata[PLAN_METADATA_KEY].decode("utf-8"))
    scenarios: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in table.to_pylist():
        scenario = row.pop("scenario")
        contingency = row.pop("contingency")
        scenarios.setdefault(scenario, {}).setdefault(contingency, []).append(row)
    payload["scenarios"] = scenarios
    return payload
