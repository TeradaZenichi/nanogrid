"""Rebuild sizing summaries and reports from completed case artifacts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SIZING_SCRIPT = ROOT / "1-sizing.py"
SIZING_ROOT = ROOT / "outputs" / "sizing-sweep"
VARIANTS = ("with-degradation", "without-degradation")
CASES = ("economic", "critical_50", "full_100")


def _load_sizing_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sizing_campaign", SIZING_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {SIZING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    sizing = _load_sizing_module()
    for variant in VARIANTS:
        output = SIZING_ROOT / variant
        manifest_path = output / "manifest.json"
        campaign = json.loads(manifest_path.read_text(encoding="utf-8"))
        campaign.pop("updated_at_utc", None)
        campaign.pop("cases", None)
        campaign["selected_cases"] = [
            case.__dict__ for case in sizing.CASES if case.case_id in CASES
        ]

        results = []
        for case in CASES:
            path = output / case / "sizing_decision_variables.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            if not result.get("has_loaded_solution"):
                raise RuntimeError(f"Incomplete sizing artifact: {path}")
            results.append(result)

        sizing._save_outputs(output, results, campaign)
        print(f"[{variant}] rebuilt {len(results)} sizing rows", flush=True)


if __name__ == "__main__":
    main()
