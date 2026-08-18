from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from opt.campaigns import (
    campaign_layout,
    load_campaign_provenance,
    sha256_file,
    validate_parameter_provenance,
    validate_summary_provenance,
)


ROOT = Path(__file__).resolve().parents[1]


class CampaignLayoutTests(unittest.TestCase):
    def make_campaign(self, root: Path, campaign_id: str = "critical_50"):
        inputs = root / "campaign-inputs"
        inputs.mkdir(parents=True)
        sizing = inputs / "sizing_artifact.json"
        parameters = inputs / "parameters.json"
        sizing.write_text('{"size": 1}', encoding="utf-8")
        parameters.write_text('{"parameter": 2}', encoding="utf-8")
        manifest = {
            "campaign_id": campaign_id,
            "portable_inputs": {
                "sizing_artifact_sha256": sha256_file(sizing),
                "parameters_sha256": sha256_file(parameters),
            },
        }
        (root / "campaign_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return manifest

    def test_layout_isolated_under_one_campaign_root(self):
        root = Path("outputs/sweeps/critical_50")
        layout = campaign_layout(root)
        self.assertEqual(layout["may"], root.resolve() / "01-mesh")
        self.assertEqual(layout["baselines"], root.resolve() / "06-baselines")
        self.assertEqual(
            layout["publication"],
            root.resolve() / "02-seasonal-mesh" / "publication",
        )

    def test_provenance_accepts_only_matching_summary_and_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_campaign(root)
            provenance = load_campaign_provenance(root, "critical_50")
            sizing_hash = manifest["portable_inputs"]["sizing_artifact_sha256"]
            summary = pd.DataFrame({
                "campaign_id": ["critical_50"],
                "sizing_source_sha256": [sizing_hash],
            })
            validate_summary_provenance(summary, provenance, "test summary")
            validate_parameter_provenance(
                {"sizing_case_applied": {"source_sha256": sizing_hash}},
                provenance,
                "test parameters",
            )

            with self.assertRaises(RuntimeError):
                validate_summary_provenance(
                    summary.assign(campaign_id="full_100"), provenance, "mixed summary"
                )
            with self.assertRaises(RuntimeError):
                validate_parameter_provenance(
                    {"sizing_case_applied": {"source_sha256": "wrong"}},
                    provenance,
                    "wrong parameters",
                )

    def test_manifest_id_and_frozen_hashes_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_campaign(root)
            with self.assertRaises(RuntimeError):
                load_campaign_provenance(root, "full_100")
            (root / "campaign-inputs" / "sizing_artifact.json").write_text(
                '{"size": 2}', encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                load_campaign_provenance(root, "critical_50")

    def test_distributed_campaigns_use_isolated_output_roots(self):
        for campaign_id in ("economic", "critical_50", "full_100"):
            root = ROOT / "operation-campaigns" / campaign_id
            definition = json.loads((root / "campaign.json").read_text(encoding="utf-8"))
            self.assertEqual(definition["campaign_id"], campaign_id)
            self.assertEqual(
                definition["sizing_artifact_sha256"],
                sha256_file(root / "sizing_artifact.json"),
            )
            commands = definition["commands"]
            self.assertEqual(len(commands), 5)
            self.assertTrue(all(f"outputs/sweeps/{campaign_id}" in command for command in commands))
            self.assertTrue(all(f"--campaign-id {campaign_id}" in command for command in commands))
            self.assertIn("14_consolidate_reference_results.py", commands[2])
            self.assertIn("15_baseline_comparison.py", commands[3])
            self.assertIn("13_generate_publication_artifacts.py", commands[4])


if __name__ == "__main__":
    unittest.main()
