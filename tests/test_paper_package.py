import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
GITHUB_FILE_LIMIT_BYTES = 100 * 1024 * 1024


class PaperPackageTests(unittest.TestCase):
    def test_manifest_covers_a_github_safe_package(self):
        manifest = json.loads((PAPER / 'manifest.json').read_text(encoding='utf-8'))
        artifacts = manifest['artifacts']

        self.assertEqual(manifest['package_version'], 'paper-results-v3')
        self.assertEqual(manifest['artifact_count'], len(artifacts))
        self.assertEqual(manifest['storage_policy']['canonical_results'], 'paper/')
        self.assertEqual(
            manifest['operation_status'],
            {'economic': 'complete', 'critical': 'pending', 'full': 'pending'},
        )
        for item in artifacts:
            path = PAPER / item['destination']
            self.assertTrue(path.is_file(), item['destination'])
            self.assertLess(path.stat().st_size, GITHUB_FILE_LIMIT_BYTES)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, item['packaged_sha256'])
            if path.suffix.lower() in {'.csv', '.json', '.md', '.tex', '.txt'}:
                text = path.read_text(encoding='utf-8')
                self.assertNotIn('C:/', text, item['destination'])
                self.assertNotIn('C:\\', text, item['destination'])

        listed = {item['destination'] for item in artifacts}
        actual = {
            path.relative_to(PAPER).as_posix()
            for path in PAPER.rglob('*')
            if path.is_file() and path != PAPER / 'manifest.json'
        }
        self.assertEqual(actual, listed)


if __name__ == '__main__':
    unittest.main()
