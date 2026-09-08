# GitHub Results Release

The complete scientific outputs are distributed as a GitHub Release asset and
remain outside the Git repository history.

Current asset:

- `nanogrid-results-full.zip`
- size: 1,324,698,430 bytes (1.234 GiB)
- SHA-256: `05124A7931E100A1188E13ACA7F7AA8A12E1D2ACACC2EA2494A3B5A3E14C785A`
- local build path: `release/zenodo/packages/nanogrid-results-full.zip`

The archive contains 18,400 scientific files plus `checksums.sha256`. It covers
the complete current sizing, degradation-aware operation, baseline, paper-data,
and temporal-mesh outputs. Temporary files, Python caches, `.venv`, `.git`,
`.old`, and `outputs/_old` are excluded.

Build or refresh it with:

```powershell
.\.venv\Scripts\python.exe scripts\build_publication_release.py --package full
```

The release must be attached to the same commit that contains the corresponding
paper artifacts and inventories.

