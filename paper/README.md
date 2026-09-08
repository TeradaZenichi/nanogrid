# Paper artifacts

This directory contains the compact, versionable evidence used by the paper.
It is rebuilt from audited outputs with:

```powershell
.\.venv\Scripts\python.exe scripts\build_publication_release.py
```

Contents:

- `artifacts/figures`: vector PDF figures using the Gulliver font;
- `artifacts/tables`: LaTeX tables;
- `artifacts/data/consolidated`: paper-level result tables and statistical outputs;
- `artifacts/data/mesh`: the 27-candidate decision tables and selected meshes;
- `artifacts/manifest.json`: source paths, byte sizes, and SHA-256 hashes.

Large per-timestep operational trajectories are intentionally not duplicated
here. Their GitHub/Zenodo separation is documented in `release/zenodo/README.md`.

