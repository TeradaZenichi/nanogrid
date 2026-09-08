# Zenodo data release

The large operational outputs remain under `outputs/` and are ignored by Git.
This directory versions their release policy and generated inventories without
duplicating the data.

## Release tiers

### Core dataset

The core dataset is the recommended paper companion. It contains:

- degradation-aware and no-degradation sizing artifacts;
- consolidated paper results;
- all champion-mesh monthly trajectories for ideal, LSTM, and prototype MPC;
- all rule-based baseline trajectories;
- BESS-noise robustness trajectories for `critical_50` and `full_100`;
- annual mesh summaries and the complete 27-candidate decision tables.

It omits the thousands of per-mesh trajectories that are not required to
reproduce the paper figures and statistical comparisons.

### Full dataset

The full tier contains the complete `operation-sweep/with-degradation` and
`baseline-sweep/with-degradation` trees, excluding only temporary files and
Python caches. It is intended for archival replication and secondary analysis.

## Refresh inventories

```powershell
.\.venv\Scripts\python.exe scripts\build_publication_release.py
```

This command also refreshes `paper/artifacts`. It does not build or copy a
large archive.

## Build an upload package

```powershell
.\.venv\Scripts\python.exe scripts\build_publication_release.py --package core
```

or, for the complete archive:

```powershell
.\.venv\Scripts\python.exe scripts\build_publication_release.py --package full
```

Packages are written to the ignored `release/zenodo/packages/` directory. Each
ZIP includes `checksums.sha256`. Parquet files are stored without redundant ZIP
compression; text metadata and CSV files are compressed.

Before uploading, add the final author list, license, related Git commit, and
Zenodo metadata. After publication, record the DOI in the repository README and
the manuscript's Data Availability statement.

