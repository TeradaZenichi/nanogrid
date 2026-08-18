# Distributed operational sweeps

Each directory contains a solved sizing artifact, the two-stage 972-run operational sweep, reference consolidation, paired baselines, and publication artifacts. No sizing optimization is executed on the worker machine.

The economic sweep is published under `paper/operation/economic/mesh`. Run `critical_50/run_full_sweep.ps1` and `full_100/run_full_sweep.ps1` on separate repository copies. Return the corresponding directory under `outputs/sweeps/` without renaming it, then promote audited summaries to `paper/`.
