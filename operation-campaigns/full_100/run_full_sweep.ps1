$ErrorActionPreference = 'Stop'
python experiments/12_corrected_pipeline.py --workers 4 --n-iters 2880 --parameters data/parameters.json --sizing-artifact operation-campaigns/full_100/sizing_artifact.json --campaign-id full_100 --out-root outputs/sweeps/full_100 --stage mesh
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python experiments/12_corrected_pipeline.py --workers 4 --n-iters 2880 --parameters data/parameters.json --sizing-artifact operation-campaigns/full_100/sizing_artifact.json --campaign-id full_100 --out-root outputs/sweeps/full_100 --stage annual-mesh
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python experiments/14_consolidate_reference_results.py --campaign-root outputs/sweeps/full_100 --campaign-id full_100
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python experiments/15_baseline_comparison.py --campaign-root outputs/sweeps/full_100 --campaign-id full_100 --workers 4
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python experiments/13_generate_publication_artifacts.py --campaign-root outputs/sweeps/full_100 --campaign-id full_100
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
