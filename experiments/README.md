# Reaction-Scope Decision-Space Validation

This folder is the main entry point for the independent validation benchmark.
The older baseline scripts under `Examples/` remain available for provenance,
but routine experiments should be launched through `run_benchmark.py` and a
JSON config.

## Runtime

Create and activate the repository environment before running experiments:

```powershell
conda env create -f environment.yml
conda activate reaction-scope-validation
$PY = (Get-Command python).Source
```

If the runner is launched from a different Python, pass the experiment runtime
explicitly:

```powershell
python experiments\run_benchmark.py --python $PY --config experiments\configs\aldol_smoke.json
```

The optional manuscript builders use `python-docx` and Pillow, both included in
the environment definition:

```powershell
& $PY experiments\build_core_manuscript_docx.py
```

## Quick Smoke Test

From the project root:

```powershell
& $PY experiments\run_benchmark.py --config experiments\configs\aldol_smoke.json
```

Outputs are written to:

```text
experiments/results/aldol_smoke
```

## Core Aldol Benchmark

```powershell
& $PY experiments\run_benchmark.py --config experiments\configs\aldol_core.json
```

This config runs:

- Dimension-aware model comparison.
- Repeated split confidence intervals.
- Boundary-sensitive sampling comparison.
- Plot generation for each task.

Outputs are written to:

```text
experiments/results/aldol_core
```

## Core Cobalt Benchmark

```powershell
& $PY experiments\run_benchmark.py --config experiments\configs\cobalt_core.json
```

This config runs the condition-1 cobalt benchmark using `condition1_yield >= 1`
as the reactive/non-reactive threshold.

Outputs are written to:

```text
experiments/results/cobalt_core
```

## Cobalt Threshold Sensitivity

```powershell
& $PY experiments\run_benchmark.py --config experiments\configs\cobalt_threshold_sensitivity.json
```

This config checks whether cobalt conclusions hold under stricter success
thresholds:

- `condition1_yield >= 1`
- `condition1_yield >= 30`
- `condition1_yield >= 50`

Outputs are written to:

```text
experiments/results/cobalt_threshold_sensitivity
```

## Run One Task

Use `--only` with the task name from the config:

```powershell
& $PY experiments\run_benchmark.py --config experiments\configs\aldol_core.json --only aldol_boundary_sampling
```

## Dry Run

Print commands without running them:

```powershell
& $PY experiments\run_benchmark.py --config experiments\configs\aldol_core.json --dry-run
```

## Config Notes

Each task has:

- `name`: stable task identifier.
- `type`: one of `dimension_model`, `repeated_ci`, `boundary_sampling`, or `reference_sampling`.
- `output_dir`: result folder.
- `plot`: whether to run the matching plot script.
- `params`: command-line arguments passed to the underlying script.

Relative paths are resolved from the project root.

## Article Reference Sampling Benchmarks

Some article-test datasets provide a large candidate substrate space plus an
experimental/reference subset, but not full-space negative labels or yields. For
those cases, use the reference-coverage benchmark instead of forcing a
classification target:

```powershell
& $PY experiments\run_benchmark.py --config experiments\configs\article_reference_sampling.json
```

This config currently covers Alcohols, Thiols, oxo-carboxide substrates, and
styrenes. It compares 2D, 64D, 128D, full descriptor space, and Tanimoto space
by asking how close each sampling strategy gets to the reference subset.

## External HTE Yield Benchmarks

Two public high-throughput reaction yield datasets can be prepared from the
`rxn4chemistry/rxn_yields` workbooks:

```powershell
& $PY experiments\prepare_rxn_yields_benchmarks.py
& $PY experiments\run_benchmark.py --config experiments\configs\external_yield_benchmarks.json
```

These external tasks use reaction-component one-hot features because the Suzuki
workbook does not provide complete molecular SMILES for every component.

## Retrospective Prospective Boundary Simulation

For labelled datasets, use the prospective simulation config to mimic a real
batch-by-batch experimental campaign without running new wet-lab experiments:

```powershell
& $PY experiments\run_benchmark.py --config experiments\configs\prospective_boundary_simulation.json
```

The full label table is treated as an oracle. Within each representation, each
method starts from the same initial labelled set, selects a new batch, receives
labels only for that batch, and repeats until the budget is spent. Boundary
regions and failure clusters are defined in that representation, so this config
is useful for exploratory budget trajectories. For a strict 2D-versus-decision-
space claim with a shared initial set and fixed full-space evaluation geometry,
use `decision_space_contrast.py` and the frozen Chemical Science sprint outputs.

Outputs are written to:

```text
experiments/results/prospective_boundary_simulation
```

Use `--only aldol_prospective_boundary` or `--only cobalt_prospective_boundary`
to run one case at a time.

## Cross-Metric Trade-Off Analysis

After the summary tables and prospective simulations exist, run:

```powershell
& $PY experiments\analyze_metric_tradeoffs.py
```

This analysis uses labelled datasets only. It intentionally excludes the
article-test reference-subset tasks because those tasks do not contain reliable
full-space success/failure labels. Outputs are written to:

```text
experiments/results/summary
```

Key outputs:

- `metric_tradeoff_correlations.csv`
- `metric_tradeoff_rank_conflicts.csv`
- `metric_tradeoff_pareto_frontier.csv`
- `budget_trajectory_summary.csv`
- `metric_tradeoff_correlation_heatmap.png`
- `metric_tradeoff_pareto_scatter.png`
- `budget_trajectory_boundary_coverage.png`
