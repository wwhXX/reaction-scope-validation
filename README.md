# Decision-Space Validation for Reaction-Scope Mapping

This repository provides a paired retrospective benchmark for testing whether
reaction-scope acquisition remains valid when the representation used for
experiment selection changes. The central principle is simple:

> Visualization space is not validation space.

A two-dimensional chemical map can be useful for inspection and communication
without being a validated geometry for selecting the next substrates. The
benchmark therefore holds the labelled landscape, initial set, acquisition
budget, sampler, and final evaluation geometry fixed while varying only the
decision space used during acquisition.

## Research question

The project asks whether low-dimensional acquisition and chemically richer
descriptor or fingerprint acquisition recover the same reaction-boundary
regions under the same experimental budget.

The reported endpoints are kept separate:

- near-boundary coverage;
- failure recall;
- failure-cluster coverage;
- boundary enrichment;
- reference-region recovery when full-space outcome labels are unavailable.

The evidence does not support a universal ranking of representations or
samplers. Instead, it shows that the choice of decision space can change which
compatibility boundaries and failure families are observed.

## Evidence design

The main paired case studies use published Aldol and Cobalt reaction landscapes.
The strongest frozen final-budget contrasts increased boundary coverage over
paired 2D acquisition by 0.1293 for Aldol and 0.4867 for Cobalt. Sensitivity
analyses and chemistry-facing inspection are included to test whether these
gains correspond to reproducible and interpretable regions rather than visual
scatter.

A drawable Buchwald-Hartwig dataset is retained as an external workflow control.
It is supporting evidence, not the headline result, because richer spaces do not
win every endpoint in that dataset.

## Relationship to ScopeMap

This is an independent validation and benchmarking project, not an official
ScopeMap release, replacement, or sequel. Published ScopeMap landscapes are used
as chemically interpretable case studies, and the original sampling toolkit is
retained as a reproducibility baseline.

The inherited `ScopeMap/`, `SubstrateScore/`, and legacy `Examples/` components
remain subject to the included MIT license and original attribution. The new
decision-space contrasts, boundary-aware evaluation, robustness analyses,
external controls, and manuscript assembly workflow are maintained under
`experiments/`. See [NOTICE.md](NOTICE.md) for the provenance boundary.

## Repository layout

```text
Data/                 Input and prepared benchmark datasets
ScopeMap/             Inherited sampling baseline
SubstrateScore/       Inherited evaluation utilities
Examples/             Legacy examples and adapted simulation entry points
experiments/          Independent benchmark runners, configs, analyses, and builders
experiments/configs/  Reproducible experiment definitions
experiments/results/  Selected compact result summaries
```

## Installation

Create the Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate reaction-scope-validation
```

The environment includes NumPy, pandas, SciPy, scikit-learn, RDKit,
Matplotlib, UMAP, Optuna, and XGBoost.

## Quick start

Inspect a benchmark without launching it:

```bash
python experiments/run_benchmark.py \
  --config experiments/configs/aldol_smoke.json \
  --dry-run
```

Run the Aldol smoke test:

```bash
python experiments/run_benchmark.py \
  --config experiments/configs/aldol_smoke.json
```

Run the labelled retrospective campaign configuration:

```bash
python experiments/run_benchmark.py \
  --config experiments/configs/prospective_boundary_simulation.json
```

Detailed task descriptions and output locations are documented in
[experiments/README.md](experiments/README.md).

## Reproducibility notes

- Relative paths are resolved from the repository root.
- Generated result directories, manuscripts, and large binary artifacts are
  excluded by default; selected compact summaries are versioned explicitly.
- The main manuscript claim rests on paired repeats with shared initial sets,
  budgets, samplers, and fixed full-space evaluation geometry.
- Article-reference datasets without reliable full-space negative labels are
  evaluated for reference-region recovery, not reaction-failure boundaries.

## Data and attribution

The Aldol and Cobalt landscapes originate from the cited ScopeMap study. External
reaction-yield controls are prepared from their respective public sources. Users
are responsible for following the terms attached to each upstream dataset.

ScopeMap article DOI: [10.1002/anie.2455429](https://doi.org/10.1002/anie.2455429)

## License

The repository retains the MIT license distributed with the inherited code.
See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Dataset rights remain with
their original providers.

## Contact

Please use GitHub Issues for questions about the benchmark implementation or
reproducibility.
