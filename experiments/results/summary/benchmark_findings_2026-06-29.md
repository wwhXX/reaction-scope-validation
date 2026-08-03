# Benchmark Findings, 2026-06-29

> Historical exploratory note. This snapshot predates the current independent
> decision-space validation framing and is retained only for result provenance.
> The identifier `scopemap_hd` below is a frozen method key, not the project name.

## Scope

This update expanded the early benchmark in two directions:

- Article-test reference coverage: Alcohols, Thiols, oxo-carboxide substrates, and styrenes.
- External public HTE yield benchmarks: Buchwald-Hartwig and Suzuki-Miyaura datasets from `rxn4chemistry/rxn_yields`.

The reference-coverage tasks do not treat untested molecules as negative labels.
They evaluate how well unsupervised sampling methods cover known experimental or
reference subsets under different representation spaces.

## Article-Test Reference Coverage

Config:

```powershell
python experiments\run_benchmark.py --config experiments\configs\article_reference_sampling.json
```

Best q10 reference-coverage results:

| Dataset | Best method | Dimension | q10 reference coverage | q10 enrichment |
| --- | --- | --- | ---: | ---: |
| Styrene | ward | 128D | 0.786 | 7.86x |
| Thiol | kennard_stone | full | 0.556 | 5.56x |
| Alcohol | scopemap_hd | full | 0.447 | 4.47x |
| Oxo-carboxide | ward | full | 0.333 | 3.33x |

The legacy `scopemap_hd` method key denotes a high-dimensional iterative
CVT variant that uses already selected samples as repulsion anchors. Its strongest
new result is on the Alcohols reference-coverage task, where full-dimensional
that baseline is the best q10 coverage method.

## External HTE Yield Benchmarks

Preparation:

```powershell
python experiments\prepare_rxn_yields_benchmarks.py
```

Config:

```powershell
python experiments\run_benchmark.py --config experiments\configs\external_yield_benchmarks.json
```

The Suzuki workbook does not provide complete component SMILES, so both external
benchmarks use reaction-component one-hot features rather than invented molecular
descriptors.

RandomForest repeated-CI regression:

| Dataset | 2D R2 | 16D R2 | 32D R2 | Full R2 |
| --- | ---: | ---: | ---: | ---: |
| Buchwald-Hartwig | -0.019 | 0.741 | 0.815 | 0.870 |
| Suzuki-Miyaura | 0.071 | 0.661 | 0.782 | 0.838 |

RandomForest repeated-CI classification, yield >= 50:

| Dataset | 2D AUC | 16D AUC | 32D AUC | Full AUC |
| --- | ---: | ---: | ---: | ---: |
| Buchwald-Hartwig | 0.603 | 0.949 | 0.963 | 0.970 |
| Suzuki-Miyaura | 0.667 | 0.919 | 0.951 | 0.960 |

## Interpretation

The added datasets strengthen the central benchmark claim:

- 2D representations are useful for visualization, but weak as decision or prediction spaces.
- Medium/high-dimensional representations are consistently stronger across ScopeMap-distributed and external public HTE datasets.
- Reference coverage, nearest-reference distance, yield regression, and success/failure classification are distinct objectives and should not be collapsed into a single F1 score.
- The new `scopemap_hd` baseline is promising but not universally dominant; it should be presented as a constructive method extension, not a finished replacement for all sampling methods.

## Source Notes

External workbooks were downloaded from the public `rxn4chemistry/rxn_yields`
GitHub repository:

- `data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx`
- `data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx`
