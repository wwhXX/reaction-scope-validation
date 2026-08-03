# SubstrateScore

A Python toolkit for evaluating substrate sampling quality, implementing multiple sampling algorithms and evaluation metrics.

## Features

### Evaluation Metrics
- **U-Score**: Entropy-based sampling uniformity score
- **R-Score**: Distance-based sampling representativeness score

### Molecular Descriptor Generation
- **Morgan Fingerprints**: Support for 4-layer, 1024-bit Morgan molecular fingerprints
- **MACCS Keys**: Molecular Access System keys

## File Description

### Core Modules
- `evaluate.py`: Main evaluation script with complete sampling quality assessment workflow
- `sampling.py`: Implementation of various sampling algorithms
- `score_utils.py`: Evaluation metric calculation functions
- `desc_gen_util.py`: Molecular descriptor generation utility

### Data Files
- `test_exp.csv`: Experimental data file example
- `test_space.csv`: Complete substrate space data file example

## Installation

```bash
pip install numpy pandas scikit-learn scipy rdkit
```

## Usage

### Basic Evaluation Example

```bash
python evaluate.py \
    --substrate-file test_space.csv \
    --experimental-file test_exp.csv \
    --distance-metric euclidean \
    --k-neighbors 5 \
    --random-seed 42 \
    --random-runs 5
```

### Generate Molecular Descriptors

```bash
python desc_gen_util.py test_space.csv
```

### Parameter Description

- `--substrate-file`: Path to substrate data file
- `--substrate-fp-file`: Path to substrate fingerprint file (optional, auto-generated)
- `--experimental-file`: Path to experimental data file
- `--experimental-fp-file`: Path to experimental fingerprint file (optional, auto-generated)
- `--distance-metric`: Distance metric (euclidean, manhattan, cosine, tanimoto)
- `--k-neighbors`: Number of neighbors for entropy calculation (default: 5)
- `--random-seed`: Random seed (default: 42)
- `--random-runs`: Number of random sampling runs (default: 5)

## Evaluation Workflow

1. **Data Preprocessing**: Read substrate and experimental data, auto-generate molecular fingerprints
2. **Sampling Algorithm Evaluation**:
   - CVT sampling
   - Kennard-Stone sampling
   - Random sampling (multiple runs)
3. **Experimental Data Evaluation**: Assess sampling quality of provided experimental data
4. **Score Calculation**:
   - U-Score: Entropy-based evaluation score
   - R-Score: Distance-based evaluation score

## Output Results

The program outputs detailed evaluation reports including:
- Entropy and MSD values for each sampling method
- Experimental sampling quality assessment
- U-Score and R-Score comprehensive ratings
- Complete result comparison table

## Algorithm Principles

### U-Score and R-Score
- U-Score: Compares experimental sampling entropy with random and K-S sampling
- R-Score: Compares experimental sampling MSD with random and CVT sampling

## License

MIT License