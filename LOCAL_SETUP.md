# Local setup

## Conda environment

Create the project environment from the repository root:

```powershell
conda env create -f environment.yml
conda activate reaction-scope-validation
```

To update an existing environment after dependency changes:

```powershell
conda env update -n reaction-scope-validation -f environment.yml --prune
```

## IDE configuration

Open the repository root as the project folder and select the Python interpreter
from the `reaction-scope-validation` Conda environment. Avoid saving
machine-specific absolute interpreter paths in versioned configuration files.

## Benchmark commands

Preview a configured run:

```powershell
python experiments\run_benchmark.py `
  --config experiments\configs\aldol_smoke.json `
  --dry-run
```

Run the smoke test:

```powershell
python experiments\run_benchmark.py `
  --config experiments\configs\aldol_smoke.json
```

The inherited baseline interfaces remain available for reproducibility:

```powershell
python ScopeMap\sampling.py help
python SubstrateScore\evaluate.py --help
```

## UMAP and Numba cache

If the default Numba cache is not writable, create a cache folder inside the
repository and point `NUMBA_CACHE_DIR` to it for the current shell:

```powershell
New-Item -ItemType Directory -Force .numba_cache | Out-Null
$env:NUMBA_CACHE_DIR = (Resolve-Path .numba_cache).Path
python get_umap.py
```

The cache folder is ignored by Git.
