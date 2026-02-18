# genTreeVAE - Refactored Project Structure

## Overview

This is a refactored version of the phylogenetic VAE project with a clean, modular directory structure. The code has been split into logical modules and now includes a command-line entry point for easy training and evaluation.

## Directory Structure

```
genTreeVAE/
├── run.py                          # Main entry point (CLI interface)
├── src/                            # Source code package
│   ├── __init__.py
│   ├── config.py                  # Global configuration and hyperparameters
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vae.py                # VAE model class
│   │   └── fmatrix.py            # FMatrixLayer and constraints
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py             # Data loading utilities
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── math_helpers.py       # F-matrix utility functions
│   │   ├── validation.py         # F-matrix validation functions
│   │   └── training.py           # Training/validation/test loops
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py            # Evaluation and inference functions
├── slurm_scripts/
│   ├── submit_sweep.sh          # Hyperparameter sweep submission
│   └── submit_single.sh         # Single job submission
├── data/                         # Input data directory (symlinked to your data)
│   └── BF_sim_betaSplit_*/
│       ├── beta.csv
│       └── Fmat_below_sub.csv
└── README.md                     # This file
```

## Quick Start

### 1. Configuration

Edit `src/config.py` to adjust:
- `BASE_DATA_PATH`: Path to your data directory
- `RESULTS_PATH`: Where to save results
- `EPOCHS`, `BATCH_SIZE`, `CONSTRAINT_WEIGHT`: Training hyperparameters
- `config_by_ntip`: Architecture for different dataset sizes

### 2. Running a Single Training Job

#### Command-line interface:
```bash
python3 run.py --ntip 65 --latent-dim 32 --weight 10.0 --epochs 5
```

#### Available arguments:
```
--ntip {9, 17, 33, 65}        Number of tips (taxa) - default: 65
--latent-dim <int>             Latent dimension - default: 32
--weight <float>               Constraint weight - default: 10.0
--epochs <int>                 Number of training epochs - default: 5
```

### 3. Running Hyperparameter Sweep via SLURM

#### Dry run (see what would be submitted):
```bash
bash slurm_scripts/submit_sweep.sh --dry-run
```

#### Submit the full sweep:
```bash
bash slurm_scripts/submit_sweep.sh
```

This will submit jobs for all combinations of:
- ntips: 9, 17, 33, 65
- latent_dims: varies by ntip (configured in `src/config.py`)
- weights: 0.01, 0.1, 1.0, 10.0, 100.0

#### Submit a single job via SLURM:
```bash
bash slurm_scripts/submit_single.sh --ntip 65 --latent-dim 32 --weight 10.0 --epochs 10
```

## Module Descriptions

### `src/config.py`
- Global constants and hyperparameters
- Configuration dictionaries by ntip
- Paths for data and results

### `src/models/`
- **vae.py**: VAE class with encoder/decoder architecture
- **fmatrix.py**: FMatrixLayer with constraint enforcement

### `src/data/`
- **loader.py**: Data loading with stratified splitting

### `src/utils/`
- **math_helpers.py**: Vector/matrix conversions, projection operations
- **validation.py**: F-matrix validation checks
- **training.py**: Training, validation, and test loops

### `src/evaluation/`
- **metrics.py**: Comprehensive evaluation metrics (MAE, validity, latent analysis)

## SLURM Configuration

Edit the following variables in SLURM scripts:

```bash
PARTITION="gpu"              # Your GPU partition name
TIME_LIMIT="04:00:00"       # Job time limit
MEM_PER_TASK="32G"          # Memory per task
CPUS_PER_TASK=4             # CPUs per task
GPUS_PER_TASK=1             # GPUs per task
```

## Results

Results are saved to `RESULTS_PATH` (configured in `src/config.py`) as CSV files:
- Format: `run_ntip{ntip}_ldim{latent_dim}_w{weight}.csv`
- Contains metrics: MAE (float/int), validity rates, active dims, KL divergence

## Key Changes from Original

1. **Modular Structure**: Code split into logical packages
2. **CLI Interface**: `run.py` accepts command-line arguments
3. **Configurable**: All hyperparameters in `src/config.py`
4. **Reproducible**: No hardcoded values in scripts
5. **SLURM Integration**: Two submission scripts for sweep and single jobs
6. **Better Imports**: Clean package structure with proper `__init__.py`

## Development Tips

### Adding a new ntip:
1. Add entry to `config_by_ntip` in `src/config.py`
2. Ensure data is in correct directory structure
3. Run: `python3 run.py --ntip <new_ntip> --latent-dim <dim>`

### Modifying training:
- Edit training loops in `src/utils/training.py`
- Loss function in same file

### Adding evaluation metrics:
- Add functions to `src/evaluation/metrics.py`
- Update `full_evaluation()` to call new metrics

## Dependencies

- PyTorch
- pandas
- numpy
- matplotlib (for original code)

Install with:
```bash
pip install torch pandas numpy matplotlib
```

## Notes

- Uses PyTorch JIT compilation for constraint enforcement (`@torch.jit.script`)
- Supports GPU acceleration via CUDA/Metal
- Data must be in CSV format (Fmat_below_sub.csv and beta.csv)
- All normalization/denormalization handled automatically

## License & Attribution

Based on original VAE implementation from treeDL_project.
Refactored for modularity and reproducibility.
