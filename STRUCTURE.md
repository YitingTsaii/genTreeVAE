# Project Structure Overview

## File Organization

```
genTreeVAE/
├── run.py                          # MAIN ENTRY POINT
│                                   # Usage: python3 run.py [--ntip] [--latent-dim] [--weight] [--epochs]
│
├── src/                            # SOURCE CODE PACKAGE
│   ├── __init__.py
│   ├── config.py                  # [CONFIG] Global settings & hyperparameters
│   │
│   ├── models/                    # [MODELS] Neural network architectures
│   │   ├── __init__.py
│   │   ├── vae.py                # VAE model with encoder/decoder
│   │   └── fmatrix.py            # FMatrixLayer for constraint enforcement
│   │
│   ├── data/                      # [DATA] Data loading & preprocessing
│   │   ├── __init__.py
│   │   └── loader.py             # get_loaders() - stratified splitting
│   │
│   ├── utils/                     # [UTILITIES] Helper functions
│   │   ├── __init__.py
│   │   ├── math_helpers.py       # F-matrix conversions & projections
│   │   ├── validation.py         # F-matrix validity checks
│   │   └── training.py           # train(), validate(), test() loops
│   │
│   └── evaluation/                # [EVALUATION] Model assessment
│       ├── __init__.py
│       └── metrics.py            # Evaluation functions & inference
│
├── slurm_scripts/                 # [SLURM JOBS] HPC job submission
│   ├── submit_sweep.sh           # Submit full hyperparameter sweep
│   └── submit_single.sh          # Submit single job
│
├── data/                          # [INPUT DATA] (symlinked to real data)
│   ├── BF_sim_betaSplit_ntip9_eachBeta1k/
│   ├── BF_sim_betaSplit_ntip17_eachBeta4k/
│   ├── BF_sim_betaSplit_ntip33_eachBeta10k/
│   └── BF_sim_betaSplit_ntip65_eachBeta40k/
│       ├── beta.csv
│       └── Fmat_below_sub.csv
│
├── results/                       # [OUTPUT DATA] Results & model checkpoints
│   └── run_ntip*_ldim*_w*.csv
│
├── README.md                      # Overview & quick start
├── USAGE.md                       # Detailed usage examples
└── VAE_MLP_hyper_sweep.py        # [LEGACY] Original monolithic script
```

## Module Dependencies

```
run.py
├── src.config
├── src.data.loader
│   └── src.config
├── src.models.vae
│   └── src.models.fmatrix
│       └── torch
├── src.utils.training
│   └── src.config
├── src.evaluation.metrics
    ├── src.utils.math_helpers
    ├── src.utils.validation
    └── torch
```

## Key Files Explained

### Entry Point
- **run.py**: Command-line interface with argparse
  - Accepts: `--ntip`, `--latent-dim`, `--weight`, `--epochs`
  - Loads data, creates model, trains, evaluates
  - Saves results to CSV

### Configuration
- **src/config.py**: Single source of truth
  - Global constants (SEED, BATCH_SIZE, etc.)
  - Architecture configs per ntip
  - Data/results paths
  - Hyperparameter sweeps

### Models
- **src/models/vae.py**: VAE architecture
  - Encoder: Linear layers with BatchNorm & ReLU
  - Sampling: Reparameterization trick
  - Decoder: Mirrors encoder
  - FMatrixLayer output processing

- **src/models/fmatrix.py**: Constraint enforcement
  - JIT-compiled constraint application
  - Correction loss computation
  - F-matrix dimension calculations

### Data
- **src/data/loader.py**: Data loading pipeline
  - CSV reading (Fmat_below_sub.csv, beta.csv)
  - Normalization by ntip
  - Stratified train/val/test splitting
  - DataLoader creation with batching

### Utilities
- **src/utils/math_helpers.py**: Vector ↔ Matrix conversions
  - `vector_to_lower_triangular()`: Flat vector → F-matrix
  - `project_to_F()`: Apply discrete constraints

- **src/utils/validation.py**: F-matrix checking
  - `is_Fmat()`: Perfect validity check
  - `check_inequalities_float()`: Soft constraint satisfaction
  - `calculate_validity_percentage()`: Batch validity

- **src/utils/training.py**: Training loops
  - `loss_function_Fmat()`: Combined loss
  - `train()`: One epoch training
  - `validate()`: Validation loop
  - `test()`: Test set evaluation

### Evaluation
- **src/evaluation/metrics.py**: Comprehensive metrics
  - `evaluate_mae()`: Float & integer reconstruction error
  - `evaluate_validity_rates()`: Strict & soft validity percentages
  - `count_active_units()`: Latent dimension analysis
  - `full_evaluation()`: Complete evaluation report
  - `get_reconstructions_projected()`: Inference pipeline

### SLURM Scripts
- **submit_sweep.sh**: Parametric job submission
  - Loops over ntips, latent_dims, weights
  - Creates individual SLURM scripts per job
  - Supports `--dry-run` for testing
  - Configurable partitions, time, memory

- **submit_single.sh**: Single job submission
  - CLI arguments for job parameters
  - Creates temporary SLURM script
  - Returns job ID on submission
  - Supports `--dry-run`

## Data Flow

```
Data Files (CSV)
    ↓
Data Loader (src/data/loader.py)
    ├─ Read & Normalize
    ├─ Stratified Split
    └─ Create DataLoaders
        ↓
    Batch (normalized vectors)
        ↓
    VAE Encoder (src/models/vae.py)
        ├─ Linear layers
        └─ μ, log(σ²) outputs
        ↓
    Reparameterize
        └─ Sample z ~ N(μ, σ)
        ↓
    VAE Decoder
        ├─ Linear layers
        ├─ Sigmoid (0-1 range)
        └─ FMatrixLayer (constraint)
            ├─ Vector → Matrix
            ├─ Apply constraints (JIT)
            └─ Matrix → Vector
        ↓
    Loss Computation (src/utils/training.py)
        ├─ Reconstruction (MSE)
        ├─ KL divergence
        └─ Structural constraint
        ↓
    Backpropagation & Optimization
        ↓
    Evaluation (src/evaluation/metrics.py)
        ├─ Denormalize
        ├─ Convert to matrix
        ├─ Project to discrete
        └─ Compute metrics
        ↓
    Results (CSV)
```

## Configuration Hierarchy

```
Global Defaults (src/config.py)
    ↓
Command-line Arguments (run.py)
    ├─ Override: ntip, latent_dim, weight, epochs
    ↓
Per-ntip Config (config_by_ntip)
    ├─ hidden_dims
    ├─ latent_dims
    └─ split ratios
    ↓
Model Instantiation (src/models/vae.py)
    └─ Uses config values
```

## Execution Paths

### Path 1: Local Development
```
run.py (main)
└── config.py (load config)
    └── data/loader.py (get_loaders)
        └── models/vae.py (create model)
            └── utils/training.py (train)
                └── evaluation/metrics.py (eval)
                    └── results/*.csv
```

### Path 2: SLURM Submission
```
submit_sweep.sh (loop & create jobs)
└── [sbatch] → submit_single.sh template
    └── run.py (same as Path 1)
```

## Code Reusability

Most modules can be imported independently:

```python
# Load data only
from src.data.loader import get_loaders
train_loader, val_loader, test_loader, input_dim = get_loaders(ntip=65)

# Create model only
from src.models import VAE
model = VAE(input_dim, hidden_dims, latent_dim, device)

# Evaluate only
from src.evaluation.metrics import evaluate_mae
mae_float, mae_int = evaluate_mae(model, val_loader, ntip, device)

# Validate matrices
from src.utils.validation import is_Fmat
is_valid = is_Fmat(matrix)
```

## Adding New Features

### Add new training loop variant:
1. Create function in `src/utils/training.py`
2. Import in `run.py`
3. Call with appropriate arguments

### Add new evaluation metric:
1. Add function to `src/evaluation/metrics.py`
2. Update `full_evaluation()`
3. Update result CSV columns

### Add new dataset size:
1. Add config to `config_by_ntip` in `src/config.py`
2. Update SLURM sweep script NTIPS list
3. Data must exist at `BASE_DATA_PATH`

## Testing

```bash
# Quick sanity check
python3 run.py --ntip 17 --latent-dim 2 --epochs 1

# Full test across ntips
for ntip in 9 17 33 65; do
    python3 run.py --ntip $ntip --latent-dim 2 --epochs 1
done

# SLURM dry-run
bash slurm_scripts/submit_sweep.sh --dry-run
```

## Performance Characteristics

- **Memory**: Scales with (hidden_dims × batch_size)
- **Compute**: Primarily in backprop + constraint enforcement
- **Disk I/O**: CSV reading at startup only
- **Output**: One CSV file per job (~1-2 KB)

Typical runtime: 5-10 minutes per job (5-10 epochs)
