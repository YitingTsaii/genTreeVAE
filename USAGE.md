# Usage Guide for genTreeVAE

## Quick Reference

### Run a single training job locally:
```bash
python3 run.py --ntip 65 --latent-dim 32 --weight 10.0 --epochs 5
```

### Submit a single SLURM job:
```bash
bash slurm_scripts/submit_single.sh --ntip 65 --latent-dim 32 --weight 10.0 --epochs 10
```

### Submit the full hyperparameter sweep:
```bash
bash slurm_scripts/submit_sweep.sh
```

---

## Detailed Usage

### 1. Local Development

#### Run with default parameters:
```bash
python3 run.py
# Uses: ntip=65, latent-dim=32, weight=10.0, epochs=5
```

#### Run specific configuration:
```bash
python3 run.py --ntip 17 --latent-dim 16 --weight 1.0 --epochs 10
```

#### Run with longer training:
```bash
python3 run.py --ntip 33 --latent-dim 64 --weight 100.0 --epochs 20
```

### 2. SLURM Job Submission

#### Single Job with Default Hyperparameters:
```bash
bash slurm_scripts/submit_single.sh
# Submits: ntip=65, latent-dim=32, weight=10.0, epochs=10
```

#### Single Job with Custom Parameters:
```bash
bash slurm_scripts/submit_single.sh \
    --ntip 17 \
    --latent-dim 16 \
    --weight 0.1 \
    --epochs 15
```

#### Preview SLURM Job (Dry Run):
```bash
bash slurm_scripts/submit_single.sh \
    --ntip 65 \
    --latent-dim 32 \
    --weight 10.0 \
    --dry-run
```

### 3. Hyperparameter Sweep

The sweep submits jobs for all combinations:

**ntips**: 9, 17, 33, 65
**latent_dims** (per ntip):
- ntip=9: 2, 8, 16
- ntip=17: 2, 16, 32
- ntip=33: 2, 32, 64
- ntip=65: 2, 64, 128

**weights**: 0.01, 0.1, 1.0, 10.0, 100.0

#### Preview all jobs (Dry Run):
```bash
bash slurm_scripts/submit_sweep.sh --dry-run
```

#### Submit all jobs:
```bash
bash slurm_scripts/submit_sweep.sh
```

This will create approximately **60 SLURM jobs** total (4 ntips × varying latent dims × 5 weights).

### 4. Monitoring SLURM Jobs

#### View all your jobs:
```bash
squeue -u $(whoami)
```

#### View specific job details:
```bash
squeue -j <JOB_ID>
```

#### Monitor job progress:
```bash
tail -f slurm_logs/vae_hypersweep_<JOB_ID>.log
```

#### View completed jobs:
```bash
sacct -u $(whoami) --format=JobID,JobName,State,Elapsed
```

---

## Configuration

### Edit SLURM Settings
File: `slurm_scripts/submit_sweep.sh` or `slurm_scripts/submit_single.sh`

```bash
PARTITION="gpu"              # Change to your GPU partition
TIME_LIMIT="04:00:00"       # Adjust for your jobs
MEM_PER_TASK="32G"          # Memory per task
CPUS_PER_TASK=4             # CPUs per task
GPUS_PER_TASK=1             # GPUs per task
```

### Edit Training Hyperparameters
File: `src/config.py`

```python
SEED = 428                  # Random seed
BATCH_SIZE = 128            # Batch size
LOG_INTERVAL = 10           # Logging frequency
EPOCHS = 5                  # Default epochs
CONSTRAINT_WEIGHT = 10      # Default constraint weight

# Add weights to sweep:
WEIGHTS_TO_TRY = [0.01, 0.1, 1.0, 10.0, 100.0]
```

### Edit Data Paths
File: `src/config.py`

```python
BASE_DATA_PATH = "/content/drive/MyDrive/genTreeVAE/beta_splitting_data/"
RESULTS_PATH = "/content/drive/MyDrive/genTreeVAE/results/"
```

---

## Output Files

Results are saved as CSV files in `RESULTS_PATH`:

### File naming:
```
run_ntip{ntip}_ldim{latent_dim}_w{weight}.csv
```

### Example:
```
run_ntip65_ldim32_w10.0.csv
```

### Columns in results:
- `latent_dim`: Latent dimension
- `weight`: Constraint weight
- `MAE_Float`: Float reconstruction MAE
- `MAE_Int`: Integer reconstruction MAE
- `Strict_Validity`: % of perfectly valid F-matrices
- `Soft_Valid_Float`: % constraints satisfied (float)
- `Soft_Valid_Int`: % constraints satisfied (int)
- `Active_Dims`: Number of active latent dimensions
- `Avg_KL`: Average KL divergence

---

## Common Workflows

### Workflow 1: Explore Latent Dimensions
```bash
# Test different latent dimensions for ntip=65
for ldim in 2 16 32 64 128; do
    python3 run.py --ntip 65 --latent-dim $ldim --weight 10.0 --epochs 5
done
```

### Workflow 2: Explore Constraint Weights
```bash
# Test different constraint weights
for weight in 0.01 0.1 1.0 10.0 100.0; do
    python3 run.py --ntip 65 --latent-dim 32 --weight $weight --epochs 5
done
```

### Workflow 3: Test All ntips
```bash
# Quick test across all dataset sizes
for ntip in 9 17 33 65; do
    python3 run.py --ntip $ntip --latent-dim 32 --weight 10.0 --epochs 3
done
```

### Workflow 4: Full Sweep on HPC
```bash
# Submit full parameter sweep
bash slurm_scripts/submit_sweep.sh

# Monitor progress
watch -n 30 'squeue -u $(whoami) | grep vae_hypersweep'

# Check results as they complete
ls -lht results/ | head -20
```

---

## Troubleshooting

### Job fails with "Data not found"
- Check `BASE_DATA_PATH` in `src/config.py`
- Verify data files exist: `Fmat_below_sub.csv` and `beta.csv`

### CUDA out of memory
- Reduce `BATCH_SIZE` in `src/config.py`
- Reduce `MEM_PER_TASK` in SLURM script
- Use smaller `latent_dim` or `hidden_dims`

### Jobs stuck in queue
- Check partition availability: `sinfo`
- Increase `TIME_LIMIT` in SLURM scripts
- Verify GPU availability: `sinfo -N`

### Import errors
- Ensure you're in the project root directory
- Check Python version (3.8+)
- Verify dependencies: `pip install torch pandas numpy`

---

## Performance Tips

1. **Start small**: Test with ntip=17, latent_dim=16 before full sweep
2. **Use gradient checkpointing**: Can reduce memory for large models
3. **Batch processing**: Submit multiple jobs to utilize HPC resources
4. **Monitor GPU**: Use `nvidia-smi` in running jobs
5. **Save checkpoints**: Implement in training loop for long jobs

---

## Advanced: Adding Custom Configurations

### Add new ntip (e.g., ntip=100):
1. Add to `src/config.py`:
   ```python
   100: {
       "samples_per_scenario": 100000,
       "hidden_dims": [2048, 1024, 512],
       "latent_dims": [2, 128, 256],
       "split": [0.95, 0.025, 0.025],
   }
   ```

2. Add to SLURM sweep script:
   ```bash
   NTIPS=(9 17 33 65 100)
   ```

### Modify evaluation metrics:
Edit `src/evaluation/metrics.py` and add functions called in `full_evaluation()`.

### Change training loss:
Edit loss function in `src/utils/training.py`.

---

## Getting Help

### View script help:
```bash
python3 run.py --help
```

### Check job logs:
```bash
tail -100 slurm_logs/vae_hypersweep_*.log
```

### View module imports:
```python
# In Python
import sys
sys.path.insert(0, '.')
from src.models import VAE
from src.data.loader import get_loaders
```
