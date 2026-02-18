# Quick Reference Card

## Most Common Commands

### Local Training
```bash
python3 run.py --ntip 65 --latent-dim 32 --weight 10.0 --epochs 10
```

### SLURM Single Job
```bash
bash slurm_scripts/submit_single.sh --ntip 65 --latent-dim 32 --weight 10.0
```

### SLURM Full Sweep
```bash
bash slurm_scripts/submit_sweep.sh
```

### Dry Run (Preview)
```bash
bash slurm_scripts/submit_sweep.sh --dry-run
bash slurm_scripts/submit_single.sh --ntip 65 --dry-run
```

---

## Parameter Ranges

### ntip (Number of Tips)
- Valid: `9`, `17`, `33`, `65`
- Default: `65`

### latent-dim (Latent Dimension)
- Recommended per ntip:
  - ntip=9: `2`, `8`, `16`
  - ntip=17: `2`, `16`, `32`
  - ntip=33: `2`, `32`, `64`
  - ntip=65: `2`, `64`, `128`

### weight (Constraint Weight)
- Sweep range: `0.01`, `0.1`, `1.0`, `10.0`, `100.0`
- Default: `10.0`
- Lower = More reconstruction focus
- Higher = More constraint focus

### epochs (Training Epochs)
- Default: `5`
- Recommended: `10-20` for good results
- Can go higher for convergence

---

## Output Files

Results saved to:
```
results/run_ntip{ntip}_ldim{latent_dim}_w{weight}.csv
```

Example:
```
results/run_ntip65_ldim32_w10.0.csv
```

### Result Columns:
- `MAE_Float`: Raw reconstruction error
- `MAE_Int`: Integer reconstruction error
- `Strict_Validity`: Perfect F-matrix %
- `Soft_Valid_Float`: Constraint satisfaction %
- `Active_Dims`: Number of active latent dimensions

---

## Directory Structure

```
genTreeVAE/
├── run.py                 ← Main entry point
├── src/
│   ├── config.py         ← Edit paths & hyperparams here
│   ├── models/           ← VAE architecture
│   ├── data/             ← Data loading
│   ├── utils/            ← Training & helpers
│   └── evaluation/       ← Metrics & inference
├── slurm_scripts/        ← HPC job submission
├── data/                 ← Input data (CSV files)
└── results/              ← Output CSV results
```

---

## Configuration Files

### Main Config
**File**: `src/config.py`
```python
SEED = 428
BATCH_SIZE = 128
EPOCHS = 5                    # ← Change default epochs
CONSTRAINT_WEIGHT = 10        # ← Change default weight
BASE_DATA_PATH = "..."       # ← Change data path
RESULTS_PATH = "..."         # ← Change results path
```

### SLURM Config
**File**: `slurm_scripts/submit_sweep.sh` or `submit_single.sh`
```bash
PARTITION="gpu"              # ← Change GPU partition
TIME_LIMIT="04:00:00"       # ← Change time limit
MEM_PER_TASK="32G"          # ← Change memory
```

---

## Monitoring

### View all your jobs:
```bash
squeue -u $(whoami)
```

### View job details:
```bash
squeue -j <JOB_ID>
```

### Watch job progress:
```bash
tail -f slurm_logs/vae_hypersweep_<JOB_ID>.log
```

### Check result files:
```bash
ls -lht results/ | head
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Data not found | Check `BASE_DATA_PATH` in `src/config.py` |
| Out of memory | Reduce `BATCH_SIZE` in `src/config.py` |
| CUDA error | Use `--device cpu` (add to run.py if needed) |
| Import error | Ensure `cd` to project root |
| Jobs not submitting | Check SLURM `PARTITION` setting |

---

## Workflow Examples

### Explore latent dimensions:
```bash
for ldim in 2 16 32 64 128; do
    python3 run.py --ntip 65 --latent-dim $ldim --epochs 5
done
```

### Test different constraints:
```bash
for w in 0.01 0.1 1.0 10.0 100.0; do
    python3 run.py --ntip 65 --weight $w --epochs 5
done
```

### Full validation on HPC:
```bash
# Check dry run first
bash slurm_scripts/submit_sweep.sh --dry-run

# Submit full sweep
bash slurm_scripts/submit_sweep.sh

# Monitor
watch 'squeue -u $(whoami) | grep vae'

# Collect results
cat results/*.csv > all_results.csv
```

---

## Key Files to Know

| File | Purpose |
|------|---------|
| `run.py` | Main entry point - always run this |
| `src/config.py` | Global settings - edit for customization |
| `src/models/vae.py` | Model architecture - modify for new designs |
| `slurm_scripts/*.sh` | Job submission - edit for HPC setup |
| `data/` | Input directory - ensure data exists here |
| `results/` | Output directory - check results here |

---

## Common Patterns

### Pattern 1: Quick Test
```bash
python3 run.py --ntip 17 --latent-dim 2 --epochs 1
```

### Pattern 2: Single Good Run
```bash
python3 run.py --ntip 65 --latent-dim 32 --weight 10.0 --epochs 20
```

### Pattern 3: Parameter Search
```bash
bash slurm_scripts/submit_sweep.sh
```

### Pattern 4: Specific Subset
```bash
for w in 0.1 1.0 10.0; do
    bash slurm_scripts/submit_single.sh --ntip 65 --weight $w
done
```

---

## Environment Setup

### Requirements:
```bash
pip install torch pandas numpy
```

### Optional (for analysis):
```bash
pip install matplotlib scipy scikit-learn
```

### Verify setup:
```bash
python3 -c "import torch; print(torch.__version__)"
python3 -c "import pandas; print(pandas.__version__)"
```

---

## Getting Help

### View script usage:
```bash
python3 run.py --help
```

### Check logs for errors:
```bash
tail -50 slurm_logs/vae_hypersweep_*.log
```

### View README:
```bash
cat README.md
```

### Check structure:
```bash
cat STRUCTURE.md
```

### See usage examples:
```bash
cat USAGE.md
```

---

## Pro Tips

1. **Start small**: Test with ntip=17 before full sweep
2. **Use dry-run**: Always check jobs before submitting
3. **Monitor GPU**: Use `nvidia-smi` during training
4. **Save commands**: Keep copy of successful commands
5. **Document changes**: Note any config modifications

---

## Default Behavior

Running without arguments:
```bash
python3 run.py
```
Uses:
- ntip = 65
- latent_dim = 32
- weight = 10.0
- epochs = 5

---

*Last updated: 2026-02-17*
