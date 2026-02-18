#!/bin/bash
# ==========================================
# SLURM Hyperparameter Sweep Submission Script
# ==========================================
# This script submits a comprehensive hyperparameter sweep
# across different ntips, latent dimensions, and constraint weights.
# 
# Usage: ./submit_sweep.sh [--dry-run]

# Configuration
NTIPS=(9 17 33 65)
WEIGHTS=(0.01 0.1 1.0 10.0 100.0)
EPOCHS=10

# SLURM settings
PARTITION="gpu"  # Change to your available partition
TIME_LIMIT="04:00:00"
MEM_PER_TASK="32G"
CPUS_PER_TASK=4
GPUS_PER_TASK=1
JOB_NAME="vae_hypersweep"
LOG_DIR="./slurm_logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Check for dry-run flag
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "DRY RUN MODE - Jobs will not be submitted"
    echo "=========================================="
fi

# Counter for submitted jobs
TOTAL_JOBS=0
SUBMITTED_JOBS=0

echo "Starting Hyperparameter Sweep Submission"
echo "=========================================="

# Loop over ntips
for ntip in "${NTIPS[@]}"; do
    echo ""
    echo "Processing ntip=$ntip..."
    
    # Get latent dimensions for this ntip from config
    case $ntip in
        9)
            LATENT_DIMS=(2 8 16)
            ;;
        17)
            LATENT_DIMS=(2 16 32)
            ;;
        33)
            LATENT_DIMS=(2 32 64)
            ;;
        65)
            LATENT_DIMS=(2 64 128)
            ;;
    esac
    
    # Loop over latent dimensions
    for latent_dim in "${LATENT_DIMS[@]}"; do
        # Loop over weights
        for weight in "${WEIGHTS[@]}"; do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
            
            # Create SLURM job script
            JOB_SCRIPT="$(mktemp /tmp/slurm_job_XXXXXX.sh)"
            
            cat > "$JOB_SCRIPT" << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=SLURM_JOBNAME
#SBATCH --partition=SLURM_PARTITION
#SBATCH --time=SLURM_TIME
#SBATCH --mem-per-task=SLURM_MEMORY
#SBATCH --cpus-per-task=SLURM_CPUS
#SBATCH --gpus-per-task=SLURM_GPUS
#SBATCH --output=SLURM_LOGDIR/%x_%j.log
#SBATCH --error=SLURM_LOGDIR/%x_%j.err

# Load modules if needed (customize for your system)
# module load pytorch/2.0
# module load cuda/11.8

# Change to project directory
cd /Users/yitingtsai/Documents/Julia_Palacios/treeDL_project/genTreeVAE || exit 1

# Run the training script
python3 run.py \
    --ntip SLURM_NTIP \
    --latent-dim SLURM_LATENT_DIM \
    --weight SLURM_WEIGHT \
    --epochs SLURM_EPOCHS

echo "Job completed: ntip=SLURM_NTIP, latent_dim=SLURM_LATENT_DIM, weight=SLURM_WEIGHT"
SLURM_EOF

            # Replace placeholders
            sed -i "s|SLURM_JOBNAME|${JOB_NAME}_ntip${ntip}_ldim${latent_dim}_w${weight}|g" "$JOB_SCRIPT"
            sed -i "s|SLURM_PARTITION|${PARTITION}|g" "$JOB_SCRIPT"
            sed -i "s|SLURM_TIME|${TIME_LIMIT}|g" "$JOB_SCRIPT"
            sed -i "s|SLURM_MEMORY|${MEM_PER_TASK}|g" "$JOB_SCRIPT"
            sed -i "s|SLURM_CPUS|${CPUS_PER_TASK}|g" "$JOB_SCRIPT"
            sed -i "s|SLURM_GPUS|${GPUS_PER_TASK}|g" "$JOB_SCRIPT"
            sed -i "s|SLURM_LOGDIR|${LOG_DIR}|g" "$JOB_SCRIPT"
            sed -i "s|SLURM_NTIP|${ntip}|g" "$JOB_SCRIPT"
            sed -i "s|SLURM_LATENT_DIM|${latent_dim}|g" "$JOB_SCRIPT"
            sed -i "s|SLURM_WEIGHT|${weight}|g" "$JOB_SCRIPT"
            sed -i "s|SLURM_EPOCHS|${EPOCHS}|g" "$JOB_SCRIPT"

            # Submit or display job
            if [[ "$DRY_RUN" == true ]]; then
                echo "[DRY RUN] Would submit: ntip=$ntip, latent_dim=$latent_dim, weight=$weight"
                cat "$JOB_SCRIPT"
                echo "---"
            else
                echo "Submitting: ntip=$ntip, latent_dim=$latent_dim, weight=$weight"
                sbatch "$JOB_SCRIPT"
                SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
            fi
            
            # Clean up temp file
            rm -f "$JOB_SCRIPT"
        done
    done
done

echo ""
echo "=========================================="
echo "Summary:"
echo "  Total jobs: $TOTAL_JOBS"
if [[ "$DRY_RUN" == false ]]; then
    echo "  Submitted jobs: $SUBMITTED_JOBS"
else
    echo "  (DRY RUN - No jobs submitted)"
fi
echo "  Log directory: $LOG_DIR"
echo "=========================================="
