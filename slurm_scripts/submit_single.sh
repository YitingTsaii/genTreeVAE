#!/bin/bash
# ==========================================
# SLURM Single Job Submission Script
# ==========================================
# This script submits a single training job.
#
# Usage: ./submit_single.sh --ntip <ntip> --latent-dim <dim> --weight <weight> [--epochs <epochs>]

# Default values
NTIP=65
LATENT_DIM=32
WEIGHT=10.0
EPOCHS=10

# SLURM settings
PARTITION="gpu"  # Change to your available partition
TIME_LIMIT="04:00:00"
MEM_PER_TASK="32G"
CPUS_PER_TASK=4
GPUS_PER_TASK=1
LOG_DIR="./slurm_logs"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ntip)
            NTIP="$2"
            shift 2
            ;;
        --latent-dim)
            LATENT_DIM="$2"
            shift 2
            ;;
        --weight)
            WEIGHT="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p "$LOG_DIR"

# Create SLURM job script
JOB_SCRIPT="$(mktemp /tmp/slurm_job_XXXXXX.sh)"

cat > "$JOB_SCRIPT" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=vae_ntip${NTIP}_ldim${LATENT_DIM}_w${WEIGHT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem-per-task=${MEM_PER_TASK}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus-per-task=${GPUS_PER_TASK}
#SBATCH --output=${LOG_DIR}/%x_%j.log
#SBATCH --error=${LOG_DIR}/%x_%j.err

# Load modules if needed (customize for your system)
# module load pytorch/2.0
# module load cuda/11.8

# Change to project directory
cd /Users/yitingtsai/Documents/Julia_Palacios/treeDL_project/genTreeVAE || exit 1

echo "=========================================="
echo "Job Details:"
echo "  ntip: ${NTIP}"
echo "  latent_dim: ${LATENT_DIM}"
echo "  constraint_weight: ${WEIGHT}"
echo "  epochs: ${EPOCHS}"
echo "=========================================="

# Run the training script
python3 run.py \
    --ntip ${NTIP} \
    --latent-dim ${LATENT_DIM} \
    --weight ${WEIGHT} \
    --epochs ${EPOCHS}

echo "Job completed successfully"
SLURM_EOF

# Check for dry-run flag
if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN MODE - Job will not be submitted"
    echo "=========================================="
    echo "Job script:"
    cat "$JOB_SCRIPT"
    echo "=========================================="
else
    echo "Submitting SLURM job..."
    echo "  ntip: $NTIP"
    echo "  latent_dim: $LATENT_DIM"
    echo "  constraint_weight: $WEIGHT"
    echo "  epochs: $EPOCHS"
    echo "=========================================="
    
    JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $NF}')
    echo "Job submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo "Monitor with: squeue -j $JOB_ID"
fi

# Clean up
rm -f "$JOB_SCRIPT"
