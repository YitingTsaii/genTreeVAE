#!/bin/bash
#SBATCH --job-name=treeVAE_sweep
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=0-14  # 3 latent_dims * 5 weights = 15 total jobs

# 1. Define your parameter lists
LATENT_DIMS=(2 4) #(2 64 128)
WEIGHTS=(0.01 0.1) #(0.001 0.01 0.1 1.0 10.0)

# 2. Use the Array Task ID to pick a unique combination
# This math maps the 0-14 index to your two lists
DIM_IDX=$((SLURM_ARRAY_TASK_ID / 5))
W_IDX=$((SLURM_ARRAY_TASK_ID % 5))

DIM=${LATENT_DIMS[$DIM_IDX]}
W=${WEIGHTS[$W_IDX]}

# 3. Load your environment (change to your actual conda/module setup)
# module load python/3.9
# source activate your_env

echo "Running Task $SLURM_ARRAY_TASK_ID: Latent Dim $DIM, Weight $W on GPU $CUDA_VISIBLE_DEVICES"

# 4. Execute the command
python3 run.py \
    --ntip 9 \
    --latent-dim "$DIM" \
    --epochs 5 \
    --weight "$W"