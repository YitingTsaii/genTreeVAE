# ==========================================
# CONFIGURATION FILE
# ==========================================

# Global Constants
SEED = 428
BATCH_SIZE = 128
LOG_INTERVAL = 10
EPOCHS = 5
CONSTRAINT_WEIGHT = 10

# Data paths (adjust to your setup)
# Prefer local workspace `data/` and `results/` directories by default
# Use project-root-relative paths so running `python3 run/run.py` works from root
BASE_DATA_PATH = "data/"
RESULTS_PATH = "results/"

# Configuration by number of tips
config_by_ntip = {
    9: {
        "samples_per_scenario": 1000,
        "hidden_dims": [40, 32, 16],
        "latent_dims": [2, 4, 8], #[2, 8, 16],
        "split": [0.8, 0.1, 0.1],
    },
    17: {
        "samples_per_scenario": 4000,
        "hidden_dims": [80, 64, 32],
        "latent_dims": [2, 8, 16], #[2, 16, 32],
        "split": [0.85, 0.075, 0.075],
    },
    33: {
        "samples_per_scenario": 10000,
        "hidden_dims": [256, 128, 64],
        "latent_dims": [2, 16, 32], #[2, 32, 64],
        "split": [0.9, 0.05, 0.05],
    },
    65: {
        "samples_per_scenario": 40000,
        "hidden_dims": [1024, 512, 256],
        "latent_dims": [2, 32, 64], #[2, 64, 128],
        "split": [0.95, 0.025, 0.025],
    }
}

# Hyperparameter sweep configurations
WEIGHTS_TO_TRY = [0.01, 0.1, 1.0, 10.0, 100.0]
