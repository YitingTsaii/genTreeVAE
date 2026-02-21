#!/usr/bin/env python3
# ==========================================
# MAIN ENTRY POINT - VAE Training Script
# ==========================================

import argparse
import os
import random

import sys
import pandas as pd
import torch
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import (
    config_by_ntip, SEED, EPOCHS, RESULTS_PATH, WEIGHTS_TO_TRY
)
from src.data.loader import get_loaders
from src.models import VAE
from src.utils.training import train, validate
from src.evaluation.metrics import full_evaluation


def main(args):
    """
    Main training and evaluation function.

    Args:
        args: Command-line arguments
    """
    # Setup device
    use_accel = torch.accelerator.is_available()

    # Seed selection: CLI-provided seed overrides config.SEED
    seed = args.seed if getattr(args, 'seed', None) is not None else SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"=" * 80)

    # Get configuration
    ntip = args.ntip
    latent_dim = args.latent_dim
    constraint_weight = args.weight
    num_epochs = args.epochs

    if ntip not in config_by_ntip:
        print(f"Error: ntip={ntip} not in config. Available: {list(config_by_ntip.keys())}")
        sys.exit(1)

    config = config_by_ntip[ntip]
    h_dims = config['hidden_dims']

    # Load data
    print(f"\nLoading data for ntip={ntip}...")
    train_loader, val_loader, test_loader, input_dim = get_loaders(ntip)

    # Initialize model
    print(f"\nInitializing VAE with latent_dim={latent_dim}, constraint_weight={constraint_weight}")
    model = VAE(input_dim, h_dims, latent_dim, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"=" * 80)
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, optimizer, device, epoch, latent_dim, constraint_weight)
        val_loss = validate(model, val_loader, device, epoch, latent_dim, constraint_weight)

    # Evaluation
    print(f"\n{'=' * 80}")
    print("Running full evaluation on validation set...")
    metrics = full_evaluation(model, val_loader, ntip, device)

    # Print results
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS:")
    print(f"  ntip={ntip}, latent_dim={latent_dim}, constraint_weight={constraint_weight}")
    print(f"  MAE (Float): {metrics['MAE_Float']:.6f}")
    print(f"  MAE (Int):   {metrics['MAE_Int']:.6f}")
    print(f"  Strict Validity (%): {metrics['Strict_Validity']:.2f}")
    print(f"  Soft Valid Float (%): {metrics['Soft_Valid_Float']:.2f}")
    print(f"  Soft Valid Int (%): {metrics['Soft_Valid_Int']:.2f}")
    print(f"  Active Latent Dims: {metrics['Active_Dims']}")
    print(f"  Avg KL: {metrics['Avg_KL']:.6f}")
    print(f"=" * 80)

    # Save results
    os.makedirs(RESULTS_PATH, exist_ok=True)
    result_dict = {
        'latent_dim': latent_dim,
        'weight': constraint_weight,
        'seed': seed,
        **metrics
    }
    
    save_path = os.path.join(
        RESULTS_PATH,
        f"run_ntip{ntip}_ldim{latent_dim}_w{constraint_weight}.csv"
    )
    
    df = pd.DataFrame([result_dict])
    df.to_csv(save_path, index=False)
    print(f"Results saved to: {save_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train VAE for phylogenetic F-matrix prediction"
    )
    
    parser.add_argument(
        "--ntip",
        type=int,
        default=65,
        choices=[9, 17, 33, 65],
        help="Number of tips (taxa) in the trees"
    )
    
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=32,
        help="Dimension of the latent space"
    )
    
    parser.add_argument(
        "--weight",
        type=float,
        default=10.0,
        help="Constraint weight for structural validity loss"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config.SEED)"
    )
    
    args = parser.parse_args()
    main(args)
