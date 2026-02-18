# ==========================================
# INFERENCE & EVALUATION TOOLS
# ==========================================

import numpy as np
import torch
from ..utils.math_helpers import vector_to_lower_triangular, project_to_F
from ..utils.validation import is_Fmat, check_inequalities_float


def get_reconstructions_projected(model, test_loader, ntip, device):
    """
    Generates valid F-matrices from the VAE model on the test set.

    This function handles the full pipeline:
    1. Inference: Runs the VAE on test data.
    2. Denormalization: Scales outputs back to integer counts (using ntip).
    3. Reshaping: Converts flat vectors back to lower-triangular matrices.
    4. Projection: Enforces strict mathematical constraints (integer, 4-point condition)
       to ensure the output is a valid evolutionary history.

    Args:
        model (nn.Module): The trained VAE model.
        test_loader (DataLoader): DataLoader for the test dataset.
        ntip (int): The scaling factor (number of tips/taxa) used to normalize
                    the data during training. Used here to un-scale.
        device (torch.device): 'cpu' or 'cuda'.

    Returns:
        tuple: (final_projected, final_original)
            - final_projected: np.array of shape (N_samples, n, n) containing
                               the corrected, valid F-matrices.
            - final_original: np.array of shape (N_samples, n, n) containing
                              the ground truth matrices for comparison.
    """
    model.eval()

    all_projected = []
    all_original_matrices = []

    print("Starting reconstruction and projection...")

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)

            # A. Get Raw Output
            # Model returns (recon_x, mu, logvar, struct_loss)
            recon_normalized, _, _, _ = model(data)

            # B. Scale back to real values
            recon_scaled = recon_normalized * ntip
            original_scaled = data * ntip

            # Move to CPU for numpy processing
            recon_np = recon_scaled.cpu().numpy()
            orig_np = original_scaled.cpu().numpy()

            # C. Iterate through batch to Project row-by-row
            for i in range(len(recon_np)):
                # 1. Convert Vector -> Matrix
                mat_raw = vector_to_lower_triangular(recon_np[i])
                mat_orig = vector_to_lower_triangular(orig_np[i])

                # 2. Apply Hard Projection
                mat_projected = project_to_F(mat_raw)

                all_projected.append(mat_projected)
                all_original_matrices.append(mat_orig)

    # Stack into big arrays: (N_samples, n, n)
    final_projected = np.array(all_projected)
    final_original = np.array(all_original_matrices)

    print(f"Done. Returned shapes: Projected {final_projected.shape}, Original {final_original.shape}")

    return final_projected, final_original


def count_active_units(model, loader, device, threshold=0.01):
    """
    Counts the number of active latent dimensions.

    Args:
        model: The VAE model
        loader: DataLoader
        device: Device to use
        threshold: Variance threshold for considering a unit active

    Returns:
        tuple: (num_active, avg_variance)
    """
    model.eval()
    all_mus = []

    # 1. Collect all mean vectors
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            all_mus.append(mu)

    # Concatenate to shape (N_samples, latent_dim)
    all_mus = torch.cat(all_mus, dim=0)

    # 2. Calculate Variance across the dataset axis (dim=0)
    # This measures how much the mean *changes* for different inputs
    variances = torch.var(all_mus, dim=0)

    # 3. Count active
    active_mask = variances > threshold
    num_active = active_mask.sum().item()

    return num_active, variances.mean().item()


def evaluate_mae(model, val_loader, ntip, device):
    """
    Calculates Mean Absolute Error (MAE) for both:
    1. Continuous (Float) predictions: Best for tracking subtle learning progress.
    2. Discrete (Integer) predictions: Best for measuring final operational accuracy.

    Args:
        model: The VAE model
        val_loader: DataLoader
        ntip: Number of tips (for denormalization)
        device: Device to use

    Returns:
        tuple: (mae_float, mae_int)
    """
    model.eval()
    total_error_float = 0.0
    total_error_int = 0.0
    total_elements = 0

    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)

            # 1. Forward Pass
            recon_norm, _, _, _ = model(data)

            # 2. Denormalize
            recon_float = recon_norm * ntip
            target_float = data * ntip

            # 3. Calc Float MAE (Raw values)
            total_error_float += torch.sum(torch.abs(recon_float - target_float)).item()

            # 4. Calc Integer MAE (Rounded values)
            recon_int = torch.round(recon_float)
            target_int = torch.round(target_float)  # Ensure target is also perfect int
            total_error_int += torch.sum(torch.abs(recon_int - target_int)).item()

            # Count elements
            total_elements += data.numel()

    mae_float = total_error_float / total_elements
    mae_int = total_error_int / total_elements

    return mae_float, mae_int


def evaluate_validity_rates(model, val_loader, ntip, device):
    """
    Calculates structural validity metrics on both Float and Integer outputs.

    Args:
        model: The VAE model
        val_loader: DataLoader
        ntip: Number of tips (for denormalization)
        device: Device to use

    Returns:
        tuple: (strict_pct, soft_float_pct, soft_int_pct)
    """
    model.eval()

    total_strict_valid = 0
    total_soft_float = 0.0
    total_soft_int = 0.0
    total_samples = 0

    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)

            # 1. Forward Pass
            recon_norm, _, _, _ = model(data)

            # 2. Denormalize to Raw Floats
            recon_float = recon_norm * ntip

            batch_size = data.size(0)
            for i in range(batch_size):
                vec_float = recon_float[i].cpu()

                # --- A. Analyze Raw Floats ---
                # Convert to Matrix (Assumes vector_to_lower_triangular handles Diag/Subdiag)
                mat_float = vector_to_lower_triangular(vec_float)

                # Ensure numpy
                if isinstance(mat_float, torch.Tensor):
                    mat_float = mat_float.numpy()

                # Metric 1: Soft Validity (Float)
                # How close is the raw geometry to being valid?
                soft_score_f = check_inequalities_float(mat_float)
                total_soft_float += soft_score_f

                # --- B. Analyze Rounded Integers ---
                # Round to nearest integer
                mat_int = np.round(mat_float).astype(int)

                # Metric 2: Soft Validity (Int)
                # How many discrete constraints are satisfied after rounding?
                soft_score_i = check_inequalities_float(mat_int)
                total_soft_int += soft_score_i

                # Metric 3: Strict Validity (Int)
                # Is the matrix 100% perfect?
                if is_Fmat(mat_int):
                    total_strict_valid += 1

            total_samples += batch_size

    # Calculate Averages
    strict_pct = (total_strict_valid / total_samples) * 100.0
    soft_float_pct = total_soft_float / total_samples
    soft_int_pct = total_soft_int / total_samples

    return strict_pct, soft_float_pct, soft_int_pct


def full_evaluation(model, val_loader, ntip, device):
    """
    Comprehensive evaluation of the model.

    Args:
        model: The VAE model
        val_loader: DataLoader
        ntip: Number of tips (for denormalization)
        device: Device to use

    Returns:
        dict: Dictionary with evaluation metrics
    """
    # 1. Get Reconstruction Errors (Float & Int)
    mae_float, mae_int = evaluate_mae(model, val_loader, ntip, device)

    # 2. Get Structural Validity (Strict, Soft-Float, Soft-Int)
    strict_valid, soft_float, soft_int = evaluate_validity_rates(model, val_loader, ntip, device)

    # 3. Get Latent Space Stats
    active_dims, avg_kl = count_active_units(model, val_loader, device)

    return {
        "MAE_Float": mae_float,       # Use this for hyperparam selection
        "MAE_Int": mae_int,           # "Real" error
        "Strict_Validity": strict_valid,
        "Soft_Valid_Float": soft_float,
        "Soft_Valid_Int": soft_int,
        "Active_Dims": active_dims,
        "Avg_KL": avg_kl
    }
