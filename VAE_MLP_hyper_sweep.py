import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
from torch import nn, optim
from torch.nn import functional as F_nn
import torch.utils.data
from torch.utils.data import Subset, TensorDataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from __future__ import print_function
# import wandb
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Tuple, Union
import os
# import argparse


# ==========================================
# 2. HELPER FUNCTIONS (MATH & VALIDATION)
# ==========================================

def vector_to_lower_triangular(vector):
    """
    Reconstructs a full n x n F-matrix from a flattened vector containing only
    the elements strictly below the subdiagonal.

    Args:
        vector (np.array): Flattened vector containing elements F[r,c]
                           where r > c + 1.
                           The length L must satisfy L = (n-1)(n-2)/2.

    Returns:
        matrix (np.array): The reconstructed n x n lower-triangular matrix.
                           - Upper triangle (r < c) is 0.
                           - Diagonal (r == c) is fixed to [2, 3, ..., n+1].
                           - Subdiagonal (r == c + 1) is fixed to [1, 2, ..., n-1].
                           - Deep lower triangle is filled from the input vector.
    """
    # Calculate matrix dimension n
    n = int((3 + np.sqrt(1 + 8 * len(vector))) / 2)

    matrix = np.zeros((n, n))

    # 2. Fill Fixed Diagonal: 2, 3, ..., n+1
    idx_diag = np.arange(n)
    matrix[idx_diag, idx_diag] = idx_diag + 2.0

    # 3. Fill Fixed Sub-diagonal: 1, 2, ..., n-1
    idx_sub_r = np.arange(1, n)
    idx_sub_c = np.arange(0, n-1)
    matrix[idx_sub_r, idx_sub_c] = idx_sub_c + 1.0

    # 4. Fill Deep Lower Triangle (Column-Wise)
    vector_idx = 0
    for c in range(n):
        # only rows strictly below subdiagonal: r > c + 1
        for r in range(c + 2, n):
            matrix[r, c] = vector[vector_idx]
            vector_idx += 1

    return matrix


def project_to_F(F):
    """
    Projects a raw input matrix onto the set of valid discrete F-matrices by
    enforcing inequality constraints row-by-row.

    This function modifies the input matrix in-place. It assumes the
    diagonal and subdiagonal values are already set correctly and only adjusts
    the lower triangle below subdiagonal.

    Args:
        F (np.ndarray): A square (n x n) numpy array containing raw predictions.
                        - Can contain floats (will be rounded to nearest int).
                        - Upper triangle must be 0.
                        - Diagonal (i,i) must already be set to i+2.
                        - Subdiagonal (i, i-1) must already be set to i+1.

    Returns:
        np.ndarray: The same matrix 'F', now satisfying all the F-matrix constraints.
    """
    n = F.shape[0]

    # Iterate row-by-row through the DEEP lower triangle
    for i in range(2, n):

        # --- A. Column 0 ---
        prev = F[i-1, 0] # Trusted fixed/projected value from row above
        curr_raw = int(round(F[i, 0]))

        low = max(0, prev - 1)
        high = prev

        F[i, 0] = min(max(curr_raw, low), high)

        # --- B. Inner Columns ---
        for j in range(1, i-1):
            left = F[i, j-1]
            up   = F[i-1, j]
            diag = F[i-1, j-1]

            curr_raw = int(round(F[i, j]))

            l1 = left
            l2 = up - 1
            l3 = left + up - diag - 1
            final_lower = max(0, l1, l2, l3)

            u1 = up
            u2 = left + up - diag
            final_upper = min(u1, u2)

            if final_lower > final_upper:
                final_upper = final_lower

            F[i, j] = min(max(curr_raw, final_lower), final_upper)

    return F


def is_Fmat_debug(Fmat, tol=0.0):
    """
    Validates if a given matrix is a valid F-matrix.

    Args:
        Fmat (np.ndarray): The candidate matrix to check.
        tol (float): Numerical tolerance for float comparisons.

    Returns:
        int or bool:
            True if valid.
            Negative integer error code if invalid:
            -0.1: Dimensions are wrong or not 2D.
            -2: Upper triangle is not zero.
            -3: Diagonal values are incorrect.
            -4: Subdiagonal values are incorrect.
            -5: First column constraints violated.
            -6: Inner column constraints violated.
    """
    F = np.asarray(Fmat, dtype=float)

    if F.ndim != 2:
        return -0.1

    n_rows, n_cols = F.shape
    n = n_cols + 1

    # Check number of rows
    if n_rows != (n - 1):
        return -0.1

    # Upper triangular entries (strictly above diagonal) must be 0
    r_idx, c_idx = np.triu_indices_from(F, k=1)
    if not np.all(np.abs(F[r_idx, c_idx]) <= tol):
        return -2

    # Diagonal must be 2, 3, ..., n
    diag = np.diag(F)
    expected_diag = np.arange(2, n + 1)
    if not np.all(np.abs(diag - expected_diag) <= tol):
        return -3

    # Subdiagonal
    for i1 in range(1, n - 1):  # i1 = 1,...,n-2
        row = i1
        col = i1 - 1
        if not (abs(F[row, col] - i1) <= tol):
            return -4

    if n >= 4:
        for i1 in range(3, n):  # i1 = 3,...,n-1
            row = i1 - 1

            # First column constraint:
            diff1 = F[row, 0] - F[row - 1, 0]
            cond_step = (abs(diff1) <= tol) or (abs(diff1 + 1) <= tol)
            cond_nonneg = F[row, 0] >= -tol

            if not (cond_step and cond_nonneg):
                return -5

            if i1 >= 4:
                for i2 in range(2, i1 - 1):  # i2 = 2,...,i1-2
                    col = i2 - 1

                    diff_vert = F[row, col] - F[row - 1, col]
                    cond_step2 = (abs(diff_vert) <= tol) or (abs(diff_vert + 1) <= tol)

                    cond_monotone = F[row, col] >= F[row, col - 1] - tol

                    cond_convex = (
                        (F[row - 1, col] - F[row, col])
                        >= (F[row - 1, col - 1] - F[row, col - 1] - tol)
                    )

                    if not (cond_step2 and cond_monotone and cond_convex):
                        return -6

    return True


def is_Fmat(Fmat, tol=0.0):
    """
    Boolean wrapper for is_Fmat_debug.

    Args:
        Fmat (np.ndarray): Candidate matrix.
        tol (float): Tolerance.

    Returns:
        bool: True if valid F-matrix, False otherwise.
    """
    return is_Fmat_debug(Fmat, tol) > 0


def calculate_validity_percentage(matrices):
    """
    Calculates the percentage of matrices in a given batch that represent
    valid phylogenetic F-matrices.

    Args:
        matrices (np.ndarray or list): A collection of matrices (N_samples, n, n).

    Returns:
        float: The percentage (0.0 to 100.0) of matrices that are valid.
    """
    valid_count = 0
    total = len(matrices)

    for i in range(total):
        # We use a try-except block in case is_Fmat fails on
        # noisy reconstructed values that don't meet basic shape requirements
        try:
            if is_Fmat(matrices[i]):
                valid_count += 1
        except Exception:
            continue

    return (valid_count / total) * 100


@torch.jit.script
def apply_iterative_constraints(F: torch.Tensor, n: int) -> torch.Tensor:
    """
    Enforces F-matrix constraints on a batch of F-matrices.

    This function performs two simultaneous tasks:
    1. Calculates a 'Correction Loss': The sum of absolute errors where the
       matrix violates the 4-point condition or integer step constraints.
    2. Corrects the Matrix In-Place: Adjusts the values of F to satisfy
       constraints so that subsequent rows/columns rely on valid predecessors.

    Args:
        F (torch.Tensor): Batch of matrices of shape (Batch_Size, n, n).
                          This tensor is modified IN-PLACE.
        n (int): The dimension of the matrices.

    Returns:
        torch.Tensor: Scalar tensor representing the total correction loss
                      (sum of all violations across the batch).
    """

    total_correction_loss = torch.tensor(0.0, device=F.device)

    # Iterate i from 2 to n-1
    # row by row sweep
    for i in range(2, n):
        # --- Column 0 ---
        # CRITICAL: .clone() is used here to prevent in-place modification errors
        prev = F[:, i-1, 0].clone()
        curr = F[:, i, 0].clone()

        lower = torch.clamp(prev - 1.0, min=0.0)
        upper = prev

        v_low = torch.relu(lower - curr)
        v_high = torch.relu(curr - upper)
        total_correction_loss += (v_low.sum() + v_high.sum())

        # In-place update (Safe now because we cloned 'curr' and 'prev')
        F[:, i, 0] = curr + v_low - v_high

        # --- Inner Columns ---
        if i >= 3:
            for j in range(1, i-1):
                # CRITICAL: .clone() all inputs
                up = F[:, i-1, j].clone()
                left = F[:, i, j-1].clone()
                diag = F[:, i-1, j-1].clone()
                curr_val = F[:, i, j].clone()

                l1 = left
                l2 = up - 1.0
                l3 = left + up - diag - 1.0

                final_lower = torch.clamp(torch.max(l1, torch.max(l2, l3)), min=0.0)

                u1 = up
                u2 = left + up - diag
                final_upper = torch.max(torch.min(u1, u2), final_lower)

                v_low = torch.relu(final_lower - curr_val)
                v_high = torch.relu(curr_val - final_upper)

                total_correction_loss += (v_low.sum() + v_high.sum())

                F[:, i, j] = curr_val + v_low - v_high

    return total_correction_loss


# ==========================================
# 3. DATA LOADING
# ==========================================

def get_loaders(ntip):
    """
    Loads phylogenetic tree data (F-matrices) and beta-splitting labels,
    normalizes them, and creates stratified DataLoaders.

    Assumes data is stored in a specific Google Drive directory structure with
    5 distinct beta-splitting scenarios (different tree shapes).

    Args:
        ntip (int): Number of tips (taxa) in the trees. Used to select the
                    correct dataset and normalize input values.

    Returns:
        tuple: (train_loader, val_loader, test_loader, input_dim)
               - Loaders allow iteration over shuffled batches.
               - input_dim is the size of the flattened feature vector.
    """

    n_scenarios = 5   # simulate from beta-splitting model with 5 different betas
    config = config_by_ntip[ntip]
    samples_per_scenario = config['samples_per_scenario']
    base_path = f"/content/drive/MyDrive/genTreeVAE/beta_splitting_data/BF_sim_betaSplit_ntip{ntip}_eachBeta{int(samples_per_scenario/1000)}k/"
    data_path = base_path + "Fmat_below_sub.csv"
    label_path = base_path + "beta.csv"

    print(f"Loading data from: {data_path}")
    print(f"Loading labels from: {label_path}")

    # Load data
    df_X = pd.read_csv(data_path)
    X = torch.tensor(df_X.values, dtype=torch.float32) # X is a torch.FloatTensor
    X = X / ntip  # normalize to [0,1]
    input_dim = X.shape[1]

    # Load labels (betas)
    df_y = pd.read_csv(label_path)
    y = torch.tensor(df_y.values, dtype=torch.float32).view(-1)

    full_dataset = TensorDataset(X, y)

    split_ratio = config['split']
    train_frac = split_ratio[0]
    val_frac = split_ratio[1]

    # stratified
    train_per_scenario = int(train_frac * samples_per_scenario)
    val_per_scenario = int(val_frac * samples_per_scenario)
    test_per_scenario = samples_per_scenario - train_per_scenario - val_per_scenario

    rng = np.random.default_rng(seed=SEED)
    train_indices = []
    val_indices = []
    test_indices = []

    for s in range(n_scenarios):
        start = s * samples_per_scenario
        end = (s + 1) * samples_per_scenario
        indices = np.arange(start, end)

        rng.shuffle(indices)

        train_indices.extend(indices[:train_per_scenario])
        val_indices.extend(indices[train_per_scenario : train_per_scenario + val_per_scenario])
        test_indices.extend(indices[train_per_scenario + val_per_scenario:])

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    print(f"Total Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, input_dim


# ==========================================
# 4. MODEL ARCHITECTURE
# ==========================================

class FMatrixLayer(nn.Module):
    """
    A custom PyTorch layer that enforces phylogenetic constraints on a latent vector.

    This layer acts as a differentiable bridge between a flat latent vector and
    a structured F-matrix. It does not contain trainable weights itself; rather,
    it reshapes the input, applies the '4-point condition' constraints iteratively,
    and computes a correction loss.

    Attributes:
        n (int): The dimension of the full square matrix (N x N).
        rows (Tensor): Indices of rows for the deep lower triangle.
        cols (Tensor): Indices of columns for the deep lower triangle.
    """
    def __init__(self, input_dim, device):
        super().__init__()
        self.device = device

        # 1. Infer Matrix Size N from Reduced Input Dim
        # input_dim = (n-1)(n-2)/2
        self.n = int((3 + (1 + 8 * input_dim)**0.5) / 2)
        print(f"FMatrixLayer: Input={input_dim} -> inferred Matrix N={self.n}")

        # 2. Generate Indices for "Deep Lower Triangle" (Row > Col + 1)
        rows_list = []
        cols_list = []

        # Must match R extraction order (Column-Major)
        for c in range(self.n):
            for r in range(c + 2, self.n):
                rows_list.append(r)
                cols_list.append(c)

        self.register_buffer('rows', torch.tensor(rows_list, device=device))
        self.register_buffer('cols', torch.tensor(cols_list, device=device))

        # 3. Pre-compute Fixed Value Indices
        # Diagonal (k, k) -> Value k+2
        self.diag_idx = torch.arange(self.n, device=device)

        # Sub-diagonal (k+1, k) -> Value k+1
        self.sub_r = torch.arange(1, self.n, device=device)
        self.sub_c = torch.arange(0, self.n-1, device=device)

    def forward(self, x):
        """
        Forward pass: Vector -> Matrix -> Constraint Check -> Vector.

        Args:
            x (Tensor): Flat input vector of shape (Batch_Size, input_dim).
                        Values should be normalized (0-1 range).

        Returns:
            tuple:
                - F_out_flat (Tensor): The constrained/corrected vector (Batch, input_dim).
                - total_correction_loss (Tensor): Scalar loss penalizing constraint violations.
        """
        batch_size = x.size(0)
        scale = float(self.n + 1)

        F_flat = x * scale
        F = torch.zeros((batch_size, self.n, self.n), device=self.device)

        # Fill only the lower triangle below the subdiagonal
        F[:, self.rows, self.cols] = F_flat

        # 2. Force-Fill Fixed Diagonal
        target_main = self.diag_idx.float() + 2.0
        F[:, self.diag_idx, self.diag_idx] = target_main

        # Force-Fill Fixed Sub-diagonal
        target_sub = self.sub_c.float() + 1.0
        F[:, self.sub_r, self.sub_c] = target_sub

        total_correction_loss = apply_iterative_constraints(F, self.n) # the JIT function
        F_out_flat = F[:, self.rows, self.cols] / scale # Flatten BACK to vector

        return F_out_flat, total_correction_loss


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for Phylogenetic F-Matrices.

    Architecture:
    1. Encoder: Compresses the input F-matrix (flat vector) into a latent distribution
       parameterized by mean (mu) and log-variance (logvar).
    2. Sampling: Uses the reparameterization trick to sample a latent vector 'z'.
    3. Decoder: Reconstructs the flat vector from 'z'.
    4. Projection: Passes the raw reconstruction through 'FMatrixLayer' to enforce
       perfect phylogeny constraints (the 4-point condition).

    Attributes:
        input_dim (int): Size of the flattened input vector (Deep Lower Triangle).
        hidden_dims (list): List of integers defining the size of hidden layers.
        latent_dim (int): Size of the bottleneck latent space (z).
    """
    def __init__(self, input_dim, hidden_dims, latent_dim, device):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.device = device

        # Encoder
        encoder_blocks = []
        prev_dim = self.input_dim
        for h_dim in self.hidden_dims:
            encoder_blocks.append(
                nn.Sequential(
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU()
                )
            )
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_blocks)

        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        # Decoder
        decoder_blocks = []
        reversed_hidden = self.hidden_dims[::-1]
        prev_dim = self.latent_dim
        for h_dim in reversed_hidden:
            decoder_blocks.append(
                nn.Sequential(
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU()
                )
            )
            prev_dim = h_dim
        self.decoder = nn.Sequential(*decoder_blocks)

        self.fc_final = nn.Linear(self.hidden_dims[0], self.input_dim)

        # Projection Layer
        self.f_matrix_layer = FMatrixLayer(input_dim, device)

    def encode(self, x):
        """
        Passes input through encoder to get latent distribution parameters.
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """
        The 'Reparameterization Trick':
        Instead of sampling z ~ N(mu, std) directly (which breaks backpropagation),
        we sample noise eps ~ N(0, 1) and transform it: z = mu + eps * std.
        This makes the sampling process differentiable.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Reconstructs the input from the latent vector z.
        """
        h = self.decoder(z)
        out = self.fc_final(h)
        out = torch.sigmoid(out) # Ensure 0-1 range initially

        # Apply projection
        out_projected, struct_loss = self.f_matrix_layer(out)
        return out_projected, struct_loss

    def forward(self, x):
        """
        Full forward pass: Encode -> Sample -> Decode.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x, struct_loss = self.decode(z)
        return recon_x, mu, logvar, struct_loss


# ==========================================
# 5. LOSS FUNCTIONS
# ==========================================

def loss_function_Fmat(recon_x, x, mu, logvar, struct_loss, constraint_weight):
    """
    Computes the Fmat-aware loss for the Phylogenetic VAE.

    Args:
        recon_x (Tensor): Reconstructed flattened F-matrix (batch_size, input_dim).
        x (Tensor): Original input flattened F-matrix (batch_size, input_dim).
        mu (Tensor): Mean of the latent Gaussian distribution.
        logvar (Tensor): Log-variance of the latent Gaussian distribution.
        struct_loss (Tensor): Scalar loss from FMatrixLayer representing constraint violations.
        constraint_weight (float): Hyperparameter scaling the importance of structural validity.

    Returns:
        tuple: (total_loss, recon_loss, kl_loss, weighted_struct_loss)
               Returns individual components for logging/monitoring.
    """
    recon_loss = F_nn.mse_loss(recon_x, x, reduction='sum') # MSE loss for reconstruction
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # KL divergence
    weighted_struct_loss = constraint_weight * struct_loss
    total_loss = recon_loss + kl_loss + weighted_struct_loss
    return total_loss, recon_loss, kl_loss, weighted_struct_loss

# Loss function - vanilla
# Reconstruction + KL divergence losses summed over all elements and batch
# def loss_function(recon_x, x, mu, logvar):
#     recon_loss = F.mse_loss(recon_x, x, reduction='sum') # MSE loss for reconstruction
#     kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # KL divergence
#     total_loss = recon_loss + kl_loss
#     return total_loss, recon_loss, kl_loss

# Loss function - beta-VAE
# def loss_function(recon_x, x, mu, logvar, beta = 5):
#     recon_loss = F.mse_loss(recon_x, x, reduction='sum') # MSE loss for reconstruction
#     kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # KL divergence
#     total_loss = recon_loss + beta * kl_loss
#     return total_loss, recon_loss, kl_loss


# ==========================================
# 6. TRAINING & EVALUATION LOOPS
# ==========================================

def train(epoch, latent_dim, constraint_weight):
    model.train()
    train_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0
    train_struct_loss = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, struct_loss = model(data)
        loss, recon_loss, kl_loss, weighted_struct_loss = loss_function_Fmat(
            recon_batch, data, mu, logvar, struct_loss, constraint_weight = constraint_weight
            )
        loss.backward()
        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()
        train_struct_loss += weighted_struct_loss.item()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_recon_loss = train_recon_loss / len(train_loader.dataset)
    avg_train_kl_loss = train_kl_loss / len(train_loader.dataset)
    avg_train_struct_loss = train_struct_loss / len(train_loader.dataset)

    # KL per latent dim - for diagnosing collapes across latent dim
    avg_kl_per_dim = avg_train_kl_loss / latent_dim

    # comment out if not using wandb
    # wandb.log({
    #     "train/loss": avg_train_loss,
    #     "train/recon_loss": avg_train_recon_loss,
    #     "train/kl_loss": avg_train_kl_loss,
    #     "train/kl_per_dim": avg_kl_per_dim,
    #     "train/struct_loss": avg_train_struct_loss,
    #     "epoch": epoch,
    # })

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, avg_train_loss))
    return avg_train_loss


def validate(epoch, latent_dim, constraint_weight):
    model.eval()
    val_loss = 0
    val_recon_loss = 0
    val_kl_loss = 0
    val_struct_loss = 0

    with torch.no_grad():
        for (data, targets) in val_loader:
            data = data.to(device)
            recon, mu, logvar, struct_loss = model(data)

            loss, recon_loss, kl_loss, weighted_struct_loss = loss_function_Fmat(
                recon, data, mu, logvar, struct_loss, constraint_weight = constraint_weight
                )
            val_loss += loss.item()
            val_recon_loss += recon_loss.item()
            val_kl_loss += kl_loss.item()
            val_struct_loss += weighted_struct_loss.item()

    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_recon_loss = val_recon_loss / len(val_loader.dataset)
    avg_val_kl_loss = val_kl_loss / len(val_loader.dataset)
    avg_val_struct_loss = val_struct_loss / len(val_loader.dataset)
    avg_kl_per_dim = avg_val_kl_loss / latent_dim

    # comment out if not using wandb
    # wandb.log({
    #     "val/loss": avg_val_loss,
    #     "val/recon_loss": avg_val_recon_loss,
    #     "val/kl_loss": avg_val_kl_loss,
    #     "val/struct_loss": avg_val_struct_loss,
    #     "val/kl_per_dim": avg_kl_per_dim,
    # })

    print(f'====> Validation set loss: {avg_val_loss:.4f} (Recon: {avg_val_recon_loss:.4f}, KL: {avg_val_kl_loss:.4f})')
    return avg_val_loss

def test(latent_dim, constraint_weight):
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    test_struct_loss = 0

    with torch.no_grad():
        for (data, targets) in test_loader:
            data = data.to(device)
            recon, mu, logvar, struct_loss = model(data)

            loss, recon_loss, kl_loss, weighted_struct_loss = loss_function_Fmat(
                recon, data, mu, logvar, struct_loss, constraint_weight=constraint_weight
                )
            test_loss += loss.item()
            test_recon_loss += recon_loss.item()
            test_kl_loss += kl_loss.item()
            test_struct_loss += weighted_struct_loss.item()

    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_test_recon_loss = test_recon_loss / len(test_loader.dataset)
    avg_test_kl_loss = test_kl_loss / len(test_loader.dataset)
    avg_test_struct_loss = test_struct_loss / len(test_loader.dataset)
    avg_kl_per_dim = avg_test_kl_loss / latent_dim

    # comment out if not using wandb
    # wandb.log({
    #     "test/loss": avg_test_loss,
    #     "test/recon_loss": avg_test_recon_loss,
    #     "test/kl_loss": avg_test_kl_loss,
    #     "test/struct_loss": avg_test_struct_loss,
    #     "test/kl_per_dim": avg_kl_per_dim,
    # })

    print(f'====> Test set loss: {avg_test_loss:.4f} (Recon: {avg_test_recon_loss:.4f}, KL: {avg_test_kl_loss:.4f})')
    return avg_test_loss

# ==========================================
# 7. INFERENCE & EVALUATION TOOLS
# ==========================================

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


def check_inequalities_float(F, tol=1e-4):
    """
    Checks what percentage of the lower-triangular elements satisfy the 
    relaxed F-matrix inequality constraints.

    Args:
        F (np.ndarray): Square matrix (n x n) of floats.
        tol (float): Tolerance for floating point comparisons. 
                     (e.g., 0.999 is considered >= 1.0 with tol=1e-3)

    Returns:
        float: Percentage (0-100) of satisfied constraints.
    """
    n = F.shape[0]
    total_checks = 0
    satisfied_checks = 0

    # Iterate row-by-row through the DEEP lower triangle
    # (Same indices as your project_to_F: i > j + 1)
    for i in range(2, n):

        # --- A. Column 0 ---
        # Constraint: F[i,0] must be "close" to F[i-1,0] or F[i-1,0]-1
        # Relaxed: F[i-1,0] - 1 <= F[i,0] <= F[i-1,0]
        
        prev = F[i-1, 0]
        curr = F[i, 0]
        
        # Calculate bounds (Relaxed)
        lower_bound = prev - 1.0
        upper_bound = prev
        
        # We clamp the lower bound to 0 because F-matrix entries are non-negative lengths
        lower_bound = max(0.0, lower_bound)

        total_checks += 1
        if (curr >= lower_bound - tol) and (curr <= upper_bound + tol):
            satisfied_checks += 1

        # --- B. Inner Columns ---
        for j in range(1, i-1):
            left = F[i, j-1]
            up   = F[i-1, j]
            diag = F[i-1, j-1]
            curr = F[i, j]

            # 1. Calculate Lower Bound
            # Corresponds to: max(left, up-1, left+up-diag-1)
            l1 = left
            l2 = up - 1.0
            l3 = left + up - diag - 1.0
            final_lower = max(0.0, l1, l2, l3)

            # 2. Calculate Upper Bound
            # Corresponds to: min(up, left+up-diag)
            u1 = up
            u2 = left + up - diag
            final_upper = min(u1, u2)

            total_checks += 1

            # Check: Does the value fall within the valid interval?
            # Note: If neighbors are bad, it's possible final_lower > final_upper.
            # In that case, the condition is impossible, so it counts as a failure.
            if final_lower <= final_upper + tol: 
                if (curr >= final_lower - tol) and (curr <= final_upper + tol):
                    satisfied_checks += 1

    if total_checks == 0:
        return 0.0

    return (satisfied_checks / total_checks) * 100.0



def evaluate_mae(model, val_loader, ntip, device):
    """
    Calculates Mean Absolute Error (MAE) for both:
    1. Continuous (Float) predictions: Best for tracking subtle learning progress.
    2. Discrete (Integer) predictions: Best for measuring final operational accuracy.
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
            target_int = torch.round(target_float) # Ensure target is also perfect int
            total_error_int += torch.sum(torch.abs(recon_int - target_int)).item()
            
            # Count elements
            total_elements += data.numel()

    mae_float = total_error_float / total_elements
    mae_int = total_error_int / total_elements
    
    return mae_float, mae_int


def evaluate_validity_rates(model, val_loader, ntip, device):
    """
    Calculates structural validity metrics on both Float and Integer outputs.
    
    Returns:
        strict_pct (float): % of perfectly valid discrete F-matrices.
        soft_float_pct (float): Average % of satisfied constraints (Raw Floats).
        soft_int_pct (float): Average % of satisfied constraints (Rounded Integers).
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

# ==========================================
# 1. GLOBAL CONFIG
# ==========================================

config_by_ntip = {
    17: {
        "samples_per_scenario": 4000,
        "hidden_dims": [80, 64, 32],
        "latent_dims": [2, 16, 32],
        "split": [0.85, 0.075, 0.075],
    },
    33: {
        "samples_per_scenario": 10000,
        "hidden_dims": [256, 128, 64],
        "latent_dims": [2, 32, 64],
        "split": [0.9, 0.05, 0.05],
    },
    65: {
        "samples_per_scenario": 40000,
        "hidden_dims": [1024, 512, 256],
        "latent_dims": [2, 64, 128],
        "split": [0.95, 0.025, 0.025],
    }
}

SEED = 428
BATCH_SIZE = 128 # 256
LOG_INTERVAL = 10
EPOCHS = 5 # 10
CONSTRAINT_WEIGHT = 10

#### main
# ==========================================
# HYPERPARAMETER SWEEP
# ==========================================
use_accel = torch.accelerator.is_available()
torch.manual_seed(SEED)
if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_accel else {}


# 1. Setup Data (Load once)
ntip = 65
config = config_by_ntip[ntip]
# Note: Ensure get_loaders returns the correct input_dim for ntip=17
train_loader, val_loader, test_loader, input_dim = get_loaders(ntip)

h_dims = config['hidden_dims']

# 2. Define Sweep Parameters
latent_dims_to_try = config['latent_dims']
weights_to_try = [0.01, 0.1, 1.0, 10.0, 100.0]
sweep_results = []


print(f"\nSTARTING SWEEP: NTIP={ntip}")

# Updated Header to include separate Float/Int metrics
# MAE(F): Float MAE, MAE(I): Integer MAE
# Soft(F): Float Soft Validity, Soft(I): Integer Soft Validity
header_fmt = "{:<6} | {:<6} | {:<7} | {:<7} | {:<7} | {:<8} | {:<8} | {:<6}"
print(header_fmt.format("L_Dim", "Weight", "MAE(F)", "MAE(I)", "Strict%", "Soft(F)%", "Soft(I)%", "Active"))
print("-" * 85)

for l_dim in latent_dims_to_try: # removed [1:] so you test all dims
    for w in weights_to_try:
        
        # Initialize Model
        model = VAE(input_dim, h_dims, l_dim, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Training Loop
        for epoch in range(1, EPOCHS + 1):
            # Ensure train() accepts constraint_weight
            train_loss = train(epoch, l_dim, constraint_weight=w) 
        
        # 3. Evaluation (Returns dict with separate Float/Int metrics)
        metrics = full_evaluation(model, val_loader, ntip, device)
        
        # 4. Store Results
        res = {
            'latent_dim': l_dim,
            'weight': w,
            **metrics # Unpacks MAE_Float, MAE_Int, Strict_Validity, Soft_Valid_Float, Soft_Valid_Int, etc.
        }
        sweep_results.append(res)
        
        # 5. Print Progress
        print(header_fmt.format(
            l_dim, 
            w, 
            f"{metrics['MAE_Float']:.4f}", 
            f"{metrics['MAE_Int']:.4f}", 
            f"{metrics['Strict_Validity']:.1f}", 
            f"{metrics['Soft_Valid_Float']:.1f}", 
            f"{metrics['Soft_Valid_Int']:.1f}", 
            metrics['Active_Dims']
        ))


        # 6. Save Intermediate Results
        df_results = pd.DataFrame(sweep_results)

        # Reorder columns for readability
        desired_order = [
            'latent_dim', 'weight', 
            'MAE_Float', 'MAE_Int', 
            'Strict_Validity', 'Soft_Valid_Float', 'Soft_Valid_Int', 
            'Active_Dims', 'Avg_KL'
        ]
        
        # Filter to only include columns that actually exist
        cols = [c for c in desired_order if c in df_results.columns]
        df_results = df_results[cols]

        # Construct Save Path
        base_path = f"/content/drive/MyDrive/genTreeVAE/results/"
        os.makedirs(base_path, exist_ok=True) 
        
        # Save individual run file
        save_path = base_path + f"hyperparam_sweep_ntip{ntip}_ldim{l_dim}_w{int(w*100)}.csv"
        df_results.to_csv(save_path, index=False)
        