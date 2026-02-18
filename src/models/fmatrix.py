# ==========================================
# F-MATRIX LAYER
# ==========================================

import torch
from torch import nn


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

        total_correction_loss = apply_iterative_constraints(F, self.n)  # the JIT function
        F_out_flat = F[:, self.rows, self.cols] / scale  # Flatten BACK to vector

        return F_out_flat, total_correction_loss
