# ==========================================
# MATH HELPERS & F-MATRIX UTILITIES
# ==========================================

import numpy as np


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
        prev = F[i-1, 0]  # Trusted fixed/projected value from row above
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
