# ==========================================
# VALIDATION & F-MATRIX CHECKING
# ==========================================

import numpy as np


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
        try:
            if is_Fmat(matrices[i]):
                valid_count += 1
        except Exception:
            continue

    return (valid_count / total) * 100


def check_inequalities_float(F, tol=1e-4):
    """
    Checks what percentage of the lower-triangular elements satisfy the 
    relaxed F-matrix inequality constraints.

    Args:
        F (np.ndarray): Square matrix (n x n) of floats.
        tol (float): Tolerance for floating point comparisons.

    Returns:
        float: Percentage (0-100) of satisfied constraints.
    """
    n = F.shape[0]
    total_checks = 0
    satisfied_checks = 0

    # Iterate row-by-row through the DEEP lower triangle
    for i in range(2, n):

        # --- A. Column 0 ---
        prev = F[i-1, 0]
        curr = F[i, 0]

        lower_bound = prev - 1.0
        upper_bound = prev

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

            l1 = left
            l2 = up - 1.0
            l3 = left + up - diag - 1.0
            final_lower = max(0.0, l1, l2, l3)

            u1 = up
            u2 = left + up - diag
            final_upper = min(u1, u2)

            total_checks += 1

            if final_lower <= final_upper + tol:
                if (curr >= final_lower - tol) and (curr <= final_upper + tol):
                    satisfied_checks += 1

    if total_checks == 0:
        return 0.0

    return (satisfied_checks / total_checks) * 100.0
