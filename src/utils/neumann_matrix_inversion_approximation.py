#
# import numpy as np
#
#
# def neumann_matrix_inversion_approximation(
#     matrix: np.ndarray,
#     order: int,
#     diagonal_loading: float = 1e-18,
# ) -> np.ndarray:
#     if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
#         raise ValueError("matrix must be square")
#
#     identity_matrix = np.eye(matrix.shape[0], dtype=matrix.dtype)
#
#     working_matrix = matrix.copy()
#     working_matrix = working_matrix / np.linalg.norm(working_matrix, ord='fro')
#
#     # if not np.allclose(working_matrix, working_matrix.conj().T, atol=1e-20):
#     #     # Falls nicht hermitesch: direkt robuster Fallback
#     #     try:
#     #         inverse = np.linalg.pinv(working_matrix)
#     #         if np.all(np.isfinite(inverse)):
#     #             return inverse / np.linalg.norm(working_matrix, ord='fro')
#     #     except Exception:
#     #         return np.zeros_like(working_matrix)
#
#     eigenvalues = np.linalg.eigvalsh(working_matrix)
#     eigenvalue_min = np.min(eigenvalues)
#     eigenvalue_max = np.max(eigenvalues)
#
#     # Fall 1: Matrix ist gut genug positiv definit
#     if eigenvalue_min > 1e-12:
#         scaling_factor = 2.0 / (eigenvalue_min + eigenvalue_max)
#
#         operator = identity_matrix - scaling_factor * working_matrix
#
#         approximation = identity_matrix.copy()
#         current_power = identity_matrix.copy()
#
#         for _ in range(1, order + 1):
#             current_power = current_power @ operator
#             approximation += current_power
#
#         inverse = scaling_factor * approximation
#
#         if np.all(np.isfinite(inverse)):
#             return inverse / np.linalg.norm(working_matrix, ord='fro')
#
#
#     # Fall 2: Letzte Notlösung
#     return np.zeros_like(matrix)

# import numpy as np
#
# def neumann_matrix_inversion_approximation(
#     matrix: np.ndarray,
#     order: int,
# ) -> np.ndarray:
#     if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
#         raise ValueError("matrix must be square")
#
#     n = matrix.shape[0]
#     identity_matrix = np.eye(n, dtype=matrix.dtype)
#
#     fro_norm = np.linalg.norm(matrix, ord='fro')
#     if fro_norm == 0:
#         raise ValueError("zero matrix is not invertible")
#
#     normalized_matrix = matrix / fro_norm
#
#     # Only for hermitian
#     if not np.allclose(normalized_matrix, normalized_matrix.conj().T, atol=1e-12):
#         raise ValueError("Neumann version here requires Hermitian matrix")
#
#     eigvals = np.linalg.eigvalsh(normalized_matrix)
#     eigenvalue_min = np.min(eigvals)
#     eigenvalue_max = np.max(eigvals)
#
#     if eigenvalue_min <= 0:
#         raise ValueError("matrix must be positive definite")
#
#     scaling_factor = 2.0 / (eigenvalue_min + eigenvalue_max)
#     operator = identity_matrix - scaling_factor * normalized_matrix
#
#     approx = identity_matrix.copy()
#     power = identity_matrix.copy()
#
#     for _ in range(order):
#         power = power @ operator
#         approx += power
#
#     A_norm_inv_approx = scaling_factor * approx
#
#     # Rückskalierung auf inverse der Originalmatrix
#     A_inv_approx = A_norm_inv_approx / fro_norm
#     return A_inv_approx

import numpy as np

def neumann_matrix_inversion_approximation(
    matrix: np.ndarray,
    order: int,
) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    n = matrix.shape[0]
    identity_matrix = np.eye(n, dtype=matrix.dtype)

    fro_norm = np.linalg.norm(matrix, ord='fro')
    if fro_norm == 0 or not np.isfinite(fro_norm):
        return np.zeros_like(matrix)

    normalized_matrix = matrix / fro_norm

    # Only for Hermitian matrices
    if not np.allclose(normalized_matrix, normalized_matrix.conj().T, atol=1e-12):
        return np.zeros_like(matrix)

    try:
        eigvals = np.linalg.eigvalsh(normalized_matrix)
    except Exception:
        return np.zeros_like(matrix)

    if not np.all(np.isfinite(eigvals)):
        return np.zeros_like(matrix)

    eigenvalue_min = np.min(eigvals)
    eigenvalue_max = np.max(eigvals)

    # Need positive definiteness
    if eigenvalue_min <= 1e-12:
        return np.zeros_like(matrix)

    denominator = eigenvalue_min + eigenvalue_max
    if denominator <= 0 or not np.isfinite(denominator):
        return np.zeros_like(matrix)

    scaling_factor = 2.0 / denominator
    if not np.isfinite(scaling_factor):
        return np.zeros_like(matrix)

    operator = identity_matrix - scaling_factor * normalized_matrix
    if not np.all(np.isfinite(operator)):
        return np.zeros_like(matrix)

    approx = identity_matrix.copy()
    power = identity_matrix.copy()

    for _ in range(order):
        power = power @ operator
        if not np.all(np.isfinite(power)):
            return np.zeros_like(matrix)

        approx += power
        if not np.all(np.isfinite(approx)):
            return np.zeros_like(matrix)

    normalized_inverse_approximation = scaling_factor * approx
    if not np.all(np.isfinite(normalized_inverse_approximation)):
        return np.zeros_like(matrix)

    matrix_inverse_approximation = normalized_inverse_approximation / fro_norm
    if not np.all(np.isfinite(matrix_inverse_approximation)):
        return np.zeros_like(matrix)

    return matrix_inverse_approximation