import numpy as np

def neumann_matrix_inversion_approximation(
    matrix: np.ndarray,
    order: int,
) -> np.ndarray:

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix for inversion must be square")

    identity_matrix = np.eye(matrix.shape[0], dtype=matrix.dtype)

    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalue_min = np.min(eigenvalues)
    eigenvalue_max = np.max(eigenvalues)

    # if eigenvalue_min <= 0:
    #     raise ValueError(
    #         "matrix must be positive definite for Neumann approximation scaling"
    #     )

    scaling_factor = 2.0 / (eigenvalue_min + eigenvalue_max)

    operator = identity_matrix - scaling_factor * matrix

    approximation = identity_matrix.copy()
    current_power = identity_matrix.copy()

    for _ in range(1, order + 1):
        current_power = current_power @ operator
        approximation += current_power

    return scaling_factor * approximation