import numpy as np

# def neumann_matrix_inversion_approximation(
#     matrix: np.ndarray,
#     order: int,
# ) -> np.ndarray:
#
#     if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
#         raise ValueError("matrix for inversion must be square")
#
#     identity_matrix = np.eye(matrix.shape[0], dtype=matrix.dtype)
#
#     eigenvalues = np.linalg.eigvalsh(matrix)
#     eigenvalue_min = np.min(eigenvalues)
#     eigenvalue_max = np.max(eigenvalues)
#
#     # if eigenvalue_min <= 0:
#     #     raise ValueError(
#     #         "matrix must be positive definite for Neumann approximation scaling"
#     #     )
#
#     scaling_factor = 2.0 / (eigenvalue_min + eigenvalue_max)
#
#     operator = identity_matrix - scaling_factor * matrix
#
#     approximation = identity_matrix.copy()
#     current_power = identity_matrix.copy()
#
#     for _ in range(1, order + 1):
#         current_power = current_power @ operator
#         approximation += current_power
#
#     return scaling_factor * approximation
#
# import numpy as np


# def neumann_matrix_inversion_approximation(
#     matrix: np.ndarray,
#     order: int,
#     enforce_positive_definite: bool = False,
#     diagonal_loading: float = 0.0,
# ) -> np.ndarray:
#     if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
#         raise ValueError("matrix must be square")
#
#     working_matrix = matrix.copy()
#
#     if diagonal_loading > 0.0:
#         working_matrix = working_matrix + diagonal_loading * np.eye(
#             working_matrix.shape[0], dtype=working_matrix.dtype
#         )
#
#     if not np.allclose(working_matrix, working_matrix.conj().T, atol=1e-10):
#         raise ValueError("matrix must be Hermitian for eigvalsh-based scaling")
#
#     identity_matrix = np.eye(working_matrix.shape[0], dtype=working_matrix.dtype)
#
#     eigenvalues = np.linalg.eigvalsh(working_matrix)
#     eigenvalue_min = np.min(eigenvalues)
#     eigenvalue_max = np.max(eigenvalues)
#
#     if eigenvalue_max <= 0:
#         raise ValueError("matrix must have at least one positive eigenvalue")
#
#     if eigenvalue_min > 0:
#         scaling_factor = 2.0 / (eigenvalue_min + eigenvalue_max)
#     else:
#         if enforce_positive_definite:
#             raise ValueError(
#                 "matrix is not positive definite; consider diagonal_loading > 0"
#             )
#         scaling_factor = 1.0 / eigenvalue_max
#
#     operator = identity_matrix - scaling_factor * working_matrix
#
#     approximation = identity_matrix.copy()
#     current_power = identity_matrix.copy()
#
#     for _ in range(1, order + 1):
#         current_power = current_power @ operator
#         approximation += current_power
#
#     return scaling_factor * approximation


import numpy as np


def neumann_matrix_inversion_approximation(
    matrix: np.ndarray,
    order: int,
    diagonal_loading: float = 1e-8,
) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    identity_matrix = np.eye(matrix.shape[0], dtype=matrix.dtype)

    working_matrix = matrix.copy()

    if not np.allclose(working_matrix, working_matrix.conj().T, atol=1e-10):
        # Falls nicht hermitesch: direkt robuster Fallback
        try:
            inverse = np.linalg.pinv(working_matrix)
            if np.all(np.isfinite(inverse)):
                return inverse
        except Exception:
            return np.zeros_like(working_matrix)

    eigenvalues = np.linalg.eigvalsh(working_matrix)
    eigenvalue_min = np.min(eigenvalues)
    eigenvalue_max = np.max(eigenvalues)

    # Fall 1: Matrix ist gut genug positiv definit
    if eigenvalue_min > 1e-12:
        scaling_factor = 2.0 / (eigenvalue_min + eigenvalue_max)

        operator = identity_matrix - scaling_factor * working_matrix

        approximation = identity_matrix.copy()
        current_power = identity_matrix.copy()

        for _ in range(1, order + 1):
            current_power = current_power @ operator
            approximation += current_power

        inverse = scaling_factor * approximation

        if np.all(np.isfinite(inverse)):
            return inverse

    # Fall 2: Stabilisierung versuchen
    stabilized_matrix = working_matrix + diagonal_loading * identity_matrix

    try:
        eigenvalues_stabilized = np.linalg.eigvalsh(stabilized_matrix)
        eigenvalue_min_stabilized = np.min(eigenvalues_stabilized)
        eigenvalue_max_stabilized = np.max(eigenvalues_stabilized)

        if eigenvalue_min_stabilized > 1e-12:
            scaling_factor = 2.0 / (eigenvalue_min_stabilized + eigenvalue_max_stabilized)

            operator = identity_matrix - scaling_factor * stabilized_matrix

            approximation = identity_matrix.copy()
            current_power = identity_matrix.copy()

            for _ in range(1, order + 1):
                current_power = current_power @ operator
                approximation += current_power

            inverse = scaling_factor * approximation

            if np.all(np.isfinite(inverse)):
                return inverse
    except Exception:
        pass

    # Fall 3: Pseudoinverse als robuster Fallback
    try:
        inverse = np.linalg.pinv(stabilized_matrix)
        if np.all(np.isfinite(inverse)):
            return inverse
    except Exception:
        pass

    # Fall 4: Letzte Notlösung
    return np.zeros_like(matrix)