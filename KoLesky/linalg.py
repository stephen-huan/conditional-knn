import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator

from .typehints import Matrix


def frobenius_norm(A: Matrix) -> float:
    """Frobenius norm of A."""
    return norm(A, ord="fro")  # type: ignore


def operator_norm(
    rng: np.random.Generator, A: Matrix | LinearOperator, eps: float = 1e-8
) -> float:
    """Operator norm of A by power method."""
    n = A.shape[0]
    x = rng.standard_normal((n,))
    x /= norm(x)
    y: np.ndarray = A @ x  # type: ignore
    while norm(y - np.inner(y, x) * x) > eps:
        x = y / norm(y)
        y: np.ndarray = A @ x  # type: ignore
    return np.abs(np.inner(y, x))
