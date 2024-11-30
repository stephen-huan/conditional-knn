import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator

from .typehints import Matrix


def frobenius_norm(A: Matrix) -> float:
    """Frobenius norm of A."""
    return norm(A, ord="fro")  # type: ignore


def __operator_norm(
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


def operator_norm(
    rng: np.random.Generator,
    A: Matrix | LinearOperator,
    eps: float = 1e-8,
    hermitian: bool = False,
) -> float:
    """Operator norm of A by power method."""
    if hermitian:
        return __operator_norm(rng, A, eps)
    else:
        n, m = A.shape
        B = (
            LinearOperator((n, n), matvec=lambda x: A @ (A.T @ x))
            if n < m
            else LinearOperator((m, m), matvec=lambda x: A.T @ (A @ x))
        )
        return np.sqrt(__operator_norm(rng, B, eps))
