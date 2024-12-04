import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator

from .typehints import Matrix


def frobenius_norm(A: Matrix) -> float:
    """Frobenius norm of A."""
    return norm(A, ord="fro")  # type: ignore


def __operator_norm(
    rng: np.random.Generator,
    A: Matrix | LinearOperator,
    rtol: float = 1e-8,
    maxiters: int = 100,
) -> float:
    """Operator norm of A by power method."""
    n = A.shape[0]
    x = rng.standard_normal((n,))
    x /= norm(x)
    y: np.ndarray = A @ x  # type: ignore
    prev = np.inf
    eig = np.inner(y, x)
    i = 0
    while (
        norm(y) > 0
        and np.abs(prev - eig) >= rtol * np.abs(eig)
        and i < maxiters
    ):
        prev = eig
        x = y / norm(y)
        y: np.ndarray = A @ x  # type: ignore
        eig = np.inner(y, x)
        i += 1
    return np.abs(eig)


def operator_norm(
    rng: np.random.Generator,
    A: Matrix | LinearOperator,
    rtol: float = 1e-8,
    maxiters: int = 100,
    hermitian: bool = False,
) -> float:
    """Operator norm of A by power method."""
    if hermitian:
        return __operator_norm(rng, A, rtol=rtol, maxiters=maxiters)
    else:
        n, m = A.shape
        B = (
            LinearOperator((n, n), matvec=lambda x: A @ (A.T @ x))
            if n < m
            else LinearOperator((m, m), matvec=lambda x: A.T @ (A @ x))
        )
        return np.sqrt(__operator_norm(rng, B, rtol=rtol, maxiters=maxiters))
