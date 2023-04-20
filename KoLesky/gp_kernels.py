import numpy as np
import sklearn.gaussian_process.kernels as kernels

from .typehints import Indices, Matrix, Sparse


class MatrixKernel(kernels.Kernel):
    """A wrapper over Kernel for matrices."""

    def __init__(self, m: Matrix) -> None:
        # we can't use the variable name "theta" for scikit-learn API reasons
        # force Fortran order memory contiguity for ease in Cython wrapping
        self.m = np.asfortranarray(m)

    def __flatten(self, m: Matrix) -> Indices:
        """Flatten m for use in indexing."""
        return np.asarray(m).flatten().astype(np.int64)

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        eval_gradient: bool = False,  # pyright: ignore
    ) -> Matrix:
        """Return the kernel k(X, Y) and possibly its gradient."""
        if Y is None:
            Y = X
        return self.m[np.ix_(self.__flatten(X), self.__flatten(Y))]

    def diag(self, X: np.ndarray) -> Matrix:
        """Returns the diagonal of the kernel k(X, X)."""
        return self.m[self.__flatten(X), self.__flatten(X)]

    def is_stationary(self) -> bool:
        """Returns whether the kernel is stationary."""
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(m={repr(self.m)})"


class DotKernel(kernels.Kernel):
    """A wrapper over Kernel for sparse matrices."""

    def __call__(
        self,
        X: Sparse,
        Y: Sparse | None = None,
        eval_gradient: bool = False,  # pyright: ignore
    ) -> Matrix:
        """Return the kernel k(X, Y) and possibly its gradient."""
        if Y is None:
            Y = X
        return X.dot(Y.T)

    def diag(self, X: Sparse) -> Matrix:
        """Returns the diagonal of the kernel k(X, X)."""
        return np.array([row.dot(row.T)[0, 0] for row in X])

    def is_stationary(self) -> bool:
        """Returns whether the kernel is stationary."""
        return False


def matrix_kernel(theta: Matrix) -> tuple[Matrix, MatrixKernel]:
    """Turns a matrix into "points" and a kernel function."""
    points = np.arange(theta.shape[0], dtype=np.float64).reshape(-1, 1)
    return points, MatrixKernel(theta)
