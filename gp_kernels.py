import numpy as np
import scipy.sparse as sparse
import sklearn.gaussian_process.kernels as kernels

Kernel = kernels.Kernel

class MatrixKernel(Kernel):
    """
    A wrapper over sklearn.gaussian_process.kernels.Kernel for matrices.
    """

    def __init__(self, m: np.ndarray) -> None:
        # we can't use the variable name "theta" for scikit-learn API reasons
        # force Fortran order memory contiguity for ease in Cython wrapping
        self.m = np.asfortranarray(m)

    def __flatten(self, m: np.ndarray) -> np.ndarray:
        """ Flatten m for use in indexing. """
        return np.array(m).flatten().astype(np.int64)

    def __call__(self, X: np.ndarray, Y: np.ndarray=None,
                 eval_gradient: bool=False) -> np.ndarray:
        """ Return the kernel k(X, Y) and possibly its gradient. """
        if Y is None: Y = X
        return self.m[np.ix_(self.__flatten(X), self.__flatten(Y))]

    def diag(self, X: np.ndarray) -> np.ndarray:
        """ Returns the diagonal of the kernel k(X, X). """
        return self.m[self.__flatten(X), self.__flatten(X)]

    def is_stationary(self) -> bool:
        """ Returns whether the kernel is stationary. """
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(m={repr(self.m)})"

class DotKernel(Kernel):
    """
    A wrapper over sklearn.gaussian_process.kernels.Kernel for sparse matrices.
    """

    def __call__(self, X: sparse.csc_matrix, Y: sparse.csc_matrix=None,
                 eval_gradient: bool=False) -> np.ndarray:
        """ Return the kernel k(X, Y) and possibly its gradient. """
        if Y is None: Y = X
        return X.dot(Y.T)

    def diag(self, X: np.ndarray) -> np.ndarray:
        """ Returns the diagonal of the kernel k(X, X). """
        return np.array([row.dot(row.T)[0, 0] for row in X])

    def is_stationary(self) -> bool:
        """ Returns whether the kernel is stationary. """
        return False

def matrix_kernel(theta: np.ndarray) -> tuple:
    """ Turns a matrix into "points" and a kernel function. """
    points = np.arange(theta.shape[0], dtype=np.float64).reshape(-1, 1)
    return points, MatrixKernel(theta)

