import time
import numpy as np
import cknn

d = 3    # dimension of points
N = 100  # number of points
s = 20   # number of entries to pick

# display settings
np.set_printoptions(precision=3, suppress=True)

# set random seed
rng = np.random.default_rng(1)

### covariance functions

Kernel = cknn.Kernel

def matern12(l: float) -> Kernel:
    def k(u: np.ndarray, v: np.ndarray) -> float:
        """ Matern Kernel function with v = 1/2 and with length scale l. """
        r = np.linalg.norm(u - v)
        return np.exp(-r/l)
    return k

def matern32(l: float) -> Kernel:
    def k(u: np.ndarray, v: np.ndarray) -> float:
        """ Matern Kernel function with v = 3/2 and with length scale l. """
        r = np.linalg.norm(u - v)
        return (1 + np.sqrt(3)*r/l)*np.exp(-np.sqrt(3)*r/l)
    return k

def matern52(l: float) -> Kernel:
    def k(u: np.ndarray, v: np.ndarray) -> float:
        """ Matern Kernel function with v = 5/2 and with length scale l. """
        r = np.linalg.norm(u - v)
        return (1 + np.sqrt(5)*r/l + 5*r**2/(3*l**2))*np.exp(-np.sqrt(5)*r/l)
    return k

if __name__ == "__main__":
    # generate input data and corresponding output
    # data matrix is each *row* is point, gram matrix X X^T
    X = rng.random((N, d))
    y = np.zeros(N)
    w = np.array([2, -1, -1])
    for i in range(N):
        y[i] = w.T@X[i]

    x_test = rng.random((d, 1))
    y_test = w.T@x_test

    # predictions
    kernel = matern52(0.1)

    y_pred = cknn.estimate(X, y, x_test, kernel)
    print(y_test, y_pred)

    indexes = cknn.cknn_selection(X, x_test, kernel, s)
    K = cknn.covariance_matrix(np.vstack((X, x_test.T)), kernel)
    assert indexes == cknn.__cknn_selection(K, s), "indexes mismatch"

    y_pred = cknn.cknn_estimate(X, y, x_test, kernel, indexes)
    print(y_test, y_pred)

