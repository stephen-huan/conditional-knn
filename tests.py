import time
import numpy as np
import scipy.spatial
import cknn

D = 3    # dimension of points
N = 100  # number of training points
M = 1    # number of prediction points 
S = 20   # number of entries to pick

# display settings
np.set_printoptions(precision=3, suppress=True)

# set random seed
rng = np.random.default_rng(1)

### covariance functions

Kernel = cknn.Kernel

def matern12(l: float) -> Kernel:
    def k(u: np.ndarray, v: np.ndarray) -> float:
        """ Matern kernel function with v = 1/2 and with length scale l. """
        r = scipy.spatial.distance.cdist(u, v, "euclidean")
        return np.exp(-r/l)
    return k

def matern32(l: float) -> Kernel:
    def k(u: np.ndarray, v: np.ndarray) -> float:
        """ Matern kernel function with v = 3/2 and with length scale l. """
        r = scipy.spatial.distance.cdist(u, v, "euclidean")
        return (1 + np.sqrt(3)*r/l)*np.exp(-np.sqrt(3)*r/l)
    return k

def matern52(l: float) -> Kernel:
    def k(u: np.ndarray, v: np.ndarray) -> float:
        """ Matern kernel function with v = 5/2 and with length scale l. """
        r = scipy.spatial.distance.cdist(u, v, "euclidean")
        return (1 + np.sqrt(5)*r/l + 5*r**2/(3*l**2))*np.exp(-np.sqrt(5)*r/l)
    return k

if __name__ == "__main__":
    # generate input data and corresponding output
    # data matrix is each *row* is point, gram matrix X X^T
    X = rng.random((N, D))
    y = np.zeros(N)
    w = np.array([2, -1, -1])
    f = lambda x: w.dot(x.T)
    for i in range(N):
        y[i] = f(X[i])

    x_test = rng.random((M, D))
    y_test = np.zeros(M)
    for i in range(M):
        y_test[i] = f(x_test[i])

    # predictions
    kernel = matern52(0.1)

    mu_pred, var_pred = cknn.estimate(X, y, x_test, kernel)
    print(y_test, mu_pred)
    print(var_pred)

    indexes = cknn.cknn_selection(X, x_test, kernel, S)
    K = kernel(*((np.vstack((X, x_test)),)*2))
    assert indexes == cknn.__cknn_selection(K, S), "indexes mismatch"

    mu_pred, var_pred = cknn.cknn_estimate(X, y, x_test, kernel, indexes)
    print(y_test, mu_pred)
    print(var_pred)

