from typing import Callable
import numpy as np
import scipy.linalg

### helper methods

Kernel = Callable[[np.ndarray, np.ndarray], float]

### estimation methods

def estimate(x_train: np.ndarray, y_train: np.ndarray,
             x_test: np.ndarray, kernel: Kernel) -> np.ndarray:
    """ Estimate y_test with direct Gaussian process regression. """
    # O(n^3)
    K_TT = kernel(x_train, x_train)
    K_PP = kernel(x_test,  x_test )
    K_PT = kernel(x_test,  x_train)
    K_TT_TP = scipy.linalg.solve(K_TT, K_PT.T, assume_a="pos")

    mu_pred = K_TT_TP.T@y_train
    var_pred = K_PP - K_PT@K_TT_TP
    return mu_pred, var_pred

def cknn_selection(x_train: np.ndarray, x_test: np.ndarray,
                   kernel: Kernel, s: int) -> list:
    """ Wrapper over __cknn_selection to pre-process theta ahead of time. """
    n, m = x_train.shape[0], x_test.shape[0]
    K = kernel(*((np.vstack((x_train, x_test)),)*2))
    x_train, x_test = list(range(n)), n + 1
    kernel = lambda i, j: K[i, j]
    # return __cknn_selection(x_train, x_test, kernel, s)
    return __chol_cknn_selection(K, s)

def __naive_cknn_selection(K: np.ndarray, s: int) -> list:
    """ Brute force selection method. """
    # O(s*n*s^3) = O(n s^4)
    n = K.shape[0] - 1
    indexes = []
    for _ in range(s):
        score, best = -1, None
        for i in set(range(n)) - set(indexes):
            new = indexes + [i]
            v = K[new, -1]
            cov = v.T@np.linalg.inv(K[new][:, new])@v
            if cov > score:
                score, best = cov, i
        indexes.append(best)

    return indexes

def __cknn_selection(K: np.ndarray, s: int) -> list:
    """ Select the s most informative entries
        by maximizing Var[E[y_test | y_train]]. """
    # O(s*(n*s + s^2)) = O(n s^2 + s^3)
    n = K.shape[0] - 1
    # initialization
    indexes = []
    inv = np.zeros((0, 0))
    cond_cov = np.array(K[:, n])
    cond_var = np.array(np.diagonal(K))

    for _ in range(s):
        # pick best entry
        k = max(set(range(n)) - set(indexes),
                key=lambda j: cond_cov[j]**2/cond_var[j])
        # update block inverse
        col = K[indexes, k].reshape((-1, 1))
        v = inv@col
        schur = 1/(K[k, k] - col.T.dot(v))
        inv = np.block([[inv + schur*v@v.T, -v*schur],
                        [       -schur*v.T,    schur]])
        indexes.append(k)
        # update Schur complements
        x, x1 = v.T.dot(K[indexes[:-1], n]), K[n, k]
        for j in set(range(n)) - set(indexes):
            y, y1 = v.T.dot(K[indexes[:-1], j]), K[j, k]
            cond_cov[j] -= schur*(x - x1)*(y - y1)
            cond_var[j] -= schur*(y - y1)**2

    return indexes

def __chol_cknn_selection(K: np.ndarray, s: int) -> list:
    """ Select the s most informative entries, storing a Cholesky factor. """
    # O(s*(n*s)) = O(n s^2)
    n = K.shape[0] - 1
    # initialization
    indexes = []
    factors = np.zeros((n + 1, s))
    cond_cov = np.array(K[:, n])
    cond_var = np.array(np.diagonal(K))

    for i in range(s):
        # pick best entry
        k = max(set(range(n)) - set(indexes),
                key=lambda j: cond_cov[j]**2/cond_var[j])
        indexes.append(k)
        # update Cholesky factors
        factors[:, i] = K[:, k]
        factors[:, i] -= factors[:, :i]@factors[k, :i]
        factors[:, i] /= np.sqrt(factors[k, i])
        # update conditional covariance and variance
        for j in range(n):
            cond_cov[j] -= factors[j, i]*factors[n, i]
            cond_var[j] -= factors[j, i]**2

    return indexes

def cknn_estimate(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                  kernel: Kernel, indexes: list) -> np.ndarray:
    """ Estimate y_test according to the given sparisty pattern. """
    return estimate(x_train[indexes], y_train[indexes], x_test, kernel)

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

