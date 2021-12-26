from typing import Callable
import numpy as np
import scipy.linalg

Kernel = Callable[[np.ndarray, np.ndarray], np.ndarray]

### helper methods

def logdet(m: np.ndarray) -> float:
    """ Computes the logarithm of the determinant of m. """
    return np.product(np.linalg.slogdet(m))

### estimation methods

def estimate(x_train: np.ndarray, y_train: np.ndarray,
             x_test: np.ndarray, kernel: Kernel) -> np.ndarray:
    """ Estimate y_test with direct Gaussian process regression. """
    # O(n^3)
    K_TT = kernel(x_train, x_train)
    K_PP = kernel(x_test,  x_test )
    K_TP = kernel(x_train,  x_test)
    K_PT_TT = scipy.linalg.solve(K_TT, K_TP, assume_a="pos").T

    mu_pred = K_PT_TT@y_train
    var_pred = K_PP - K_PT_TT@K_TP
    return mu_pred, var_pred

def cknn_estimate(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                  kernel: Kernel, indexes: list) -> np.ndarray:
    """ Estimate y_test according to the given sparsity pattern. """
    return estimate(x_train[indexes], y_train[indexes], x_test, kernel)

### selection methods

def cknn_selection(x_train: np.ndarray, x_test: np.ndarray,
                   kernel: Kernel, s: int) -> list:
    """ Wrapper over __cknn_selection to pre-process theta ahead of time. """
    n, m = x_train.shape[0], x_test.shape[0]
    K = kernel(*((np.vstack((x_train, x_test)),)*2))
    x_train, x_test = list(range(n)), n + 1
    kernel = lambda i, j: K[i, j]
    # return __cknn_selection(x_train, x_test, kernel, s)
    # return __chol_cknn_selection(K, s)
    return __cknn_mult_selection(K, m, s)

def __naive_cknn_selection(K: np.ndarray, s: int) -> list:
    """ Brute force selection method. """
    # O(s*n*s^3) = O(n s^4)
    n = K.shape[0] - 1
    indexes = []

    for _ in range(s):
        score, best = K[-1, -1], None
        for i in set(range(n)) - set(indexes):
            new = indexes + [i]
            v = K[new, -1]
            cov = K[-1, -1] - v.T@np.linalg.inv(K[new][:, new])@v
            if cov < score:
                score, best = cov, i
        indexes.append(best)

    return indexes

def __cknn_selection(K: np.ndarray, s: int) -> list:
    """ Greedily select the s entries minimizing conditional covariance. """
    # O(s*(n*s + s^2)) = O(n s^2)
    n = K.shape[0] - 1
    # initialization
    indexes, candidates = [], list(range(n))
    inv = np.zeros((0, 0))
    cond_cov = np.array(K[:n, n])
    cond_var = np.array(np.diagonal(K[:n]))

    for _ in range(s):
        # pick best entry
        k = max(candidates, key=lambda j: cond_cov[j]**2/cond_var[j])
        indexes.append(k)
        candidates.remove(k)
        # update block inverse
        v = (inv@K[indexes[:-1], k]).reshape((-1, 1))
        var = 1/cond_var[k]
        inv = np.block([[inv + var*v@v.T, -v*var],
                        [     -var*v.T,      var]])
        # update conditional covariance and variance
        cond_cov_k = np.array(K[:, k])
        cond_cov_k -= K[:, indexes[:-1]]@v.flatten()
        cond_cov_k /= np.sqrt(cond_var[k])
        for j in candidates:
            cond_cov[j] -= cond_cov_k[j]*cond_cov_k[n]
            cond_var[j] -= cond_cov_k[j]**2

    return indexes

def __chol_cknn_selection(K: np.ndarray, s: int) -> list:
    """ Select the s most informative entries, storing a Cholesky factor. """
    # O(s*(n*s)) = O(n s^2)
    n = K.shape[0] - 1
    # initialization
    indexes, candidates = [], list(range(n))
    factors = np.zeros((n + 1, s))
    cond_cov = np.array(K[:n, n])
    cond_var = np.array(np.diagonal(K[:n]))

    for i in range(s):
        # pick best entry
        k = max(candidates, key=lambda j: cond_cov[j]**2/cond_var[j])
        indexes.append(k)
        candidates.remove(k)
        # update Cholesky factors
        factors[:, i] = K[:, k]
        factors[:, i] -= factors[:, :i]@factors[k, :i]
        factors[:, i] /= np.sqrt(factors[k, i])
        # update conditional covariance and variance
        for j in candidates:
            cond_cov[j] -= factors[j, i]*factors[n, i]
            cond_var[j] -= factors[j, i]**2

    return indexes

### multiple point selection 

def __naive_cknn_mult_selection(K: np.ndarray, m: int, s: int) -> list:
    """ Brute force multiple point selection method. """
    # O(s*n*(s^3 + m^3)) = O(n s^4 + n s m^3)
    n = K.shape[0] - m
    indexes = []

    for _ in range(s):
        score, best = logdet(K[n:, n:]), None
        for i in set(range(n)) - set(indexes):
            new = indexes + [i]
            v = K[new, n:]
            cov = logdet(K[n:, n:] - v.T@np.linalg.inv(K[new][:, new])@v)
            if cov < score:
                score, best = cov, i
        indexes.append(best)

    return indexes

def __cknn_mult_selection(K: np.ndarray, m: int, s: int) -> list:
    """ Greedily select the s entries minimizing conditional covariance. """
    # O((n + m) m^2 + s*(n + m)*(s + m)) = O(n s^2 + n m^2 + m^3)
    n = K.shape[0] - m
    # initialization
    indexes, candidates = [], list(range(n))
    inv = np.zeros((0, 0))
    inv_pr = np.linalg.inv(K[n:, n:])
    cond_cov = np.array(K[:n, n:])
    cond_var = np.array(np.diagonal(K[:n]))
    cond_var_pr = cond_var - np.sum((cond_cov@inv_pr)*cond_cov, axis=1)

    for _ in range(s):
        # pick best entry
        k = min(candidates, key=lambda j: cond_var_pr[j]/cond_var[j])
        indexes.append(k)
        candidates.remove(k)
        # update block inverse
        v = (inv@K[indexes[:-1], k]).reshape((-1, 1))
        var = 1/cond_var[k]
        inv = np.block([[inv + var*v@v.T, -v*var],
                        [     -var*v.T,      var]])
        # update inverse of prediction covariance
        u = inv_pr@cond_cov[k]
        inv_pr = inv_pr + np.outer(u, u)/cond_var_pr[k]
        # update conditional covariance and variance
        cond_cov_k = np.array(K[:, k])
        cond_cov_k -= K[:, indexes[:-1]]@v.flatten()
        cond_cov_pr_k = np.array(cond_cov_k[:n])
        cond_cov_pr_k -= cond_cov@u
        cond_cov_k /= np.sqrt(cond_var[k])
        cond_cov_pr_k /= np.sqrt(cond_var_pr[k])
        for j in candidates:
            cond_var[j] -= cond_cov_k[j]**2
            cond_var_pr[j] -= cond_cov_pr_k[j]**2
            for c in range(m):
                cond_cov[j, c] -= cond_cov_k[j]*cond_cov_k[n + c]

    return indexes

def __chol_update(K: np.ndarray, i: int, k: int, factors: np.ndarray,
                  cond_var: np.ndarray, cond_cov: np.ndarray=None) -> None:
    """ Updates the ith column of the Cholesky factor with column k. """
    # update Cholesky factors
    factors[:, i] = K[:, k]
    factors[:, i] -= factors[:, :i]@factors[k, :i]
    factors[:, i] /= np.sqrt(factors[k, i])
    # update conditional variance and covariance
    for j in range(len(cond_var)):
        cond_var[j] -= factors[j, i]**2
        if cond_cov is not None:
            cond_cov[j] -= factors[j, i]*factors[-1, i]

def __chol_cknn_mult_selection(K: np.ndarray, m: int, s: int) -> list:
    """ Greedily select the s entries minimizing conditional covariance. """
    # O((n + m) m^2 + s*(n + m)*(s + m)) = O(n s^2 + n m^2 + m^3)
    n = K.shape[0] - m
    # initialization
    indexes, candidates = [], list(range(n))
    factors = np.zeros((n + m, s))
    cond_var = np.array(np.diagonal(K[:n]))
    # pre-condition on the m prediction points
    factors_pr = np.zeros((n + m, s + m))
    cond_var_pr = np.array(cond_var)
    for i in range(m):
        __chol_update(K, i, n + i, factors_pr, cond_var_pr)

    for i in range(s):
        # pick best entry
        k = min(candidates, key=lambda j: cond_var_pr[j]/cond_var[j])
        indexes.append(k)
        candidates.remove(k)
        # update Cholesky factors
        __chol_update(K, i, k, factors, cond_var)
        __chol_update(K, i + m, k, factors_pr, cond_var_pr)

    return indexes

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

