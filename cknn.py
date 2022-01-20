import numpy as np
import scipy.linalg
from sklearn.gaussian_process.kernels import Kernel

### helper methods

def logdet(m: np.ndarray) -> float:
    """ Computes the logarithm of the determinant of m. """
    return np.product(np.linalg.slogdet(m))

def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ Solve the system Ax = b for symmetric positive definite A. """
    return scipy.linalg.solve(A, b, assume_a="pos")

def inv(m: np.ndarray) -> np.ndarray:
    """ Inverts a symmetric positive definite matrix m. """
    return solve(m, np.identity(len(m)))

### estimation methods

def estimate(x_train: np.ndarray, y_train: np.ndarray,
             x_test: np.ndarray, kernel: Kernel) -> np.ndarray:
    """ Estimate y_test with direct Gaussian process regression. """
    # O(n^3)
    K_TT = kernel(x_train)
    K_PP = kernel(x_test)
    K_TP = kernel(x_train, x_test)
    K_PT_TT = solve(K_TT, K_TP).T

    mu_pred = K_PT_TT@y_train
    var_pred = K_PP - K_PT_TT@K_TP
    return mu_pred, var_pred

def cknn_estimate(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                  kernel: Kernel, indexes: list) -> np.ndarray:
    """ Estimate y_test according to the given sparsity pattern. """
    return estimate(x_train[indexes], y_train[indexes], x_test, kernel)

### selection methods

def cknn_select(x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel,
                s: int) -> list:
    """ Wrapper over various cknn selection methods. """
    return __prec_mult_select(x_train, x_test, kernel, s)

def __naive_select(x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel,
                   s: int) -> list:
    """ Brute force selection method. """
    # O(s*n*s^3) = O(n s^4)
    n = len(x_train)
    indexes, candidates = [], list(range(n))

    while len(candidates) > 0 and len(indexes) < s:
        score, best = float("inf"), None
        for i in candidates:
            new = indexes + [i]
            v = kernel(x_train[new], x_test)
            cov = kernel(x_test) - v.T@inv(kernel(x_train[new]))@v
            if cov < score:
                score, best = cov, i
        indexes.append(best)
        candidates.remove(best)

    return indexes

def __prec_select(x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel,
                  s: int) -> list:
    """ Greedily select the s entries minimizing conditional covariance. """
    # O(s*(n*s + s^2)) = O(n s^2)
    n = len(x_train)
    # initialization
    indexes, candidates = [], list(range(n))
    prec = np.zeros((0, 0))
    cond_cov = kernel(x_train, x_test).flatten()
    cond_var = kernel.diag(x_train)

    while len(candidates) > 0 and len(indexes) < s:
        # pick best entry
        k = max(candidates, key=lambda j: cond_cov[j]**2/cond_var[j])
        indexes.append(k)
        candidates.remove(k)
        # update precision of selected entries
        v = prec@kernel(x_train[indexes[:-1]], [x_train[k]])
        var = 1/cond_var[k]
        prec = np.block([[prec + var*v@v.T, -v*var],
                         [        -var*v.T,    var]])
        # compute column k of conditional covariance
        points = np.vstack((x_train, x_test))
        cond_cov_k = kernel(points, [x_train[k]])
        cond_cov_k -= kernel(points, x_train[indexes[:-1]])@v
        cond_cov_k = cond_cov_k.flatten()/np.sqrt(cond_var[k])
        # update conditional variance and covariance
        cond_var -= cond_cov_k[:n]**2
        cond_cov -= cond_cov_k[:n]*cond_cov_k[n]

    return indexes

def __chol_select(x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel,
                  s: int) -> list:
    """ Select the s most informative entries, storing a Cholesky factor. """
    # O(s*(n*s)) = O(n s^2)
    n = len(x_train)
    # initialization
    indexes, candidates = [], list(range(n))
    factors = np.zeros((n + 1, s))
    cond_cov = kernel(x_train, x_test).flatten()
    cond_var = kernel.diag(x_train)

    while len(candidates) > 0 and len(indexes) < s:
        # pick best entry
        k = max(candidates, key=lambda j: cond_cov[j]**2/cond_var[j])
        indexes.append(k)
        candidates.remove(k)
        # update Cholesky factors
        i, points = len(indexes) - 1, np.vstack((x_train, x_test))
        factors[:, i] = kernel(points, [x_train[k]]).flatten()
        factors[:, i] -= factors[:, :i]@factors[k, :i]
        factors[:, i] /= np.sqrt(factors[k, i])
        # update conditional variance and covariance
        cond_var -= factors[:, i][:n]**2
        cond_cov -= factors[:, i][:n]*factors[n, i]

    return indexes

### multiple point selection 

def __naive_mult_select(x_train: np.ndarray, x_test: np.ndarray,
                        kernel: Kernel, s: int) -> list:
    """ Brute force multiple point selection method. """
    # O(s*n*(s^3 + m^3)) = O(n s^4 + n s m^3)
    n = len(x_train)
    indexes, candidates = [], list(range(n))

    while len(candidates) > 0 and len(indexes) < s:
        score, best = float("inf"), None
        for i in candidates:
            new = indexes + [i]
            v = kernel(x_train[new], x_test)
            cov = logdet(kernel(x_test) - v.T@inv(kernel(x_train[new]))@v)
            if cov < score:
                score, best = cov, i
        indexes.append(best)
        candidates.remove(best)

    return indexes

def __prec_mult_select(x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel,
                      s: int) -> list:
    """ Greedily select the s entries minimizing conditional covariance. """
    # O(m^3 + n*m^2 + s*(s^2 + m^2 + n*s + n*m + n*m)) = O(n s^2 + n m^2 + m^3)
    n, m = len(x_train), len(x_test)
    # initialization
    indexes, candidates = [], list(range(n))
    prec = np.zeros((0, 0))
    prec_pr = np.linalg.inv(kernel(x_test, x_test))
    cond_cov = kernel(x_train, x_test)
    cond_var = kernel.diag(x_train)
    cond_var_pr = cond_var - np.sum((cond_cov@prec_pr)*cond_cov, axis=1)

    while len(candidates) > 0 and len(indexes) < s:
        # pick best entry
        k = min(candidates, key=lambda j: cond_var_pr[j]/cond_var[j])
        indexes.append(k)
        candidates.remove(k)
        # update precision of selected entries
        v = prec@kernel(x_train[indexes[:-1]], [x_train[k]])
        var = 1/cond_var[k]
        prec = np.block([[prec + var*v@v.T, -v*var],
                         [        -var*v.T,    var]])
        # update precision of prediction covariance
        u = prec_pr@cond_cov[k]
        prec_pr += np.outer(u, u)/cond_var_pr[k]
        # compute column k of conditional covariance
        points = np.vstack((x_train, x_test))
        cond_cov_k = kernel(points, [x_train[k]])
        cond_cov_k -= kernel(points, x_train[indexes[:-1]])@v
        cond_cov_k = cond_cov_k.flatten()
        cond_cov_pr_k = np.array(cond_cov_k[:n])
        cond_cov_pr_k -= cond_cov@u
        cond_cov_k /= np.sqrt(cond_var[k])
        cond_cov_pr_k /= np.sqrt(cond_var_pr[k])
        # update conditional variance and covariance
        cond_var -= cond_cov_k[:n]**2
        cond_var_pr -= cond_cov_pr_k**2
        cond_cov -= np.outer(cond_cov_k[:n], cond_cov_k[n:])

    return indexes

def __chol_update(x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel,
                  i: int, k: int, factors: np.ndarray,
                  cond_var: np.ndarray, cond_cov: np.ndarray=None) -> None:
    """ Updates the ith column of the Cholesky factor with column k. """
    n = len(x_train)
    # update Cholesky factors
    points = np.vstack((x_train, x_test))
    factors[:, i] = kernel(points, [points[k]]).flatten()
    factors[:, i] -= factors[:, :i]@factors[k, :i]
    factors[:, i] /= np.sqrt(factors[k, i])
    # update conditional variance and covariance
    cond_var -= factors[:, i][:n]**2
    if cond_cov is not None:
        cond_cov -= np.outer(factors[:, i][:n], factors[:, i][n:])

def __chol_mult_select(x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel,
                       s: int) -> list:
    """ Greedily select the s entries minimizing conditional covariance. """
    # O(m*(n + m)*m + s*(n + m)*(s + m)) = O(n s^2 + n m^2 + m^3)
    n, m = len(x_train), len(x_test)
    args = (x_train, x_test, kernel)
    # initialization
    indexes, candidates = [], list(range(n))
    factors = np.zeros((n + m, s))
    cond_var = kernel.diag(x_train)
    # pre-condition on the m prediction points
    factors_pr = np.zeros((n + m, s + m))
    cond_var_pr = np.array(cond_var)
    for i in range(m):
        __chol_update(*args, i, n + i, factors_pr, cond_var_pr)

    while len(candidates) > 0 and len(indexes) < s:
        # pick best entry
        k = min(candidates, key=lambda j: cond_var_pr[j]/cond_var[j])
        indexes.append(k)
        candidates.remove(k)
        # update Cholesky factors
        i = len(indexes) - 1
        __chol_update(*args, i, k, factors, cond_var)
        __chol_update(*args, i + m, k, factors_pr, cond_var_pr)

    return indexes

