import numpy as np
import scipy.linalg
import scipy.spatial.distance
from sklearn.gaussian_process.kernels import Kernel
import ccknn

# TODO: conda venv and update readme

### helper methods

def logdet(m: np.ndarray) -> float:
    """ Computes the logarithm of the determinant of m. """
    return np.product(np.linalg.slogdet(m))

def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ Solve the system Ax = b for symmetric positive definite A. """
    return scipy.linalg.solve(A, b, assume_a="pos")

def inv(m: np.ndarray) -> np.ndarray:
    """ Inverts a symmetric positive definite matrix m. """
    return np.linalg.inv(m)
    # below only starts to get faster for large matrices (~10^4)
    # return solve(m, np.identity(len(m)))

def divide(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """ Return u/v without error messages if there are 0's in v. """
    np.seterr(divide="ignore", invalid="ignore")
    result = u/v
    np.seterr(divide="warn", invalid="warn")
    return result

def argmax(array: np.ndarray, indexes: list, max_: bool=True) -> int:
    """ Return the arg[min/max] of array limited to candidates. """
    arg = np.argmax if max_ else np.argmin
    # ignore indices outside of candidates
    array[indexes] = (-1 if max_ else 1)*np.inf
    # faster than candidates[arg(array[candidates])]
    # likely because it skips the slow indexing of candidates
    return arg(array)

def euclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ Return the Euclidean distance between points in x and y.  """
    # larger kernel value implies more similarity, flip
    return -scipy.spatial.distance.cdist(x, y, "euclidean")

### estimation methods

def __estimate(x_train: np.ndarray, y_train: np.ndarray,
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

def estimate(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
             kernel: Kernel, indexes: list=slice(None)) -> np.ndarray:
    """ Estimate y_test according to the given sparsity pattern. """
    return __estimate(x_train[indexes], y_train[indexes], x_test, kernel)

### selection methods

def knn_select(x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel,
               s: int) -> list:
    """ Select s points in x_train "closest" to x_test by kernel function. """
    # O(n log n)
    # for Matern, sorting by kernel is equivalent to sorting by distance
    # aggregate for multiple prediction points
    dists = np.max(kernel(x_train, x_test), axis=1)
    return np.argsort(dists)[-s:]

def select(x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel,
           s: int, select_method=ccknn.select) -> list:
    """ Wrapper over various cknn selection methods. """
    # early exit
    if s <= 0 or len(x_train) == 0:
        return []
    selected = select_method(x_train, x_test, kernel, s)
    assert len(selected) == len(set(selected)), "selected indices not distinct"
    return selected

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
    points = np.vstack((x_train, x_test))
    # initialization
    indexes, candidates = [], list(range(n))
    prec = np.zeros((0, 0))
    cond_cov = kernel(x_train, x_test).flatten()
    cond_var = kernel.diag(x_train)

    while len(candidates) > 0 and len(indexes) < s:
        # pick best entry
        k = argmax(divide(cond_cov**2, cond_var), indexes)
        indexes.append(k)
        # we don't actually need candidates; faster than candidates.remove(k)
        candidates.pop()
        # update precision of selected entries
        v = prec@kernel(x_train[indexes[:-1]], [x_train[k]])
        var = 1/cond_var[k]
        prec = np.block([[prec + var*v@v.T, -v*var],
                         [        -var*v.T,    var]])
        # compute column k of conditional covariance
        cond_cov_k = kernel(points, [x_train[k]])
        # n.b.: this takes O(ns) space from the kernel function call, however,
        # this can be done in O(1) space by considering each scalar in turn
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
    points = np.vstack((x_train, x_test))
    # initialization
    indexes, candidates = [], list(range(n))
    factors = np.zeros((n + 1, s))
    cond_cov = kernel(x_train, x_test).flatten()
    cond_var = kernel.diag(x_train)

    while len(candidates) > 0 and len(indexes) < s:
        # pick best entry
        k = argmax(divide(cond_cov**2, cond_var), indexes)
        indexes.append(k)
        # we don't actually need candidates; faster than candidates.remove(k)
        candidates.pop()
        # update Cholesky factors
        i = len(indexes) - 1
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
    points = np.vstack((x_train, x_test))
    # initialization
    indexes, candidates = [], list(range(n))
    prec = np.zeros((0, 0))
    prec_pr = inv(kernel(x_test, x_test))
    cond_cov = kernel(x_train, x_test)
    cond_var = kernel.diag(x_train)
    cond_var_pr = cond_var - np.sum((cond_cov@prec_pr)*cond_cov, axis=1)

    while len(candidates) > 0 and len(indexes) < s:
        # pick best entry
        k = argmax(divide(cond_var_pr, cond_var), indexes, max_=False)
        indexes.append(k)
        # we don't actually need candidates; faster than candidates.remove(k)
        candidates.pop()
        # update precision of selected entries
        v = prec@kernel(x_train[indexes[:-1]], [x_train[k]])
        var = 1/cond_var[k]
        prec = np.block([[prec + var*v@v.T, -v*var],
                         [        -var*v.T,    var]])
        # update precision of prediction covariance
        u = prec_pr@cond_cov[k]
        prec_pr += np.outer(u, u)/cond_var_pr[k]
        # compute column k of conditional covariance
        cond_cov_k = kernel(points, [x_train[k]])
        cond_cov_k -= kernel(points, x_train[indexes[:-1]])@v
        cond_cov_k = cond_cov_k.flatten()
        cond_cov_pr_k = np.copy(cond_cov_k[:n])
        cond_cov_pr_k -= cond_cov@u
        cond_cov_k /= np.sqrt(cond_var[k])
        cond_cov_pr_k /= np.sqrt(cond_var_pr[k])
        # update conditional variance and covariance
        cond_var -= cond_cov_k[:n]**2
        cond_var_pr -= cond_cov_pr_k**2
        cond_cov -= np.outer(cond_cov_k[:n], cond_cov_k[n:])

    return indexes

def __chol_update(cov_k: np.ndarray, i: int, k: int, factors: np.ndarray,
                  cond_var: np.ndarray, cond_cov: np.ndarray=None) -> None:
    """ Updates the ith column of the Cholesky factor with column k. """
    n = len(cond_var)
    # update Cholesky factors
    factors[:, i] = cov_k
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
    points = np.vstack((x_train, x_test))
    # initialization
    indexes = np.zeros(min(n, s), dtype=np.int64)
    factors = np.zeros((n, s))
    factors_pr = np.zeros((n + m, s + m))
    cond_var = kernel.diag(x_train)
    cond_var_pr = np.copy(cond_var)
    # pre-condition on the m prediction points
    for i in range(m):
        cov_k = kernel(points, [points[n + i]]).flatten()
        __chol_update(cov_k, i, n + i, factors_pr, cond_var_pr)
    factors_pr = factors_pr[:n]

    i = 0
    while i < len(indexes):
        # pick best entry
        k = argmax(divide(cond_var_pr, cond_var), indexes[:i], max_=False)
        indexes[i] = k
        # update Cholesky factors
        cov_k = kernel(x_train, [x_train[k]]).flatten()
        __chol_update(cov_k, i, k, factors, cond_var)
        __chol_update(cov_k, i + m, k, factors_pr, cond_var_pr)
        i += 1

    return list(indexes)

