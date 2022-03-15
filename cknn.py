import numpy as np
import scipy.linalg
import scipy.spatial.distance
from sklearn.gaussian_process.kernels import Kernel, Matern
import ccknn

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
    # return solve(m, np.identity(m.shape[0]))

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

def __naive_select(x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel,
                   s: int) -> list:
    """ Brute force selection method. """
    # O(s*n*s^3) = O(n s^4)
    n = x_train.shape[0]
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
    n = x_train.shape[0]
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
    n = x_train.shape[0]
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
    n = x_train.shape[0]
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
    n, m = x_train.shape[0], x_test.shape[0]
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
    n = cond_var.shape[0]
    # update Cholesky factors
    factors[:, i] = cov_k
    factors[:, i] -= factors[:, :i]@factors[k, :i]
    factors[:, i] /= np.sqrt(factors[k, i])
    # update conditional variance and covariance
    cond_var -= factors[:, i][:n]**2
    if cond_cov is not None:
        cond_cov -= np.outer(factors[:, i][:n], factors[:, i][n:])

def __chol_mult_select(x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel,
                       s: int) -> np.ndarray:
    """ Greedily select the s entries minimizing conditional covariance. """
    # O(m*(n + m)*m + s*(n + m)*(s + m)) = O(n s^2 + n m^2 + m^3)
    n, m = x_train.shape[0], x_test.shape[0]
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
    while i < indexes.shape[0]:
        # pick best entry
        k = argmax(divide(cond_var_pr, cond_var), indexes[:i], max_=False)
        indexes[i] = k
        # update Cholesky factors
        cov_k = kernel(x_train, [x_train[k]]).flatten()
        __chol_update(cov_k, i, k, factors, cond_var)
        __chol_update(cov_k, i + m, k, factors_pr, cond_var_pr)
        i += 1

    return indexes

### non-adjacent multiple point selection

def __chol_insert(cov_k: np.ndarray, order: np.ndarray, i: int,
                  index: int, k: int, factors: np.ndarray) -> None:
    """ Updates the ith column of the Cholesky factor with column k. """
    # move columns over to make space at index
    for col in range(i, index, -1):
        factors[:, col] = factors[:, col - 1]

    # insert conditional covariance with k into Cholesky factor at index
    factors[:, index] = cov_k
    factors[:, index] -= factors[:, :index]@factors[k, :index]
    factors[:, index] /= np.sqrt(factors[k, index])

    # update downstream Cholesky factor by rank-one downdate
    cov_k = np.copy(factors[:, index])
    for col in range(index + 1, i + 1):
        k = order[col]
        c1, c2 = factors[k, col], cov_k[k]
        dp = np.sqrt(c1*c1 - c2*c2)
        c1, c2 = c1/dp, c2/dp
        factors[:, col] *= c1
        factors[:, col] += -c2*cov_k
        cov_k *= 1/c1
        cov_k += -c2/c1*factors[:, col]

def __insert_index(order: np.ndarray, locations: np.ndarray,
                   i: int, k: int) -> int:
    """ Finds the index to insert index k into the order. """
    index = -1
    for index in range(i):
        # bigger than current value, insertion spot
        if locations[k] >= locations[order[index]]:
            return index
    return index + 1

def __select_point(order: np.ndarray, locations: np.ndarray, i: int, k: int,
                   points: np.ndarray, kernel: Kernel, factors: np.ndarray):
    """ Add the kth point to the Cholesky factor. """
    index = __insert_index(order, locations, i, k)
    # shift values over to make room for k at index
    for col in range(i, index, -1):
        order[col] = order[col - 1]
    order[index] = k
    # update Cholesky factor
    cov_k = kernel(points, [points[k]]).flatten()
    __chol_insert(cov_k, order, i, index, k, factors)

def __chol_nonadj_select(x: np.ndarray, train: np.ndarray, test: np.ndarray,
                         kernel: Kernel, s: int) -> np.ndarray:
    """ Greedily select the s entries minimizing conditional covariance. """
    # O(m*(n + m)*m + s*(n + m)*(s + m)) = O(n s^2 + n m^2 + m^3)
    n, m = train.shape[0], test.shape[0]
    locations = np.append(train, test)
    points = x[locations]
    # initialization
    indexes = np.zeros(min(n, s), dtype=np.int64)
    order = np.zeros(indexes.shape[0] + test.shape[0], dtype=np.int64)
    factors = np.zeros((n + m, s + m))
    var = kernel.diag(x[train])
    # pre-condition on the m prediction points
    for i in range(m):
        __select_point(order, locations, i, n + i,
                       points, kernel, factors)
    i = 0
    while i < indexes.shape[0]:
        # pick best entry
        best, best_k = np.inf, 0
        for j in range(n):
            # selected already, don't consider as candidate
            if var[j] == np.inf:
                continue

            cond_var_j = var[j]
            index = __insert_index(order, locations, m + i, j)
            key = -np.log(cond_var_j) if index == 0 else 0

            for col in range(m + i):
                k = order[col]
                cond_var_k = factors[k, col]
                cond_cov_k = factors[j, col]*cond_var_k
                cond_var_k *= cond_var_k
                cond_cov_k *= cond_cov_k
                # remove spurious contribution from selected training point
                if k < n:
                    key -= np.log(cond_var_k - (cond_cov_k/cond_var_j
                                                if col >= index else 0))
                cond_var_j -= cond_cov_k/cond_var_k
                # remove spurious contribution of j
                if col + 1 == index:
                    key -= np.log(cond_var_j)
            # add logdet of entire covariance matrix
            key += np.log(cond_var_j)

            if key < best:
                best, best_k = key, j

        indexes[i] = best_k
        # mark as selected
        var[best_k] = np.inf
        # update Cholesky factor
        __select_point(order, locations, i + m, best_k,
                       points, kernel, factors)
        i += 1

    return indexes

### high-level selection methods

def random_select(x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel,
                  s: int, seed: int=1) -> np.ndarray:
    """ Randomly select s points out of x_train. """
    s = np.clip(s, 0, x_train.shape[0])
    rng = np.random.default_rng(seed)
    return rng.choice(x_train.shape[0], s, replace=False)

def corr_select(x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel,
                s: int) -> np.ndarray:
    """ Select s points in x_train with highest correlation to x_test. """
    # O(n log n)
    s = np.clip(s, 0, x_train.shape[0])
    # aggregate for multiple prediction points
    corrs = np.max(kernel(x_train, x_test)**2, axis=1)/kernel.diag(x_train)
    return np.argsort(corrs)[x_train.shape[0] - s:]

def knn_select(x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel,
               s: int) -> np.ndarray:
    """ Select s points in x_train "closest" to x_test by kernel function. """
    # O(n log n)
    s = np.clip(s, 0, x_train.shape[0])
    # for Matern, sorting by kernel is equivalent to sorting by distance
    # aggregate for multiple prediction points
    dists = np.max(kernel(x_train, x_test), axis=1)
    return np.argsort(dists)[x_train.shape[0] - s:]

def select(x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel,
           s: int, select_method=ccknn.select) -> np.ndarray:
    """ Wrapper over various cknn selection methods. """
    # early exit
    s = np.clip(s, 0, x_train.shape[0])
    if s == 0:
        return []
    selected = select_method(x_train, x_test, kernel, s)
    assert len(set(selected)) == s, "selected indices not distinct"
    return selected

def nonadj_select(x: np.ndarray,
                  train: np.ndarray, test: np.ndarray, kernel: Kernel,
                  s: int, select_method=ccknn.nonadj_select) -> np.ndarray:
    """ Wrapper over various cknn selection methods. """
    # early exit
    s = np.clip(s, 0, train.shape[0])
    if s == 0:
        return []
    selected = select_method(x, train, test, kernel, s)
    assert len(set(selected)) == s, "selected indices not distinct"
    return selected

