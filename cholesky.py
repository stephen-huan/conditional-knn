import numpy as np
import scipy.linalg
import scipy.sparse as sparse
import cknn
from cknn import logdet, inv, solve, Kernel
import ordering

class MatrixKernel(Kernel):
    """
    A wrapper over sklearn.gaussian_process.kernels.Kernel for matrices.
    """

    def __init__(self, m: np.ndarray) -> None:
        self.m = m

    def __flatten(self, m: np.ndarray) -> np.ndarray:
        """ Flatten m for use in indexing. """
        return np.array(m).flatten()

    def __call__(self, X: np.ndarray, Y: np.ndarray=None,
                 eval_gradient: bool=False) -> np.ndarray:
        """ Return the kernel k(X, Y) and possibly its gradient. """
        if Y is None: Y = X
        return self.m[self.__flatten(X)][:, self.__flatten(Y)]

    def diag(self, X: np.ndarray) -> np.ndarray:
        """ Returns the diagonal of the kernel k(X, X). """
        return np.array(np.diag(self.m))[self.__flatten(X)]

    def is_stationary(self) -> bool:
        """ Returns whether the kernel is stationary. """
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(m={repr(self.m)})"

def matrix_kernel(theta: np.ndarray) -> tuple:
    """ Turns a matrix into "points" and a kernel function. """
    return np.arange(len(theta)).reshape(-1, 1), MatrixKernel(theta)

### helper methods

def kl_div(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes the KL divergence between the multivariate Gaussians
    at 0 with covariance X and Y, i.e. D_KL(N(0, X) || N(0, Y)).
    """
    # O(n^3)
    return np.trace(solve(Y, X)) + logdet(Y) - logdet(X) - len(X)

def sparse_kl_div(L: sparse.csc.csc_matrix) -> float:
    """
    Computes the KL divergence assuming L is optimal.

    Equivalent to kl_div(theta, inv(L@L.T)) + logdet(theta)
    """
    # O(n)
    return -2*np.sum(np.log(L.diagonal()))

### sparse Cholesky methods

def __col(x: np.ndarray, kernel: Kernel, s: list) -> np.ndarray:
    """ Computes a single column of the sparse Cholesky factor. """
    # O(s^3)
    m = inv(kernel(x[s]))
    return m[:, 0]/np.sqrt(m[0, 0])

def __cholesky(x: np.ndarray, kernel: Kernel,
               sparsity: dict) -> sparse.csc.csc_matrix:
    """ Computes the best Cholesky factor following the sparsity pattern. """
    # O(n s^3)
    n = len(x)
    indptr = np.cumsum([0] + [len(sparsity[i]) for i in range(n)])
    data, indexes = np.zeros(indptr[-1]), np.zeros(indptr[-1])
    for i in range(n):
        # make sure diagonal entry is first in the sparsity pattern
        s = sorted(sparsity[i])
        data[indptr[i]: indptr[i + 1]] = __col(x, kernel, s)
        indexes[indptr[i]: indptr[i + 1]] = s

    return sparse.csc_matrix((data, indexes, indptr), shape=(n, n))

def __cols(x: np.ndarray, kernel: Kernel, s: list) -> np.ndarray:
    """ Computes multiple columns of the sparse Cholesky factor. """
    # O(s^3)
    # equivalent to inv(chol(inv(kernel(x[s]))))
    L = np.flip(np.linalg.cholesky(np.flip(kernel(x[s])))).T
    return L

def __mult_cholesky(x: np.ndarray, kernel: Kernel,
                    sparsity: dict, groups: list) -> sparse.csc.csc_matrix:
    """ Computes the best Cholesky factor following the sparsity pattern. """
    # O((n/m)*(s^3 + m*s^2)) = O((n s^3)/m)
    n = len(x)
    indptr = np.cumsum([0] + [len(sparsity[i]) for i in range(n)])
    data, indexes = np.zeros(indptr[-1]), np.zeros(indptr[-1])
    for group in groups:
        # points only interact with points after them (lower triangularity)
        s = sorted(sparsity[min(group)])
        positions = {i: k for k, i in enumerate(s)}
        L = __cols(x, kernel, s)
        for i in group:
            k = positions[i]
            e_k = np.zeros(len(s))
            e_k[k] = 1
            col = scipy.linalg.solve_triangular(L, e_k, lower=True)
            data[indptr[i]: indptr[i + 1]] = col[k:]
            indexes[indptr[i]: indptr[i + 1]] = s[k:]

    return sparse.csc_matrix((data, indexes, indptr), shape=(n, n))

def naive_cholesky(theta: np.ndarray, s: int) -> np.ndarray:
    """ Computes Cholesky with at most s nonzero entries per column. """
    # O(n*s*n*s^3) = O(n^2 s^4)
    n = len(theta)
    sparsity = {}
    points, kernel = matrix_kernel(theta)
    for i in range(n):
        # start with diagonal entry
        indexes, candidates = [i], list(range(i + 1, n))
        # add according to best KL divergence
        while len(candidates) > 0 and len(indexes) < s:
            k = max(candidates,
                    key=lambda j: __col(points, kernel, indexes + [j])[0])
            indexes.append(k)
            candidates.remove(k)
        sparsity[i] = indexes

    return __cholesky(points, kernel, sparsity)

def naive_mult_cholesky(theta: np.ndarray, s: int,
                        groups: list=None) -> np.ndarray:
    """ Computes Cholesky with at most s nonzero entries per column. """
    # O((n/m)*(s - m)*n*m*s^3 + n s^3) = O(n^2 s^3 (s - m) + n s^3)
    n = len(theta)
    sparsity = {}
    points, kernel = matrix_kernel(theta)
    # each column in its own group
    if groups is None: groups = [[i] for i in range(n)]
    for group in groups:
        indexes, candidates = [], list(range(max(group) + 1, n))
        # start with diagonal entries
        group.sort()
        for k, i in enumerate(group):
            indexes.append(group[k:])
        # add according to best KL divergence
        while len(candidates) > 0 and len(indexes[0]) < s:
            k = max(candidates, key=lambda j:
                    sum(np.log(__col(points, kernel, indexes[i] + [j])[0])
                        for i in range(len(group))))
            for col in indexes:
                col.append(k)
            candidates.remove(k)
        for k, i in enumerate(group):
            sparsity[i] = indexes[k]

    return __cholesky(points, kernel, sparsity)

def cholesky_select(x: np.ndarray, kernel: Kernel,
                    s: int, groups: list=None) -> np.ndarray:
    """ Computes Cholesky with at most s nonzero entries per column. """
    # without aggregation: O(n*(n s^2) + n s^3) = O(n^2 s^2)
    # with    aggregation: O((n/m)*(n (s - m)^2 + n m^2 + m^3) + (n s^3)/m)
    # = O((n^2 s^2)/m)
    n = len(x)
    sparsity = {}
    # each column in its own group
    if groups is None: groups = [[i] for i in range(n)]
    for group in groups:
        candidates = np.arange(max(group) + 1, n)
        selected = cknn.select(x[candidates], x[group], kernel, s - len(group))
        indexes = list(candidates[selected])
        # could save memory by only storing for the minimum index
        for i in sorted(group, reverse=True):
            indexes.append(i)
            # put diagonal entry first
            sparsity[i] = indexes[::-1]

    return __mult_cholesky(x, kernel, sparsity, groups)

def naive_nonadj_cholesky(theta: np.ndarray, s: int,
                          groups: list=None) -> np.ndarray:
    """ Computes Cholesky with at most s nonzero entries per column. """
    # O((n/m)*(s - m)*n*m*s^3 + n s^3) = O(n^2 s^3 (s - m) + n s^3)
    n = len(theta)
    sparsity = {}
    points, kernel = matrix_kernel(theta)
    # each column in its own group
    if groups is None: groups = [[i] for i in range(n)]
    for group in groups:
        indexes, candidates = [], list(set(range(n)) - set(group))
        # start with diagonal entries
        group.sort()
        for k, i in enumerate(group):
            indexes.append(group[k:])
        # add according to best KL divergence
        while len(candidates) > 0 and len(indexes[0]) < s:
            k = max(candidates, key=lambda j:
                    sum(np.log(__col(points, kernel, indexes[i] +
                                     ([j] if j >= group[i] else []))[0])
                        for i in range(len(group))))
            for i, col in enumerate(indexes):
                if k >= group[i]:
                    col.append(k)
            candidates.remove(k)
        for k, i in enumerate(group):
            sparsity[i] = indexes[k]

    return __cholesky(points, kernel, sparsity)

def cholesky_nonadj_select(x: np.ndarray, kernel: Kernel,
                    s: int, groups: list=None) -> np.ndarray:
    """ Computes Cholesky with at most s nonzero entries per column. """
    # without aggregation: O(n*(n s^2) + n s^3) = O(n^2 s^2)
    # with    aggregation: O((n/m)*(n (s - m)^2 + n m^2 + m^3) + (n s^3)/m)
    # = O((n^2 s^2)/m)
    n = len(x)
    sparsity = {}
    # each column in its own group
    if groups is None: groups = [[i] for i in range(n)]
    for group in groups:
        candidates = np.array(list(set(range(min(group), n)) - set(group)),
                              dtype=np.int64)
        group = np.array(group)
        selected = cknn.nonadj_select(x, candidates, group,
                                      kernel, s - len(group))
        indexes = list(candidates[selected])
        # could save memory by only storing for the minimum index
        for i in sorted(group, reverse=True):
            indexes.append(i)
            # put diagonal entry first
            sparsity[i] = [j for j in indexes[::-1] if j >= i]

    return __mult_cholesky(x, kernel, sparsity, groups)

def cholesky(theta: np.ndarray, s: int, groups: list=None,
             chol=cholesky_select) -> np.ndarray:
    """ Wrapper over point methods to deal with arbitrary matrices. """
    return chol(*matrix_kernel(theta), s, groups)

### Geometric algorithms

def naive_cholesky_kl(x: np.ndarray, kernel: Kernel,
                      rho: float, lambd: float=None) -> tuple:
    """ Computes Cholesky by KL-divergence with tuning parameters. """
    order, lengths = ordering.naive_reverse_maximin(x)
    x = x[order]
    sparsity = ordering.naive_sparsity(x, lengths, rho)
    groups, sparsity = ([[i] for i in range(len(x))], sparsity) \
        if lambd is None else ordering.supernodes(sparsity, lengths, lambd)
    return __mult_cholesky(x, kernel, sparsity, groups), order

def cholesky_kl(x: np.ndarray, kernel: Kernel,
                rho: float, lambd: float=None) -> tuple:
    """ Computes Cholesky by KL-divergence with tuning parameters. """
    order, lengths = ordering.reverse_maximin(x)
    x = x[order]
    sparsity = ordering.sparsity_pattern(x, lengths, rho)
    groups, sparsity = ([[i] for i in range(len(x))], sparsity) \
        if lambd is None else ordering.supernodes(sparsity, lengths, lambd)
    return __mult_cholesky(x, kernel, sparsity, groups), order

def __cholesky_subsample(x: np.ndarray, kernel: Kernel,
                         ref_sparsity: dict, candidate_sparsity: dict,
                         ref_groups: list, select) -> sparse.csc.csc_matrix:
    """ Subsample Cholesky within a reference sparsity and groups. """
    sparsity = {}
    for group in ref_groups:
        m = len(group)
        # select within existing sparsity pattern
        candidates = np.array(list(
            {k for j in group for k in candidate_sparsity[j]} - set(group)
        ), dtype=np.int64)
        num = max(len(ref_sparsity[group[0]]) - len(group), 0)
        selected = select(x, candidates, np.array(group), kernel, num)
        s = sorted(group + list(candidates[selected]))
        sparsity[group[0]] = s
        positions = {i: k for k, i in enumerate(s)}
        for i in group[1:]:
            # fill in blanks for rest to maintain proper number
            sparsity[i] = np.empty(len(s) - positions[i])

    return __mult_cholesky(x, kernel, sparsity, ref_groups)

def cholesky_subsample(x: np.ndarray, kernel: Kernel, s: float,
                       rho: float, lambd: float=None,
                       select=cknn.nonadj_select) -> tuple:
    """ Computes Cholesky with a mix of geometric and selection ideas. """
    # standard geometric algorithm
    order, lengths = ordering.reverse_maximin(x)
    x = x[order]
    sparsity = ordering.sparsity_pattern(x, lengths, rho)
    groups, sparsity = ([[i] for i in range(len(x))], sparsity) \
        if lambd is None else ordering.supernodes(sparsity, lengths, lambd)
    # create bigger sparsity pattern for candidates
    candidate_sparsity = ordering.sparsity_pattern(x, lengths, s*rho)
    return __cholesky_subsample(x, kernel, sparsity, candidate_sparsity,
                                groups, select), order

### Gaussian process sensor placement

# see: "Near-Optimal Sensor Placements in Gaussian Processes: Theory,
# Efficient Algorithms and Empirical Studies" by Krause et al., 2008

def naive_entropy(x: np.ndarray, kernel: Kernel, s: int) -> list:
    """ Returns a list of the most entropic points in x greedily. """
    # O(s*(s^3 + n*s^2)) = O(n s^3 + s^4)
    n = len(x)
    indexes, candidates = [], list(range(n))

    while len(candidates) > 0 and len(indexes) < s:
        score, best = -1, None
        theta_inv = inv(kernel(x[indexes]))
        for i in candidates:
            v = kernel(x[indexes], [x[i]])
            var = kernel(x[i]) - (v.T@theta_inv@v if len(indexes) > 0 else 0)
            if var > score:
                score, best = var, i
        indexes.append(best)
        candidates.remove(best)

    return indexes

def entropy(x: np.ndarray, kernel: Kernel, s: int) -> list:
    """ Returns a list of the most entropic points in x greedily. """
    # O(s*(n*s + s^2)) = O(n s^2)
    n = len(x)
    # initialization
    indexes, candidates = [], list(range(n))
    prec = np.zeros((0, 0))
    cond_var = kernel.diag(x)

    while len(candidates) > 0 and len(indexes) < s:
        # pick best entry
        k = max(candidates, key=lambda j: cond_var[j])
        indexes.append(k)
        candidates.remove(k)
        # update precision of selected entries
        v = prec@kernel(x[indexes[:-1]], [x[k]])
        var = 1/cond_var[k]
        prec = np.block([[prec + var*v@v.T, -v*var],
                         [        -var*v.T,    var]])
        # compute column k of conditional covariance
        cond_cov_k = kernel(x, [x[k]])
        cond_cov_k -= kernel(x, x[indexes[:-1]])@v
        cond_cov_k = cond_cov_k.flatten()/np.sqrt(cond_var[k])
        # update conditional variance
        cond_var -= cond_cov_k**2

    return indexes

def naive_mi(x: np.ndarray, kernel: Kernel, s: int) -> list:
    """ Max mutual information between selected and non-selected points. """
    # O(s*(s^3 + n*n^3)) = O(n^4 s)
    n = len(x)
    indexes, candidates = [], list(range(n))

    while len(candidates) > 0 and len(indexes) < s:
        score, best = -1, None
        inv1 = inv(kernel(x[indexes]))
        for i in candidates:
            v = kernel(x[indexes], [x[i]])
            var1 = kernel(x[i]) - (v.T@inv1@v if len(indexes) > 0 else 0)

            order = list(candidates)
            order.remove(i)
            v = kernel(x[order], [x[i]])
            inv2 = inv(kernel(x[order]))
            var2 = kernel(x[i]) - (v.T@inv2@v if len(order) > 0 else 0)

            if var1/var2 > score:
                score, best = var1/var2, i
        indexes.append(best)
        candidates.remove(best)

    return indexes

# simple improvement to Algorithm 1 in the Krause paper

def mi(x: np.ndarray, kernel: Kernel, s: int) -> list:
    """ Max mutual information between selected and non-selected points. """
    # O(n^3 + s*(n^2)) = O(n^3)
    n = len(x)
    # initialization
    indexes, candidates = [], list(range(n))
    prec1 = np.zeros((0, 0))
    prec2 = inv(kernel(x))
    cond_var1 = kernel.diag(x)
    # the full conditional of i corresponds to the ith diagonal in precision
    cond_var2 = 1/np.diagonal(prec2)

    while len(candidates) > 0 and len(indexes) < s:
        # pick best entry
        k = max(candidates, key=lambda j: cond_var1[j]/cond_var2[j])
        i = candidates.index(k)
        indexes.append(k)
        candidates.remove(k)
        # update precision of selected entries
        v = prec1@kernel(x[indexes[:-1]], [x[k]])
        var = 1/cond_var1[k]
        prec1 = np.block([[prec1 + var*v@v.T, -v*var],
                          [         -var*v.T,    var]])
        # compute column k of conditional covariance
        cond_cov_k = kernel(x, [x[k]])
        cond_cov_k -= kernel(x, x[indexes[:-1]])@v
        cond_cov_k = cond_cov_k.flatten()/np.sqrt(cond_var1[k])
        # update conditional variance
        cond_var1 -= cond_cov_k[:n]**2
        # update precision of candidates
        # marginalization in covariance is conditioning in precision
        prec2 -= np.outer(prec2[i], prec2[i])/prec2[i, i]
        mask = np.arange(len(prec2)) != i
        prec2 = prec2[mask][:, mask]
        # update conditional variance of candidates
        cond_var2[candidates] = 1/np.diagonal(prec2)

    return indexes

