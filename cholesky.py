import numpy as np
import scipy.linalg
import scipy.sparse as sparse
import cknn
from cknn import logdet, inv, solve, Kernel
import ordering
from gp_kernels import matrix_kernel

### helper methods

class Empty():

    """ Empty "list" simply holding the length of the data. """

    def __init__(self, n):
        self.n = n

    def __len__(self) -> int:
        return self.n

def kl_div(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes the KL divergence between the multivariate Gaussians
    at 0 with covariance X and Y, i.e. D_KL(N(0, X) || N(0, Y)).
    """
    # O(n^3)
    return 1/2*(np.trace(solve(Y, X)) + logdet(Y) - logdet(X) - len(X))

def prec_logdet(L: sparse.csc_matrix) -> float:
    """ Compute the logdet given a Cholesky factor of the precision. """
    return -2*np.sum(np.log(L.diagonal()))

def sparse_kl_div(L: sparse.csc_matrix, theta: np.ndarray=None) -> float:
    """
    Computes the KL divergence assuming L is optimal.

    Equivalent to kl_div(theta, inv(L@L.T)) if theta is provided.
    """
    # O(n^3) if a matrix theta is provided, otherwise O(n)
    logdet_theta = 0 if theta is None else \
        (cknn.logdet(theta) if isinstance(theta, np.ndarray) else theta)
    return 1/2*(prec_logdet(L) - logdet_theta)

def inv_order(order: np.ndarray) -> np.ndarray:
    """ Find the inverse permutation of the given order permutation. """
    n = order.shape[0]
    inv_order = np.arange(n)
    inv_order[order] = np.arange(n)
    return inv_order

def chol(theta: np.ndarray, sigma: float=1e-6) -> np.ndarray:
    """ Cholesky factor for the covariance. """
    try:
        return np.linalg.cholesky(theta)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(theta + sigma*np.identity(theta.shape[0]))

### sparse Cholesky methods

def __col(x: np.ndarray, kernel: Kernel, s: list) -> np.ndarray:
    """ Computes a single column of the sparse Cholesky factor. """
    # O(s^3)
    m = inv(kernel(x[s]))
    return m[:, 0]/np.sqrt(m[0, 0])

def __cholesky(x: np.ndarray, kernel: Kernel,
               sparsity: dict) -> sparse.csc_matrix:
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
    L = np.flip(chol(np.flip(kernel(x[s])))).T
    return L

def __mult_cholesky(x: np.ndarray, kernel: Kernel,
                    sparsity: dict, groups: list) -> sparse.csc_matrix:
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
    """ Computes Cholesky by KL divergence with tuning parameters. """
    order, lengths = ordering.naive_reverse_maximin(x)
    x = x[order]
    sparsity = ordering.naive_sparsity(x, lengths, rho)
    groups, sparsity = ([[i] for i in range(len(x))], sparsity) \
        if lambd is None else ordering.supernodes(sparsity, lengths, lambd)
    return __mult_cholesky(x, kernel, sparsity, groups), order

def __cholesky_kl(x: np.ndarray, kernel: Kernel, lengths: np.ndarray,
                  rho: float, lambd: float) -> tuple:
    """ Computes Cholesky given pre-ordered points and length scales. """
    sparsity = ordering.sparsity_pattern(x, lengths, rho)
    groups, sparsity = ([[i] for i in range(len(x))], sparsity) \
        if lambd is None else ordering.supernodes(sparsity, lengths, lambd)
    return sparsity, groups

def cholesky_kl(x: np.ndarray, kernel: Kernel,
                rho: float, lambd: float=None) -> tuple:
    """ Computes Cholesky by KL divergence with tuning parameters. """
    order, lengths = ordering.reverse_maximin(x)
    x = x[order]
    sparsity, groups = __cholesky_kl(x, kernel, lengths, rho, lambd)
    return __mult_cholesky(x, kernel, sparsity, groups), order

def find_cutoff(sizes: np.ndarray, nonzeros: int) -> int:
    """ Find the maximum number of nonzeros per column. """
    l, r = 0, np.max(sizes)
    while l < r:
        m = (l + r + 1)//2
        if np.sum(np.minimum(sizes, m)) <= nonzeros:
            l = m
        else:
            r = m - 1
    return l

def __cholesky_subsample(x: np.ndarray, kernel: Kernel,
                         ref_sparsity: dict, candidate_sparsity: dict,
                         ref_groups: list, select) -> sparse.csc_matrix:
    """ Subsample Cholesky within a reference sparsity and groups. """
    sparsity = {}
    # pre-compute maximum number of nonzeroes per column
    group_candidates = [np.array(list(
        {j for i in group for j in candidate_sparsity[i]} - set(group)
    ), dtype=np.int64) for group in ref_groups]
    sizes = [size for group, candidates in zip(ref_groups, group_candidates)
             for size in [len(candidates)]*len(group)]
    nonzeros = sum(map(len, ref_sparsity.values()))
    entries_left = nonzeros - sum(m*(m + 1)//2 for m in map(len, ref_groups))
    cutoff = find_cutoff(sizes, entries_left)
    # initialize nonzero trackers
    expected_total = entries_left - np.sum(np.minimum(sizes, cutoff))
    actual_total = 0
    columns_left = len(x)
    # process groups in order of increasing candidate set size
    for group, candidates in sorted(zip(ref_groups, group_candidates),
                                    key=lambda g: len(g[1])):
        m = len(group)
        # select within existing sparsity pattern
        num = max(cutoff + (expected_total - actual_total)//columns_left, 0)
        if select == cknn.chol_select:
            selected = select(x, candidates, np.array(group), kernel, num)
        else:
            selected = select(x[candidates], x[group], kernel, num)
        s = sorted(group + list(candidates[selected]))
        sparsity[group[0]] = s
        positions = {i: k for k, i in enumerate(s)}
        for i in group[1:]:
            # fill in blanks for rest to maintain proper number
            sparsity[i] = Empty(len(s) - positions[i])
        # update counters
        expected_total += m*min(cutoff, len(candidates))
        actual_total += sum(len(sparsity[i]) for i in group) - m*(m + 1)//2
        columns_left -= m

    return __mult_cholesky(x, kernel, sparsity, ref_groups)

def cholesky_subsample(x: np.ndarray, kernel: Kernel, s: float,
                       rho: float, lambd: float=None,
                       select=cknn.chol_select) -> tuple:
    """ Computes Cholesky with a mix of geometric and selection ideas. """
    # standard geometric algorithm
    order, lengths = ordering.reverse_maximin(x)
    x = x[order]
    sparsity, groups = __cholesky_kl(x, kernel, lengths, rho, lambd)
    # create bigger sparsity pattern for candidates
    candidate_sparsity = ordering.sparsity_pattern(x, lengths, s*rho)
    return __cholesky_subsample(x, kernel, sparsity, candidate_sparsity,
                                groups, select), order

def cholesky_global(x: np.ndarray, kernel: Kernel, s: float,
                    rho: float, lambd: float=None) -> tuple:
    """ Computes Cholesky by global subsampling. """
    # standard geometric algorithm
    order, lengths = ordering.reverse_maximin(x)
    x = x[order]
    sparsity, groups = __cholesky_kl(x, kernel, lengths, rho, lambd)
    # create bigger sparsity pattern for candidates
    candidate_sparsity = ordering.sparsity_pattern(x, lengths, s*rho)
    new_sparsity = cknn.global_select(x, kernel,
                                      sparsity, candidate_sparsity, groups)
    return __mult_cholesky(x, kernel, new_sparsity, groups), order

### Joint covariance methods for Gaussian process regression

def __joint_order(x_train: np.ndarray, x_test: np.ndarray) -> tuple:
    """ Return the joint ordering and length scale. """
    train_order, train_lengths = ordering.reverse_maximin(x_train)
    # initialize test point ordering with training points
    test_order, test_lengths = ordering.reverse_maximin(x_test, x_train)
    # put testing points before training points (after in transpose)
    x = np.vstack((x_test[test_order], x_train[train_order]))
    order = np.append(test_order, x_test.shape[0] + train_order)
    lengths = np.append(test_lengths, train_lengths)
    return x, order, lengths

def cholesky_joint(x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel,
                   rho: float, lambd: float=None) -> tuple:
    """ Computes Cholesky of the joint covariance. """
    x, order, lengths = __joint_order(x_train, x_test)
    sparsity, groups = __cholesky_kl(x, kernel, lengths, rho, lambd)
    return __mult_cholesky(x, kernel, sparsity, groups), order

def cholesky_joint_subsample(x_train: np.ndarray, x_test: np.ndarray,
                             kernel: Kernel, s: float, rho: float,
                             lambd: float=None,
                             select=cknn.chol_select) -> tuple:
    """ Cholesky of the joint covariance with subsampling. """
    # standard geometric algorithm
    x, order, lengths = __joint_order(x_train, x_test)
    sparsity, groups = __cholesky_kl(x, kernel, lengths, rho, lambd)
    # create bigger sparsity pattern for candidates
    candidate_sparsity = ordering.sparsity_pattern(x, lengths, s*rho)
    return __cholesky_subsample(x, kernel, sparsity, candidate_sparsity,
                                groups, select), order

def cholesky_joint_global(x_train: np.ndarray, x_test: np.ndarray,
                          kernel: Kernel, s: float, rho: float,
                          lambd: float=None,
                          select=cknn.global_select) -> tuple:
    """ Cholesky of the joint covariance with subsampling. """
    # standard geometric algorithm
    x, order, lengths = __joint_order(x_train, x_test)
    sparsity, groups = __cholesky_kl(x, kernel, lengths, rho, lambd)
    # create bigger sparsity pattern for candidates
    candidate_sparsity = ordering.sparsity_pattern(x, lengths, s*rho)
    new_sparsity = select(x, kernel, sparsity, candidate_sparsity, groups)
    return __mult_cholesky(x, kernel, new_sparsity, groups), order

