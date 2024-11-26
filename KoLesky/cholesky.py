import numpy as np
import scipy.linalg
import scipy.sparse as sparse

from . import cknn, ordering
from .cknn import inv, logdet, solve
from .gp_kernels import matrix_kernel
from .typehints import (
    CholeskyFactor,
    CholeskySelect,
    Empty,
    GlobalSelect,
    Grouping,
    Kernel,
    LengthScales,
    Matrix,
    Ordering,
    Points,
    Select,
    Sparse,
    Sparsity,
)

### helper methods


def kl_div(X: Matrix, Y: Matrix) -> float:
    """
    Computes the KL divergence between the multivariate Gaussians
    at 0 with covariance X and Y, i.e. D_KL(N(0, X) || N(0, Y)).
    """
    # O(n^3)
    return 1 / 2 * (np.trace(solve(Y, X)) + logdet(Y) - logdet(X) - len(X))


def prec_logdet(L: Sparse) -> float:
    """Compute the logdet given a Cholesky factor of the precision."""
    return -2 * np.sum(np.log(L.diagonal()))


def sparse_kl_div(L: Sparse, theta: Matrix | float | None = None) -> float:
    """
    Computes the KL divergence assuming L is optimal.

    Equivalent to kl_div(theta, inv(L@L.T)) if theta is provided.
    """
    # O(n^3) if a matrix theta is provided, otherwise O(n)
    logdet_theta = (
        0
        if theta is None
        else (cknn.logdet(theta) if isinstance(theta, np.ndarray) else theta)
    )
    return 1 / 2 * (prec_logdet(L) - logdet_theta)


def inv_order(order: Ordering) -> Ordering:
    """Find the inverse permutation of the given order permutation."""
    n = order.shape[0]
    inv_order = np.arange(n)
    inv_order[order] = np.arange(n)
    return inv_order


def chol(theta: Matrix, sigma: float = 1e-6) -> Matrix:
    """Cholesky factor for the covariance."""
    try:
        return np.linalg.cholesky(theta)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(theta + sigma * np.identity(theta.shape[0]))


def to_dense(L: Sparse, order: Ordering, overwrite: bool = False) -> Matrix:
    """Return the dense covariance L approximates."""
    n = L.shape[0]
    theta = sparse.linalg.spsolve_triangular(  # pyright: ignore
        L, np.identity(n), lower=True, overwrite_A=False, overwrite_b=True
    )
    theta = sparse.linalg.spsolve_triangular(  # pyright: ignore
        L.T, theta, lower=False, overwrite_A=overwrite, overwrite_b=True
    )
    order = inv_order(order)
    return theta[np.ix_(order, order)]


### sparse Cholesky methods


def __col(x: Matrix, kernel: Kernel, s: list[int]) -> Matrix:
    """Computes a single column of the sparse Cholesky factor."""
    # O(s^3)
    m = inv(kernel(x[s]))
    return m[:, 0] / np.sqrt(m[0, 0])


def __cholesky(x: Points, kernel: Kernel, sparsity: Sparsity) -> Sparse:
    """Computes the best Cholesky factor following the sparsity pattern."""
    # O(n s^3)
    n = len(x)
    indptr = np.cumsum([0] + [len(sparsity[i]) for i in range(n)])
    data, indexes = np.zeros(indptr[-1]), np.zeros(indptr[-1])
    for i in range(n):
        # make sure diagonal entry is first in the sparsity pattern
        s = sorted(sparsity[i])  # type: ignore
        data[indptr[i] : indptr[i + 1]] = __col(x, kernel, s)
        indexes[indptr[i] : indptr[i + 1]] = s

    return sparse.csc_matrix((data, indexes, indptr), shape=(n, n))


def __cols(x: Points, kernel: Kernel, s: list[int]) -> Matrix:
    """Computes multiple columns of the sparse Cholesky factor."""
    # O(s^3)
    # equivalent to inv(chol(inv(kernel(x[s]))))
    L = np.flip(chol(np.flip(kernel(x[s])))).T
    return L


def __mult_cholesky(
    x: Points, kernel: Kernel, sparsity: Sparsity, groups: Grouping
) -> Sparse:
    """Computes the best Cholesky factor following the sparsity pattern."""
    # O((n/m)*(s^3 + m*s^2)) = O((n s^3)/m)
    n = len(x)
    indptr = np.cumsum([0] + [len(sparsity[i]) for i in range(n)])
    data, indexes = np.zeros(indptr[-1]), np.zeros(indptr[-1])
    for group in groups:
        # points only interact with points after them (lower triangularity)
        s = sorted(sparsity[min(group)])  # type: ignore
        positions = {i: k for k, i in enumerate(s)}
        L = __cols(x, kernel, s)
        for i in group:
            k = positions[i]
            e_k = np.zeros(len(s))
            e_k[k] = 1
            col = scipy.linalg.solve_triangular(L, e_k, lower=True)
            data[indptr[i] : indptr[i + 1]] = col[k:]
            indexes[indptr[i] : indptr[i + 1]] = s[k:]

    return sparse.csc_matrix((data, indexes, indptr), shape=(n, n))


def naive_cholesky(theta: Matrix, s: int) -> Sparse:
    """Computes Cholesky with at most s nonzero entries per column."""
    # O(n*s*n*s^3) = O(n^2 s^4)
    n = len(theta)
    sparsity = {}
    points, kernel = matrix_kernel(theta)
    for i in range(n):
        # start with diagonal entry
        indexes, candidates = [i], list(range(i + 1, n))
        # add according to best KL divergence
        while len(candidates) > 0 and len(indexes) < s:
            k = max(
                candidates,
                key=lambda j: __col(points, kernel, indexes + [j])[0],
            )
            indexes.append(k)
            candidates.remove(k)
        sparsity[i] = indexes

    return __cholesky(points, kernel, sparsity)


def naive_mult_cholesky(
    theta: Matrix, s: int, groups: Grouping | None = None
) -> Sparse:
    """Computes Cholesky with at most s nonzero entries per column."""
    # O((n/m)*(s - m)*n*m*s^3 + n s^3) = O(n^2 s^3 (s - m) + n s^3)
    n = len(theta)
    sparsity = {}
    points, kernel = matrix_kernel(theta)
    # each column in its own group
    if groups is None:
        groups = [[i] for i in range(n)]
    for group in groups:
        indexes, candidates = [], list(range(max(group) + 1, n))
        # start with diagonal entries
        group.sort()
        for k, i in enumerate(group):
            indexes.append(group[k:])
        # add according to best KL divergence
        while len(candidates) > 0 and len(indexes[0]) < s:
            k = max(
                candidates,
                key=lambda j: sum(
                    np.log(__col(points, kernel, indexes[i] + [j])[0])
                    for i in range(len(group))
                ),
            )
            for col in indexes:
                col.append(k)
            candidates.remove(k)
        for k, i in enumerate(group):
            sparsity[i] = indexes[k]

    return __cholesky(points, kernel, sparsity)


def cholesky_select(
    x: Points, kernel: Kernel, s: int, groups: Grouping | None = None
) -> Sparse:
    """Computes Cholesky with at most s nonzero entries per column."""
    # without aggregation: O(n*(n s^2) + n s^3) = O(n^2 s^2)
    # with    aggregation: O((n/m)*(n (s - m)^2 + n m^2 + m^3) + (n s^3)/m)
    # = O((n^2 s^2)/m)
    n = len(x)
    sparsity = {}
    # each column in its own group
    if groups is None:
        groups = [[i] for i in range(n)]
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


def naive_nonadj_cholesky(
    theta: Matrix, s: int, groups: Grouping | None = None
) -> Sparse:
    """Computes Cholesky with at most s nonzero entries per column."""
    # O((n/m)*(s - m)*n*m*s^3 + n s^3) = O(n^2 s^3 (s - m) + n s^3)
    n = len(theta)
    sparsity = {}
    points, kernel = matrix_kernel(theta)
    # each column in its own group
    if groups is None:
        groups = [[i] for i in range(n)]
    for group in groups:
        indexes, candidates = [], list(set(range(n)) - set(group))
        # start with diagonal entries
        group.sort()
        for k, i in enumerate(group):
            indexes.append(group[k:])
        # add according to best KL divergence
        while len(candidates) > 0 and len(indexes[0]) < s:
            k = max(
                candidates,
                key=lambda j: sum(
                    np.log(
                        __col(
                            points,
                            kernel,
                            indexes[i] + ([j] if j >= group[i] else []),
                        )[0]
                    )
                    for i in range(len(group))
                ),
            )
            for i, col in enumerate(indexes):
                if k >= group[i]:
                    col.append(k)
            candidates.remove(k)
        for k, i in enumerate(group):
            sparsity[i] = indexes[k]

    return __cholesky(points, kernel, sparsity)


def cholesky_nonadj_select(
    x: Points, kernel: Kernel, s: int, groups: Grouping | None = None
) -> Sparse:
    """Computes Cholesky with at most s nonzero entries per column."""
    # without aggregation: O(n*(n s^2) + n s^3) = O(n^2 s^2)
    # with    aggregation: O((n/m)*(n (s - m)^2 + n m^2 + m^3) + (n s^3)/m)
    # = O((n^2 s^2)/m)
    n = len(x)
    sparsity = {}
    # each column in its own group
    if groups is None:
        groups = [[i] for i in range(n)]
    for group in groups:
        candidates = np.array(
            list(set(range(min(group), n)) - set(group)), dtype=np.int64
        )
        group = np.array(group)
        selected = cknn.nonadj_select(
            x, candidates, group, kernel, s - len(group)
        )
        indexes = list(candidates[selected])
        # could save memory by only storing for the minimum index
        for i in sorted(group, reverse=True):
            indexes.append(i)
            # put diagonal entry first
            sparsity[i] = [j for j in indexes[::-1] if j >= i]

    return __mult_cholesky(x, kernel, sparsity, groups)


def cholesky(
    theta: Matrix,
    s: int,
    groups: Grouping | None = None,
    chol: CholeskySelect = cholesky_select,
) -> Sparse:
    """Wrapper over point methods to deal with arbitrary matrices."""
    return chol(*matrix_kernel(theta), s, groups)


### Geometric algorithms


def naive_cholesky_kl(
    x: Points,
    kernel: Kernel,
    rho: float,
    lambd: float | None = None,
    p: int = 1,
) -> CholeskyFactor:
    """Computes Cholesky by KL divergence with tuning parameters."""
    order, lengths = ordering.naive_p_reverse_maximin(x, p=p)
    x = x[order]
    sparsity = ordering.naive_sparsity(x, lengths, rho)
    groups, sparsity = (
        ([[i] for i in range(len(x))], sparsity)
        if lambd is None
        else ordering.supernodes(sparsity, lengths, lambd)
    )
    return __mult_cholesky(x, kernel, sparsity, groups), order


def __cholesky_kl(
    x: Points,
    lengths: LengthScales,
    rho: float,
    lambd: float | None,
) -> tuple[Sparsity, Grouping]:
    """Computes Cholesky given pre-ordered points and length scales."""
    sparsity = ordering.sparsity_pattern(x, lengths, rho)
    groups, sparsity = (
        ([[i] for i in range(len(x))], sparsity)
        if lambd is None
        else ordering.supernodes(sparsity, lengths, lambd)
    )
    return sparsity, groups


def cholesky_kl(
    x: Points,
    kernel: Kernel,
    rho: float,
    lambd: float | None = None,
    p: int = 1,
) -> CholeskyFactor:
    """Computes Cholesky by KL divergence with tuning parameters."""
    order, lengths = ordering.p_reverse_maximin(x, p=p)
    x = x[order]
    sparsity, groups = __cholesky_kl(x, lengths, rho, lambd)
    return __mult_cholesky(x, kernel, sparsity, groups), order


def find_cutoff(sizes: list[int], nonzeros: int) -> int:
    """Find the maximum number of nonzeros per column."""
    left, right = 0, np.max(sizes)
    while left < right:
        m = (left + right + 1) // 2
        if np.sum(np.minimum(sizes, m)) <= nonzeros:
            left = m
        else:
            right = m - 1
    return left


def __cholesky_subsample(
    x: Points,
    kernel: Kernel,
    ref_sparsity: Sparsity,
    candidate_sparsity: Sparsity,
    ref_groups: Grouping,
    select: Select,
) -> Sparse:
    """Subsample Cholesky within a reference sparsity and groups."""
    sparsity = {}
    # pre-compute maximum number of nonzeroes per column
    group_candidates = [
        # fmt: off
        np.array(list(
            {j for i in group for j in candidate_sparsity[i]}  # type: ignore
            - set(group)
        ), dtype=np.int64)
        # fmt: on
        for group in ref_groups
    ]
    sizes = [
        size
        for group, candidates in zip(ref_groups, group_candidates)
        for size in [len(candidates)] * len(group)
    ]
    nonzeros = sum(map(len, ref_sparsity.values()))
    entries_left = nonzeros - sum(
        m * (m + 1) // 2 for m in map(len, ref_groups)
    )
    cutoff = find_cutoff(sizes, entries_left)
    # initialize nonzero trackers
    expected_total = entries_left - np.sum(np.minimum(sizes, cutoff))
    actual_total = 0
    columns_left = len(x)
    # process groups in order of increasing candidate set size
    for group, candidates in sorted(
        zip(ref_groups, group_candidates), key=lambda g: len(g[1])
    ):
        m = len(group)
        # select within existing sparsity pattern
        num = max(cutoff + (expected_total - actual_total) // columns_left, 0)
        selected = (
            select(x, candidates, np.array(group), kernel, num)  # type: ignore
            if select == cknn.chol_select or select == cknn.nonadj_select
            else select(x[candidates], x[group], kernel, num)  # type: ignore
        )
        s = sorted(group + list(candidates[selected]))
        sparsity[group[0]] = s
        positions = {i: k for k, i in enumerate(s)}
        for i in group[1:]:
            # fill in blanks for rest to maintain proper number
            sparsity[i] = Empty(len(s) - positions[i])
        # update counters
        expected_total += m * min(cutoff, len(candidates))
        actual_total += sum(len(sparsity[i]) for i in group) - m * (m + 1) // 2
        columns_left -= m

    return __mult_cholesky(x, kernel, sparsity, ref_groups)


def cholesky_subsample(
    x: Points,
    kernel: Kernel,
    s: float,
    rho: float,
    lambd: float | None = None,
    p: int = 1,
    select: Select = cknn.chol_select,
    *,
    reference: tuple[Sparsity, Grouping] | None = None,
) -> CholeskyFactor:
    """Computes Cholesky with a mix of geometric and selection ideas."""
    # standard geometric algorithm
    order, lengths = ordering.p_reverse_maximin(x, p=p)
    x = x[order]
    sparsity, groups = (
        __cholesky_kl(x, lengths, rho, lambd)
        if reference is None
        else reference
    )
    # create bigger sparsity pattern for candidates
    candidate_sparsity = ordering.sparsity_pattern(x, lengths, s * rho)
    return (
        __cholesky_subsample(
            x, kernel, sparsity, candidate_sparsity, groups, select
        ),
        order,
    )


def cholesky_knn(
    x: Points,
    kernel: Kernel,
    rho: float,
    p: int = 1,
    *,
    reference: tuple[Sparsity, Grouping] | None = None,
) -> CholeskyFactor:
    """Computes Cholesky with k-nearest neighbor sets."""
    order, lengths = ordering.p_reverse_maximin(x, p=p)
    x = x[order]
    sparsity, groups = (
        __cholesky_kl(x, lengths, rho, None)
        if reference is None
        else reference
    )
    avg_nonzeros = sum(map(len, sparsity.values())) // len(sparsity)
    sparsity = ordering.knn_sparsity(x, avg_nonzeros)
    return __mult_cholesky(x, kernel, sparsity, groups), order


def cholesky_global(
    x: Points,
    kernel: Kernel,
    s: float,
    rho: float,
    lambd: float | None = None,
    p: int = 1,
) -> CholeskyFactor:
    """Computes Cholesky by global subsampling."""
    # standard geometric algorithm
    order, lengths = ordering.p_reverse_maximin(x, p=p)
    x = x[order]
    sparsity, groups = __cholesky_kl(x, lengths, rho, lambd)
    # create bigger sparsity pattern for candidates
    candidate_sparsity = ordering.sparsity_pattern(x, lengths, s * rho)
    new_sparsity = cknn.global_select(
        x, kernel, sparsity, candidate_sparsity, groups
    )
    return __mult_cholesky(x, kernel, new_sparsity, groups), order


### Joint covariance methods for Gaussian process regression


def __joint_order(
    x_train: Points, x_test: Points, p: int = 1
) -> tuple[Points, Ordering, LengthScales]:
    """Return the joint ordering and length scale."""
    train_order, train_lengths = ordering.p_reverse_maximin(x_train, p=p)
    # initialize test point ordering with training points
    test_order, test_lengths = ordering.p_reverse_maximin(x_test, x_train, p=p)
    # put testing points before training points (after in transpose)
    x = np.vstack((x_test[test_order], x_train[train_order]))
    order = np.append(test_order, x_test.shape[0] + train_order)
    lengths = np.append(test_lengths, train_lengths)
    return x, order, lengths


def cholesky_joint(
    x_train: Points,
    x_test: Points,
    kernel: Kernel,
    rho: float,
    lambd: float | None = None,
    p: int = 1,
) -> CholeskyFactor:
    """Computes Cholesky of the joint covariance."""
    x, order, lengths = __joint_order(x_train, x_test, p=p)
    sparsity, groups = __cholesky_kl(x, lengths, rho, lambd)
    return __mult_cholesky(x, kernel, sparsity, groups), order


def cholesky_joint_subsample(
    x_train: Points,
    x_test: Points,
    kernel: Kernel,
    s: float,
    rho: float,
    lambd: float | None = None,
    p: int = 1,
    select: Select = cknn.chol_select,
) -> CholeskyFactor:
    """Cholesky of the joint covariance with subsampling."""
    # standard geometric algorithm
    x, order, lengths = __joint_order(x_train, x_test, p=p)
    sparsity, groups = __cholesky_kl(x, lengths, rho, lambd)
    # create bigger sparsity pattern for candidates
    candidate_sparsity = ordering.sparsity_pattern(x, lengths, s * rho)
    return (
        __cholesky_subsample(
            x, kernel, sparsity, candidate_sparsity, groups, select
        ),
        order,
    )


def cholesky_joint_knn(
    x_train: Points,
    x_test: Points,
    kernel: Kernel,
    rho: float,
    p: int = 1,
) -> CholeskyFactor:
    """Cholesky of the joint covariance with k-nearest neighbor sets."""
    # standard geometric algorithm
    x, order, lengths = __joint_order(x_train, x_test, p=p)
    sparsity, groups = __cholesky_kl(x, lengths, rho, None)
    # create bigger sparsity pattern for candidates
    avg_nonzeros = sum(map(len, sparsity.values())) // len(sparsity)
    sparsity = ordering.knn_sparsity(x, avg_nonzeros)
    return __mult_cholesky(x, kernel, sparsity, groups), order


def cholesky_joint_global(
    x_train: Points,
    x_test: Points,
    kernel: Kernel,
    s: float,
    rho: float,
    lambd: float | None = None,
    p: int = 1,
    select: GlobalSelect = cknn.global_select,
) -> CholeskyFactor:
    """Cholesky of the joint covariance with subsampling."""
    # standard geometric algorithm
    x, order, lengths = __joint_order(x_train, x_test, p=p)
    sparsity, groups = __cholesky_kl(x, lengths, rho, lambd)
    # create bigger sparsity pattern for candidates
    candidate_sparsity = ordering.sparsity_pattern(x, lengths, s * rho)
    new_sparsity = select(x, kernel, sparsity, candidate_sparsity, groups)
    return __mult_cholesky(x, kernel, new_sparsity, groups), order
