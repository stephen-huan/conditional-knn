import numpy as np
import scipy.linalg
import scipy.spatial.distance
from sklearn.gaussian_process.kernels import Kernel, Matern

import ccknn
from maxheap import Heap

### helper methods


def logdet(m: np.ndarray) -> float:
    """Computes the logarithm of the determinant of m."""
    return np.linalg.slogdet(m)[1]


def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the system Ax = b for symmetric positive definite A."""
    return scipy.linalg.solve(A, b, assume_a="pos")


def inv(m: np.ndarray) -> np.ndarray:
    """Inverts a symmetric positive definite matrix m."""
    return np.linalg.inv(m)
    # below only starts to get faster for large matrices (~10^4)
    # return solve(m, np.identity(m.shape[0]))


def euclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the Euclidean distance between points in x and y."""
    # larger kernel value implies more similarity, flip
    return -scipy.spatial.distance.cdist(x, y, "euclidean")


### selection methods


def __naive_select(
    x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel, s: int
) -> list:
    """Brute force selection method."""
    # O(s*n*s^3) = O(n s^4)
    n = x_train.shape[0]
    indexes, candidates = [], list(range(n))

    while len(candidates) > 0 and len(indexes) < s:
        score, best = float("inf"), None
        for i in candidates:
            new = indexes + [i]
            v = kernel(x_train[new], x_test)
            cov = kernel(x_test) - v.T @ inv(kernel(x_train[new])) @ v
            if cov < score:
                score, best = cov, i
        indexes.append(best)
        candidates.remove(best)

    return indexes


def __prec_select(
    x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel, s: int
) -> list:
    """Greedily select the s entries maximizing mutual information."""
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
        k = max(candidates, key=lambda j: cond_cov[j] ** 2 / cond_var[j])
        indexes.append(k)
        candidates.remove(k)
        # update precision of selected entries
        v = prec @ kernel(x_train[indexes[:-1]], [x_train[k]])
        var = 1 / cond_var[k]
        # fmt: off
        prec = np.block([
            [prec + var*v@v.T, -var*v],
            [        -var*v.T,    var],
        ])
        # fmt: on
        # compute column k of conditional covariance
        cond_cov_k = kernel(points, [x_train[k]])
        # n.b.: this takes O(ns) space from the kernel function call, however,
        # this can be done in O(1) space by considering each scalar in turn
        cond_cov_k -= kernel(points, x_train[indexes[:-1]]) @ v
        cond_cov_k = cond_cov_k.flatten() / np.sqrt(cond_var[k])
        # update conditional variance and covariance
        cond_var -= cond_cov_k[:n] ** 2
        cond_cov -= cond_cov_k[:n] * cond_cov_k[n]

    return indexes


def __chol_update(
    cov_k: np.ndarray,
    i: int,
    k: int,
    factors: np.ndarray,
    cond_var: np.ndarray,
    cond_cov: np.ndarray = None,
) -> None:
    """Updates the ith column of the Cholesky factor with column k."""
    n = cond_var.shape[0]
    # update Cholesky factors
    factors[:, i] = cov_k
    factors[:, i] -= factors[:, :i] @ factors[k, :i]
    factors[:, i] /= np.sqrt(factors[k, i])
    # update conditional variance and covariance
    cond_var -= factors[:, i][:n] ** 2
    if cond_cov is not None:
        cond_cov -= np.outer(factors[:, i][:n], factors[:, i][n:])


def __select_update(
    x_train: np.ndarray,
    x_test: np.ndarray,
    kernel: Kernel,
    i: int,
    k: int,
    factors: np.ndarray,
    cond_var: np.ndarray,
    cond_cov: np.ndarray,
) -> None:
    """Update the selection data structures after selecting a point."""
    n = x_train.shape[0]
    points = np.vstack((x_train, x_test))
    cov_k = kernel(points, x_train[k : k + 1]).flatten()
    __chol_update(cov_k, i, k, factors, cond_var, cond_cov[:, np.newaxis])
    # mark index as selected
    if k < n:
        cond_var[k] = np.inf


def __chol_select(
    x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel, s: int
) -> list:
    """Select the s most informative entries, storing a Cholesky factor."""
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
        k = max(candidates, key=lambda j: cond_cov[j] ** 2 / cond_var[j])
        indexes.append(k)
        candidates.remove(k)
        # update data structures
        __select_update(
            x_train,
            x_test,
            kernel,
            len(indexes) - 1,
            k,
            factors,
            cond_var,
            cond_cov,
        )

    return indexes


### multiple point selection


def __naive_mult_select(
    x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel, s: int
) -> list:
    """Brute force multiple point selection method."""
    # O(s*n*(s^3 + m^3)) = O(n s^4 + n s m^3)
    n = x_train.shape[0]
    indexes, candidates = [], list(range(n))

    while len(candidates) > 0 and len(indexes) < s:
        score, best = float("inf"), None
        for i in candidates:
            new = indexes + [i]
            v = kernel(x_train[new], x_test)
            cov = logdet(kernel(x_test) - v.T @ inv(kernel(x_train[new])) @ v)
            if cov < score:
                score, best = cov, i
        indexes.append(best)
        candidates.remove(best)

    return indexes


def __prec_mult_select(
    x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel, s: int
) -> list:
    """Greedily select the s entries minimizing conditional covariance."""
    # O(m^3 + n*m^2 + s*(s^2 + m^2 + n*s + n*m + n*m)) = O(n s^2 + n m^2 + m^3)
    n, m = x_train.shape[0], x_test.shape[0]
    points = np.vstack((x_train, x_test))
    # initialization
    indexes, candidates = [], list(range(n))
    prec = np.zeros((0, 0))
    prec_pr = inv(kernel(x_test, x_test))
    cond_cov = kernel(x_train, x_test)
    cond_var = kernel.diag(x_train)
    cond_var_pr = cond_var - np.sum((cond_cov @ prec_pr) * cond_cov, axis=1)

    while len(candidates) > 0 and len(indexes) < s:
        # pick best entry
        k = min(candidates, key=lambda j: cond_var_pr[j] / cond_var[j])
        indexes.append(k)
        candidates.remove(k)
        # update precision of selected entries
        v = prec @ kernel(x_train[indexes[:-1]], [x_train[k]])
        var = 1 / cond_var[k]
        # fmt: off
        prec = np.block([
            [prec + var*v@v.T, -var*v],
            [        -var*v.T,    var],
        ])
        # fmt: on
        # update precision of prediction covariance
        u = prec_pr @ cond_cov[k]
        prec_pr += np.outer(u, u) / cond_var_pr[k]
        # compute column k of conditional covariance
        cond_cov_k = kernel(points, [x_train[k]])
        cond_cov_k -= kernel(points, x_train[indexes[:-1]]) @ v
        cond_cov_k = cond_cov_k.flatten()
        cond_cov_pr_k = np.copy(cond_cov_k[:n])
        cond_cov_pr_k -= cond_cov @ u
        cond_cov_k /= np.sqrt(cond_var[k])
        cond_cov_pr_k /= np.sqrt(cond_var_pr[k])
        # update conditional variance and covariance
        cond_var -= cond_cov_k[:n] ** 2
        cond_var_pr -= cond_cov_pr_k**2
        cond_cov -= np.outer(cond_cov_k[:n], cond_cov_k[n:])

    return indexes


def __chol_mult_select(
    x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel, s: int
) -> np.ndarray:
    """Greedily select the s entries minimizing conditional covariance."""
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

    for i in range(indexes.shape[0]):
        # pick best entry
        k = min(
            set(range(n)) - set(indexes[:i]),
            key=lambda j: cond_var_pr[j] / cond_var[j],
        )
        indexes[i] = k
        # update Cholesky factors
        cov_k = kernel(x_train, [x_train[k]]).flatten()
        __chol_update(cov_k, i, k, factors, cond_var)
        __chol_update(cov_k, i + m, k, factors_pr, cond_var_pr)

    return indexes


### non-adjacent multiple point selection


def __chol_insert(
    cov_k: np.ndarray,
    order: np.ndarray,
    i: int,
    index: int,
    k: int,
    factors: np.ndarray,
) -> None:
    """Updates the ith column of the Cholesky factor with column k."""
    # move columns over to make space at index
    for col in range(i, index, -1):
        factors[:, col] = factors[:, col - 1]

    # insert conditional covariance with k into Cholesky factor at index
    factors[:, index] = cov_k
    factors[:, index] -= factors[:, :index] @ factors[k, :index]
    factors[:, index] /= np.sqrt(factors[k, index])

    # update downstream Cholesky factor by rank-one downdate
    cov_k = np.copy(factors[:, index])
    for col in range(index + 1, i + 1):
        k = order[col]
        c1, c2 = factors[k, col], cov_k[k]
        dp = np.sqrt(c1 * c1 - c2 * c2)
        c1, c2 = c1 / dp, c2 / dp
        factors[:, col] *= c1
        factors[:, col] += -c2 * cov_k
        cov_k *= 1 / c1
        cov_k += -c2 / c1 * factors[:, col]


def __insert_index(
    order: np.ndarray, locations: np.ndarray, i: int, k: int
) -> int:
    """Finds the index to insert index k into the order."""
    index = -1
    for index in range(i):
        # bigger than current value, insertion spot
        if locations[k] >= locations[order[index]]:
            return index
    return index + 1


def __select_point(
    order: np.ndarray,
    locations: np.ndarray,
    points: np.ndarray,
    i: int,
    k: int,
    kernel: Kernel,
    factors: np.ndarray,
    var: np.ndarray,
):
    """Add the kth point to the Cholesky factor."""
    index = __insert_index(order, locations, i, k)
    # shift values over to make room for k at index
    for col in range(i, index, -1):
        order[col] = order[col - 1]
    order[index] = k
    # update Cholesky factor
    cov_k = kernel(points, [points[k]]).flatten()
    __chol_insert(cov_k, order, i, index, k, factors)
    # mark as selected
    if k < var.shape[0]:
        var[k] = np.inf


def __scores_update(
    order: np.ndarray,
    locations: np.ndarray,
    i: int,
    factors: np.ndarray,
    var: np.ndarray,
    scores: np.ndarray,
) -> int:
    """Update the scores for each candidate."""
    n, m = var.shape[0], locations.shape[0] - var.shape[0]
    # compute baseline log determinant before conditioning
    prev_logdet = 0
    for col in range(i):
        k = order[col]
        # add log conditional variance of prediction point
        if k >= n:
            prev_logdet += 2 * np.log(factors[k, col])
    # pick best entry
    best, best_k = -np.inf, 0
    for j in range(scores.shape[0]):
        # selected already, don't consider as candidate
        if var[j] == np.inf:
            continue

        cond_var_j = var[j]
        index = __insert_index(order, locations, i, j)
        key = 0

        for col in range(i):
            k = order[col]
            cond_var_k = factors[k, col]
            cond_cov_k = factors[j, col] * cond_var_k
            cond_var_k *= cond_var_k
            cond_cov_k *= cond_cov_k
            # add log conditional variance of prediction point
            if k >= n:
                key += np.log(
                    cond_var_k
                    - (cond_cov_k / cond_var_j if col >= index else 0)
                )
            cond_var_j -= cond_cov_k / cond_var_k

        key = prev_logdet - key
        scores[j] = key
        if key > best:
            best, best_k = key, j

    return best_k


def __chol_nonadj_select(
    x: np.ndarray, train: np.ndarray, test: np.ndarray, kernel: Kernel, s: int
) -> np.ndarray:
    """Greedily select the s entries minimizing conditional covariance."""
    # O(m*(n + m)*m + s*(n + m)*(s + m)) = O(n s^2 + n m^2 + m^3)
    n, m = train.shape[0], test.shape[0]
    locations = np.append(train, test)
    points = x[locations]
    # initialization
    indexes = np.zeros(min(n, s), dtype=np.int64)
    order = np.zeros(indexes.shape[0] + test.shape[0], dtype=np.int64)
    factors = np.zeros((n + m, s + m))
    var = kernel.diag(x[train])
    scores = np.zeros(n)
    # pre-condition on the m prediction points
    for i in range(m):
        __select_point(
            order, locations, points, i, n + i, kernel, factors, var
        )
    for i in range(indexes.shape[0]):
        # pick best entry
        k = __scores_update(order, locations, i + m, factors, var, scores)
        indexes[i] = k
        # update Cholesky factor
        __select_point(
            order, locations, points, i + m, k, kernel, factors, var
        )

    return indexes


def __budget_select(
    x: np.ndarray, train: np.ndarray, test: np.ndarray, kernel: Kernel, s: int
) -> np.ndarray:
    """Greedily select the s entries minimizing conditional covariance."""
    # O(m*(n + m)*m + s*(n + m)*(s + m)) = O(n s^2 + n m^2 + m^3)
    n, m = train.shape[0], test.shape[0]
    locations = np.append(train, test)
    points = x[locations]
    # allow each selected point to condition all the prediction points
    budget = m * s
    max_sel = min(n, budget)
    # initialization
    indexes = np.zeros(max_sel, dtype=np.int64)
    order = np.zeros(indexes.shape[0] + test.shape[0], dtype=np.int64)
    factors = np.zeros((n + m, max_sel + m))
    var = kernel.diag(x[train])
    scores = np.zeros(n)
    num_cond = np.array(
        [sum(i < train[k] for i in test) for k in range(len(train))]
    )
    # pre-condition on the m prediction points
    for i in range(m):
        __select_point(
            order, locations, points, i, n + i, kernel, factors, var
        )
    i = 0
    while i < indexes.shape[0] and budget > 0:
        # pick best entry
        __scores_update(order, locations, i + m, factors, var, scores)
        k = max(
            set(range(len(train))) - set(indexes[:i]),
            key=lambda i: scores[i] / num_cond[i],
        )
        # subtract number conditioned from budget
        budget -= num_cond[k]
        if budget < 0:
            break
        indexes[i] = k
        # update Cholesky factor
        __select_point(
            order, locations, points, i + m, k, kernel, factors, var
        )
        i += 1

    return indexes[:i]


### high-level selection methods


def random_select(
    x_train: np.ndarray,
    x_test: np.ndarray,
    kernel: Kernel,
    s: int,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Randomly select s points out of x_train."""
    s = np.clip(s, 0, x_train.shape[0])
    if rng is None:
        rng = np.random.default_rng(1)
    return rng.choice(x_train.shape[0], s, replace=False)


def corr_select(
    x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel, s: int
) -> np.ndarray:
    """Select s points in x_train with highest correlation to x_test."""
    # O(n log n)
    s = np.clip(s, 0, x_train.shape[0])
    # aggregate for multiple prediction points
    corrs = np.max(kernel(x_train, x_test) ** 2, axis=1) / kernel.diag(x_train)
    return np.argsort(corrs)[x_train.shape[0] - s :]


def knn_select(
    x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel, s: int
) -> np.ndarray:
    """Select s points in x_train "closest" to x_test by kernel function."""
    # O(n log n)
    s = np.clip(s, 0, x_train.shape[0])
    # for Matern, sorting by kernel is equivalent to sorting by distance
    # aggregate for multiple prediction points
    dists = np.max(kernel(x_train, x_test), axis=1)
    return np.argsort(dists)[-1 : -1 - s : -1]


def select(
    x_train: np.ndarray,
    x_test: np.ndarray,
    kernel: Kernel,
    s: int,
    select_method=ccknn.select,
) -> np.ndarray:
    """Wrapper over various cknn selection methods."""
    # early exit
    s = np.clip(s, 0, x_train.shape[0])
    if s == 0:
        return []
    selected = select_method(x_train, x_test, kernel, s)
    assert len(set(selected)) == len(selected), "selected indices not distinct"
    return selected


def nonadj_select(
    x: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    kernel: Kernel,
    s: int,
    select_method=ccknn.nonadj_select,
) -> np.ndarray:
    """Wrapper over various cknn selection methods."""
    # early exit
    s = np.clip(s, 0, train.shape[0])
    if s == 0:
        return []
    selected = select_method(x, train, test, kernel, s)
    assert len(set(selected)) == len(selected), "selected indices not distinct"
    return selected


def chol_select(
    x: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    kernel: Kernel,
    s: int,
    select_method=ccknn.chol_select,
) -> np.ndarray:
    """Wrapper over selection specialized to Cholesky factorization."""
    # early exit
    s = np.clip(s, 0, train.shape[0])
    if s == 0:
        return []
    selected = select_method(x, train, test, kernel, s)
    assert len(set(selected)) == len(selected), "selected indices not distinct"
    return selected


### global selection


def __global_select(
    x: np.ndarray,
    kernel: Kernel,
    ref_sparsity: dict,
    candidate_sparsity: dict,
    ref_groups: list,
) -> dict:
    """Construct a sparsity pattern from a candidate set over all columns."""
    sparsity = {group[0]: [] for group in ref_groups}
    N = len(x)

    group_candidates = [
        np.array(
            list(
                {j for i in group for j in candidate_sparsity[i]} - set(group)
            ),
            dtype=np.int64,
        )
        for group in ref_groups
    ]
    nonzeros = sum(map(len, ref_sparsity.values()))
    entries_left = nonzeros - sum(
        m * (m + 1) // 2 for m in map(len, ref_groups)
    )
    max_sel = 3 * (entries_left // N)

    # initialize data structures
    factors, cond_covs, cond_vars = [], [], []
    for group, candidates in zip(ref_groups, group_candidates):
        factor = np.zeros((len(candidates) + len(group), max_sel), order="F")
        factors.append(factor)
        cond_covs.append(kernel(x[candidates], x[group]).flatten())
        cond_vars.append(kernel.diag(x[candidates]))

    num_sel = np.zeros(len(ref_groups), dtype=np.int64)
    group_var = kernel.diag(x[[i for group in ref_groups for i in group]])

    # add candidates to max heap
    values, ids = [], []
    for i, (group, candidates) in enumerate(zip(ref_groups, group_candidates)):
        cond_var, cond_cov, var = cond_vars[i], cond_covs[i], group_var[i]
        for j in range(len(candidates)):
            values.append((cond_cov[j] ** 2 / cond_var[j]) / var)
            ids.append(N * j + i)
    values, ids = np.array(values), np.array(ids, dtype=np.int64)

    heap = Heap(values, ids)
    while len(heap) > 0 and entries_left > 0:
        _, entry = heap.pop()
        k, i = entry // N, entry % N
        # do not select if group already has enough entries
        if num_sel[i] >= max_sel:
            continue

        group, candidates = ref_groups[i], group_candidates[i]
        factor, cond_var, cond_cov = factors[i], cond_vars[i], cond_covs[i]

        # add entry to sparsity pattern
        sparsity[group[0]].append(candidates[k])
        # update data structures
        group_var[i] -= cond_cov[k] ** 2 / cond_var[k]
        __select_update(
            x[candidates],
            x[group],
            kernel,
            num_sel[i],
            k,
            factor,
            cond_var,
            cond_cov,
        )
        # update affected candidates
        for j in range(len(candidates)):
            # if hasn't been selected already
            if cond_var[j] != np.inf:
                heap.update_key(
                    N * j + i, (cond_cov[j] ** 2 / cond_var[j]) / group_var[i]
                )

        num_sel[i] += 1
        entries_left -= len(group)

    return {
        group[0]: sorted(group + sparsity[group[0]]) for group in ref_groups
    }


def __global_mult_select(
    x: np.ndarray,
    kernel: Kernel,
    ref_sparsity: dict,
    candidate_sparsity: dict,
    ref_groups: list,
) -> dict:
    """Construct a sparsity pattern from a candidate set over all columns."""
    sparsity = {group[0]: [] for group in ref_groups}
    N = len(x)

    group_candidates = [
        np.array(
            list(
                {j for i in group for j in candidate_sparsity[i]} - set(group)
            ),
            dtype=np.int64,
        )
        for group in ref_groups
    ]
    nonzeros = sum(map(len, ref_sparsity.values()))
    entries_left = nonzeros - sum(
        m * (m + 1) // 2 for m in map(len, ref_groups)
    )
    max_sel = 3 * (entries_left // N)

    # initialize data structures
    factors, orders, init_vars, scores = [], [], [], []
    for i, (group, candidates) in enumerate(zip(ref_groups, group_candidates)):
        n, m = len(candidates), len(group)
        factor = np.zeros((n + m, max_sel + m), order="F")
        factors.append(factor)
        order = np.zeros(min(n, max_sel) + m, dtype=np.int64)
        orders.append(order)
        var = kernel.diag(x[candidates])
        init_vars.append(var)
        score = np.zeros(n)
        scores.append(score)
        # pre-condition on the m prediction points
        locations = np.append(candidates, group)
        points = x[locations]
        for j in range(m):
            __select_point(
                order, locations, points, j, n + j, kernel, factor, var
            )
        __scores_update(order, locations, m, factor, var, score)

    num_sel = np.zeros(len(ref_groups), dtype=np.int64)
    # pre-compute prediction points conditioned for each candidate
    num_conds = [
        [sum(i < candidates[k] for i in group) for k in range(len(candidates))]
        for group, candidates in zip(ref_groups, group_candidates)
    ]

    # add candidates to max heap
    values, ids = [], []
    for i, (group, candidates) in enumerate(zip(ref_groups, group_candidates)):
        score, num_cond = scores[i], num_conds[i]
        for j in range(len(candidates)):
            values.append(score[j] / num_cond[j])
            ids.append(N * j + i)
    values, ids = np.array(values), np.array(ids, dtype=np.int64)

    heap = Heap(values, ids)
    while len(heap) > 0 and entries_left > 0:
        _, entry = heap.pop()
        k, i = entry // N, entry % N
        num_cond = num_conds[i]
        # do not select if group already has enough entries
        # or there are not enough entries left
        if num_sel[i] >= max_sel or num_cond[k] > entries_left:
            continue

        group, candidates = ref_groups[i], group_candidates[i]
        n, m = len(candidates), len(group)
        locations = np.append(candidates, group)
        points = x[locations]
        factor = factors[i]
        order, var, score = orders[i], init_vars[i], scores[i]

        # add entry to sparsity pattern
        sparsity[group[0]].append(candidates[k])
        # update data structures
        __select_point(
            order, locations, points, num_sel[i] + m, k, kernel, factor, var
        )
        num_sel[i] += 1
        __scores_update(order, locations, num_sel[i] + m, factor, var, score)
        # update affected candidates
        for j in range(len(candidates)):
            # if hasn't been selected already
            if var[j] != np.inf:
                heap.update_key(N * j + i, score[j] / num_cond[j])

        entries_left -= num_cond[k]

    # construct groups
    for group in ref_groups:
        s = sorted(group + list(sparsity[group[0]]))
        sparsity[group[0]] = s
        positions = {i: k for k, i in enumerate(s)}
        for i in group[1:]:
            # fill in blanks for rest to maintain proper number
            sparsity[i] = np.empty(len(s) - positions[i])

    return sparsity


def global_select(
    x: np.ndarray,
    kernel: Kernel,
    ref_sparsity: dict,
    candidate_sparsity: dict,
    ref_groups: list,
    select_method=ccknn.global_select,
) -> dict:
    """Construct a sparsity pattern from a candidate set over all columns."""
    return select_method(
        x, kernel, ref_sparsity, candidate_sparsity, ref_groups
    )
