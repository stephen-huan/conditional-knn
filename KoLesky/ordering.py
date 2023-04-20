import numpy as np
import scipy.spatial.distance
from scipy.spatial import KDTree

from maxheap import Heap
from typehints import (
    Empty,
    Grouping,
    LengthScales,
    Matrix,
    Ordering,
    Points,
    Sparsity,
)

# reverse maximum-minimum distance (reverse-maximin) ordering


def dist(x: Points, y: Points) -> float:
    """Return the Euclidean distance between x and y."""
    # scipy.spatial.distance.euclidean is slow
    d = x - y
    return np.sqrt(d.dot(d))


def euclidean(x: Points, y: Points) -> Matrix:
    """Return the distance between points in x and y."""
    return scipy.spatial.distance.cdist(x, y, "euclidean")


def naive_reverse_maximin(
    x: Points, initial: Points | None = None
) -> tuple[Ordering, LengthScales]:
    """Return the reverse maximin ordering and length scales."""
    # O(n^2)
    n = len(x)
    indexes = np.zeros(n, dtype=np.int64)
    # minimum distance to a point in indexes at the time of each selection
    lengths = np.zeros(n)
    # arbitrarily select the first point
    if initial is None or initial.shape[0] == 0:
        k = 0
        # minimum distance to a point in indexes
        dists = euclidean(x, x[k : k + 1]).flatten()
        indexes[-1] = k
        lengths[-1] = np.inf
        start = n - 2
    # use the initial points
    else:
        dists = np.min(euclidean(x, initial), axis=1)
        start = n - 1

    for i in range(start, -1, -1):
        # select point with largest minimum distance
        k = np.argmax(dists)
        indexes[i] = k
        # update distances
        lengths[i] = dists[k]
        dists = np.minimum(dists, euclidean(x, x[k : k + 1]).flatten())

    return indexes, lengths


def reverse_maximin(
    x: Points, initial: Points | None = None
) -> tuple[Ordering, LengthScales]:
    """Return the reverse maximin ordering and length scales."""
    n = len(x)
    indexes = np.zeros(n, dtype=np.int64)
    # minimum distance to a point in indexes at the time of each selection
    lengths = np.zeros(n)
    # arbitrarily select the first point
    if initial is None or initial.shape[0] == 0:
        k = 0
        # minimum distance to a point in indexes
        dists = euclidean(x, x[k : k + 1]).flatten()
        indexes[-1] = k
        lengths[-1] = np.inf
        start = n - 2
    # use the initial points
    else:
        initial_tree = KDTree(initial)
        dists, _ = initial_tree.query(x)
        start = n - 1

    # initialize tree and heap
    tree = KDTree(x)
    heap = Heap(dists, np.arange(n))

    for i in range(start, -1, -1):
        # select point with largest minimum distance
        lk, k = heap.pop()
        indexes[i] = k
        # update distances
        lengths[i] = lk
        js = tree.query_ball_point(x[k], lk)
        dists = euclidean(x[js], x[k : k + 1])
        for index, j in enumerate(js):
            heap.decrease_key(j, dists[index])

    return indexes, lengths


def ball_reverse_maximin(
    x: Points, initial: Points | None = None
) -> tuple[Ordering, LengthScales]:
    """Return the reverse maximin ordering and length scales."""
    # O(n log^2 n rho^d)
    n = len(x)
    rho = 1
    indexes = np.zeros(n, dtype=np.int64)
    # minimum distance to a point in indexes at the time of each selection
    lengths = np.zeros(n)

    # arbitrarily select the first point
    if initial is None or initial.shape[0] == 0:
        start_k = 0
        dists = np.full(n, np.inf)
        dist_start_k = np.inf
    # use the initial points
    else:
        initial_tree = KDTree(initial)
        dists, _ = initial_tree.query(x)
        start_k = np.argmax(dists)
        dist_start_k = dists[start_k]

    indexes[-1] = start_k
    # minimum distance to a point in indexes
    dists = np.minimum(dists, euclidean(x, x[start_k : start_k + 1]).flatten())
    # guarantee that every index has at least one valid parent
    lengths[start_k] = np.inf
    parents = [[start_k] for i in range(n)]
    children = [[] for i in range(n)]
    # sort children by distance to parent
    children[start_k] = sorted(range(n), key=lambda j: dist(x[j], x[start_k]))
    # initialize heap
    heap = Heap(dists, np.arange(n))

    for i in range(n - 2, -1, -1):
        # select point with largest minimum distance
        lk, k = heap.pop()
        indexes[i] = k
        # update distances
        lengths[k] = lk
        # find closest parent p that covers k
        p = min(
            (
                j
                for j in parents[k]
                if dist(x[k], x[j]) + rho * lk <= rho * lengths[j]
            ),
            key=lambda j: dist(x[k], x[j]),
        )
        # go through children of p that could be close to k
        for j in children[p]:
            if dist(x[p], x[j]) <= dist(x[k], x[p]) + rho * lk:
                heap.decrease_key(j, dist(x[j], x[k]))
                if dist(x[j], x[k]) <= rho * lk:
                    children[k].append(j)
                    parents[j].append(k)
            else:
                break
        # sort children by distance to parent
        children[k].sort(key=lambda j: dist(x[j], x[k]))

    lengths[start_k] = dist_start_k
    length_scales = np.array([lengths[i] for i in indexes])
    return indexes, length_scales


def naive_sparsity(x: Points, lengths: LengthScales, rho: float) -> Sparsity:
    """Compute the sparity pattern given the ordered x."""
    # O(n^2)
    tree = KDTree(x)
    near = tree.query_ball_point(x, rho * lengths)
    return {i: [j for j in near[i] if j >= i] for i in range(len(x))}


def sparsity_pattern(x: Points, lengths: LengthScales, rho: float) -> Sparsity:
    """Compute the sparity pattern given the ordered x."""
    # O(n log^2 n + n s)
    tree, offset, length_scale = KDTree(x), 0, lengths[0]
    sparsity = {}
    for i in range(len(x)):
        # length scale doubled, re-build tree to remove old points
        if lengths[i] > 2 * length_scale:
            tree, offset, length_scale = KDTree(x[i:]), i, lengths[i]
        sparsity[i] = [
            offset + j
            for j in tree.query_ball_point(x[i], rho * lengths[i])
            if offset + j >= i
        ]

    return sparsity


def supernodes(
    sparsity: Sparsity, lengths: LengthScales, lambd: float
) -> tuple[Grouping, Sparsity]:
    """Aggregate indices into supernodes."""
    # O(n s)
    groups = []
    candidates = set(range(len(lengths)))
    agg_sparsity = {}
    i = 0
    while len(candidates) > 0:
        # remove minimum index that has not been aggregated
        while i not in candidates:
            i += 1
        group = sorted(
            j
            for j in sparsity[i]  # type: ignore
            if lengths[j] <= lambd * lengths[i] and j in candidates
        )
        groups.append(group)
        candidates -= set(group)
        # only store sparsity pattern for highest entry
        s = sorted({k for j in group for k in sparsity[j]})  # type: ignore
        agg_sparsity[group[0]] = s
        positions = {i: k for k, i in enumerate(s)}
        for j in group[1:]:
            # fill in blanks for rest to maintain proper number
            # np.empty is lazy (costs O(1) wrt to input size)
            agg_sparsity[j] = Empty(len(s) - positions[j])

    return groups, agg_sparsity


def supernodes_contiguous(
    sparsity: Sparsity, lengths: LengthScales, lambd: float
) -> tuple[Grouping, Sparsity]:
    """Aggregate indices into contiguous supernodes."""
    # O(n s)
    ref_groups, _ = supernodes(sparsity, lengths, lambd)
    groups = []
    agg_sparsity = {}
    i = 0
    while i < lengths.shape[0]:
        # take average group size: points left over groups left
        size = (lengths.shape[0] - i) // (len(ref_groups) - len(groups))
        group = list(range(i, i + size))
        groups.append(group)
        # only store sparsity pattern for highest entry
        s = sorted({k for j in group for k in sparsity[j]})  # type: ignore
        agg_sparsity[i] = s
        for k, j in enumerate(group[1:]):
            # fill in blanks for rest to maintain proper number
            agg_sparsity[j] = Empty(len(s) - 1 - k)
        i += size

    return groups, agg_sparsity
