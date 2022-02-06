import heapq
import numpy as np
import scipy.spatial.distance
from scipy.spatial import KDTree
from maxheap import Heap

# reverse maximum-minimum distance (reverse-maximin) ordering

def dist(x: np.ndarray, y: np.ndarray) -> float:
    """ Return the Euclidean distance between x and y. """
    # scipy.spatial.distance.euclidean is slow
    d = x - y
    return np.sqrt(d.dot(d))

def euclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ Return the distance between points in x and y. """
    return scipy.spatial.distance.cdist(x, y, "euclidean")

def naive_reverse_maximin(x: np.ndarray) -> tuple:
    """ Return the reverse maximin ordering and length scales. """
    # O(n^2)
    n = len(x)
    # add the first point arbitrarily
    indexes = np.zeros(n, dtype=np.int64)
    indexes[-1] = 0
    # minimum distance to a point in indexes at the time of each selection
    lengths = np.zeros(n)
    lengths[-1] = np.inf
    # minimum distance to a point in indexes
    dists = euclidean(x, x[0:1]).flatten()
    for i in range(n - 2, -1, -1):
        # select point with largest minimum distance
        k = np.argmax(dists)
        indexes[i] = k
        # update distances
        lengths[i] = dists[k]
        dists = np.minimum(dists, euclidean(x, x[k: k + 1]).flatten())

    return list(indexes), lengths

def reverse_maximin(x: np.ndarray) -> tuple:
    """ Return the reverse maximin ordering and length scales. """
    n = len(x)
    # add the first point arbitrarily
    indexes = np.zeros(n, dtype=np.int64)
    indexes[-1] = 0
    # minimum distance to a point in indexes at the time of each selection
    lengths = np.zeros(n)
    lengths[-1] = np.inf
    # initialize tree and heap
    tree = KDTree(x)
    dists = euclidean(x, x[0:1])
    heap = Heap([(dists[i], i) for i in range(n)])
    count = 0
    for i in range(n - 2, -1, -1):
        # select point with largest minimum distance
        lk, k = heap.pop()
        indexes[i] = k
        # update distances
        lengths[i] = lk
        js = tree.query_ball_point(x[k], lk)
        dists = euclidean(x[js], x[k: k + 1])
        count += len(js)
        for index, j in enumerate(js):
            heap.decrease_key(j, dists[index])

    return list(indexes), lengths

def ball_reverse_maximin(x: np.ndarray) -> tuple:
    """ Return the reverse maximin ordering and length scales. """
    # O(n log^2 n rho^d)
    n = len(x)
    rho = 1
    # add the first point arbitrarily
    indexes = np.zeros(n, dtype=np.int64)
    indexes[-1] = 0
    parents = [[0] for i in range(n)]
    children = [[] for i in range(n)]
    children[0] = list(range(n))
    # minimum distance to a point in indexes at the time of each selection
    lengths = np.zeros(n)
    lengths[0] = np.inf
    # initialize heap
    dists = euclidean(x, x[0:1]).flatten()
    heap = Heap([(dists[i], i) for i in range(n)])
    for i in range(n - 2, -1, -1):
        # select point with largest minimum distance
        lk, k = heap.pop()
        indexes[i] = k
        # update distances
        lengths[k] = lk
        # find closest parent p that covers k
        p = min((j for j in parents[k]
                 if dist(x[k], x[j]) + rho*lk <= rho*lengths[j]),
                key=lambda j: dist(x[k], x[j]))
        # go through children of p that could be close to k
        for j in children[p]:
            if dist(x[p], x[j]) <= dist(x[k], x[p]) + rho*lk:
                heap.decrease_key(j, dist(x[j], x[k]))
                if dist(x[j], x[k]) <= rho*lk:
                    children[k].append(j)
                    parents[j].append(k)
            else:
                break
        # sort children by distance to parent
        children[k].sort(key=lambda j: dist(x[j], x[k]))

    l = np.array([lengths[i] for i in indexes])
    return list(indexes), l

def naive_sparsity(x: np.ndarray, lengths: np.ndarray, rho: float) -> dict:
    """ Compute the sparity pattern given the ordered x. """
    # O(n^2)
    tree = KDTree(x)
    near = tree.query_ball_point(x, rho*lengths)
    return {i: [j for j in near[i] if j >= i] for i in range(len(x))}

def sparsity_pattern(x: np.ndarray, lengths: np.ndarray, rho: float) -> dict:
    """ Compute the sparity pattern given the ordered x. """
    # O(n log^2 n + n s)
    tree, offset, l = KDTree(x), 0, lengths[0]
    sparsity = {}
    for i in range(len(x)):
        # length scale doubled, re-build tree to remove old points
        if lengths[i] > 2*l:
            tree, offset, l = KDTree(x[i:]), i, lengths[i]
        sparsity[i] = [offset + j
                       for j in tree.query_ball_point(x[i], rho*lengths[i])
                       if offset + j >= i]

    return sparsity

def supernodes(sparsity: dict, lengths: np.ndarray, lambd: float) -> tuple:
    """ Aggregate indices into supernodes. """
    # O(n s)
    groups = []
    candidates = set(range(len(lengths)))
    agg_sparsity = {}
    i = 0
    while len(candidates) > 0:
        # remove minimum index that has not been aggregated
        while i not in candidates:
            i += 1
        group = sorted(j for j in sparsity[i] if
                       lengths[j] <= lambd*lengths[i] and j in candidates)
        groups.append(group)
        candidates -= set(group)
        # only store sparsity pattern for highest entry
        agg_sparsity[group[0]] = group + list(
            {k for j in group for k in sparsity[j] if k > group[-1]}
        )
        for j in range(1, len(group)):
            # fill in blanks for rest to maintain proper number
            # np.empty is lazy (costs O(1) wrt to input size)
            agg_sparsity[group[j]] = np.empty(len(agg_sparsity[group[0]]) - j)

    return groups, agg_sparsity

