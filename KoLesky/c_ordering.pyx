# cython: profile=False
cimport numpy as np

import numpy as np

from .maxheap cimport Heap

from .maxheap import Heap as pyHeap


cdef void __update_dists(
    Heap heap,
    double[:, ::1] dists,
    double[::1] dists_k,
    long[::1] js,
):
    """Update the distance table and heap."""
    cdef:
        int p, index, i, insert
        long j
        float d

    p = dists.shape[1]
    for index in range(js.shape[0]):
        j = js[index]
        d = dists_k[index]
        # insert d into dists[j], pushing out the largest value
        i = 0
        for i in range(p):
            if d <= dists[j, i]:
                break
        insert = i
        for i in range(p - 1, insert, -1):
            dists[j, i] = dists[j, i - 1]
        if insert < p:
            dists[j, insert] = d
        heap.__decrease_key(j, dists[j, p - 1])


def update_dists(
    heap: pyHeap,
    dists: np.ndarray,
    dists_k: np.ndarray,
    js: np.ndarray,
) -> None:
    """Python wrapper over __update_dists."""
    __update_dists(heap, dists, dists_k, js)
