import heapq
import time

import numpy as np

import maxheap

N = 10

# display settings
np.set_printoptions(precision=3, suppress=True)

# set random seed
rng = np.random.default_rng(1)

if __name__ == "__main__":
    dists, ids = rng.random(N), np.arange(N)
    heap = maxheap.Heap(dists, ids)
    print(heap)
    print(heap.pop())
    print(heap)
    heap.push(0.1, 1)
    print(heap)
    heap.push(1, 10)
    print(heap)
    print(heap.pop(), heap.pop(), heap.pop())
    print(heap)
    heap.update_key(0, 0.3)
    print(heap)
    print(repr(heap))

    N = 10**6
    dists, ids = rng.random(N), np.arange(N)
    l = list(zip(dists, ids))

    start = time.time()
    heap = maxheap.Heap(dists, ids)
    print(f"maxheap build: {time.time() - start:.3f}")

    start = time.time()
    while len(heap) > 0:
        heap.pop()
    print(f"maxheap   pop: {time.time() - start:.3f}")

    start = time.time()
    for i in range(N):
        heap.push(l[i][0], i)
    print(f"maxheap  push: {time.time() - start:.3f}")

    heap = list(l)
    start = time.time()
    heapq.heapify(heap)
    print(f" pyheap build: {time.time() - start:.3f}")

    start = time.time()
    while len(heap) > 0:
        heapq.heappop(heap)
    print(f" pyheap   pop: {time.time() - start:.3f}")

    start = time.time()
    for i in range(N):
        heapq.heappush(heap, l[i])
    print(f" pyheap  push: {time.time() - start:.3f}")
