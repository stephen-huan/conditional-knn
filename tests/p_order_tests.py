import time

import numpy as np

from KoLesky import ordering

# fmt: off
D = 3     # dimension of points
N = 100   # number of points
# N = int(1e5)
M = 10    # number of intial points
M = 0
R1 = 1.01 # tuning parameter, number of nonzero entries
R2 = 2    # tuning parameter, number of nonzero entries
eps = 0.01
L = 2     # tuning parameter, size of groups
P = 5     # p-length scale
# P = 1
K = 10    # k nonzero entries per column
# fmt: on

# display settings
# np.set_printoptions(precision=3, suppress=True)

# set random seed
rng = np.random.default_rng(1)

if __name__ == "__main__":
    points = rng.random((N, D))
    initial = rng.random((M, D))
    p = P

    start = time.time()
    ans_order, ans_lengths = ordering.naive_p_reverse_maximin(
        points, initial, p
    )
    print(f"  naive: {time.time() - start:.3f}")

    start = time.time()
    order, lengths = ordering.p_reverse_maximin(points, initial, p)
    print(f"kd-tree: {time.time() - start:.3f}")

    # print(ans_order)
    # print(order)
    #
    # print(ans_lengths)
    # print(lengths)

    assert (np.sort(ans_order) == np.arange(N)).all(), "naive order invalid"
    assert (np.sort(order) == np.arange(N)).all(), "kd-tree order invalid"

    assert np.allclose(ans_order, order), "kd-tree order mismatch"
    assert np.allclose(ans_lengths, lengths), "kd-tree lengths mismatch"

    # sparsity

    sparsity = ordering.sparsity_pattern(points[order], lengths, R1)
    # print(sparsity)

    assert ordering.is_k_sparsity(sparsity, p), "sparsity is not a p-sparsity"

    min_p = ordering.min_k_sparsity(points, R2, K, initial)
    order, lengths = ordering.p_reverse_maximin(points, initial, min_p)
    k_sparsity = ordering.sparsity_pattern(points[order], lengths, R2)
    assert ordering.is_k_sparsity(
        k_sparsity, K
    ), "sparsity is not a k-sparsity"

    # exit()

    N = int(1e5)

    points = rng.random((N, D))
    initial = rng.random((M, D))

    start = time.time()
    # order, lengths = ordering.reverse_maximin(points, initial)
    order, lengths = ordering.p_reverse_maximin(points, initial, p)
    print(f"kd-tree: {time.time() - start:.3f}")
