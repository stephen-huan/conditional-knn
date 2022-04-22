import numpy as np
import ordering

D = 3    # dimension of points
N = 100  # number of points
M = 10   # number of intial points
R = 2    # tuning parameter, number of nonzero entries
L = 2    # tuning parameter, size of groups

# display settings
# np.set_printoptions(precision=3, suppress=True)

# set random seed
rng = np.random.default_rng(1)

if __name__ == "__main__":
    points = rng.random((N, D))
    initial = rng.random((M, D))

    ans_order, ans_lengths = ordering.naive_reverse_maximin(points, initial)
    ans_sparsity = ordering.naive_sparsity(points[ans_order], ans_lengths, R)
    groups = ordering.supernodes(ans_sparsity, ans_lengths, L)

    order, lengths = ordering.reverse_maximin(points, initial)
    sparsity = ordering.sparsity_pattern(points[order], lengths, R)
    sparsity = {i: sorted(children) for i, children in sparsity.items()}

    # print(groups)
    # print(ans_order, ans_lengths)
    # print(order, lengths)
    #
    # print(ans_sparsity)
    # print(sparsity)

    assert np.allclose(ans_order, order), "kd-tree order mismatch"
    assert np.allclose(ans_lengths, lengths), "kd-tree lengths mismatch"

    order, lengths = ordering.ball_reverse_maximin(points, initial)

    assert np.allclose(ans_order, order), "ball-tree order mismatch"
    assert np.allclose(ans_lengths, lengths), "ball-tree lengths mismatch"

    assert ans_sparsity == sparsity, "sparsity mismatch"

