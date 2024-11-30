import time

import numpy as np

from KoLesky import linalg

D = 3  # dimension of points
M = 1000
N = 2000

# display settings
np.set_printoptions(precision=3, suppress=True)

# set random seed
rng = np.random.default_rng(1)

if __name__ == "__main__":
    A = rng.random((M, N))
    assert np.isclose(
        linalg.operator_norm(rng, A), np.linalg.norm(A, ord=2)
    ), "power method is not correct"

    A = rng.random((N * 2, M * 2))
    start = time.time()
    ans = np.linalg.norm(A, ord=2)
    print(f"{time.time() - start:.3f}")
    start = time.time()
    ans2 = linalg.operator_norm(rng, A)
    print(f"{time.time() - start:.3f}")
    assert np.isclose(ans, ans2), "power method is not correct"
