import time

import numpy as np
import sklearn.gaussian_process.kernels as kernels

from KoLesky import linalg

D = 3  # dimension of points
N = 50  # number of points
M = 4_000  # number of points

# display settings
np.set_printoptions(precision=3, suppress=True)

# set random seed
rng = np.random.default_rng(1)

if __name__ == "__main__":
    X = rng.random((N, D))
    # generate a symmetric positive definite matrix
    kernel = kernels.Matern(length_scale=1, nu=5 / 2)
    theta: np.ndarray = kernel(X)  # type: ignore

    assert np.isclose(
        linalg.operator_norm(rng, theta), np.linalg.norm(theta, ord=2)
    ), "power method is not correct"

    X = rng.random((M, D))
    theta: np.ndarray = kernel(X)  # type: ignore

    start = time.time()
    ans = np.linalg.norm(theta, ord=2)
    print(f"{time.time() - start:.3f}")
    start = time.time()
    print(time.time() - start)
    start = time.time()
    ans2 = linalg.operator_norm(rng, theta)
    print(f"{time.time() - start:.3f}")
    assert np.isclose(ans, ans2), "power method is not correct"
