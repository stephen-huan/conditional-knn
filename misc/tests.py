import os
import time

import numpy as np
import sklearn.gaussian_process.kernels as kernels

from KoLesky import cknn

# fmt: off
D = 3      # dimension of points
N = 2**12  # number of training points
M = 2**8   # number of testing points
S = 2**10  # number of points to select
# fmt: on

# set random seed
rng = np.random.default_rng(1)

root = "misc/data"
os.makedirs(root, exist_ok=True)

if __name__ == "__main__":
    kernel = kernels.Matern(length_scale=1, nu=1 / 2)
    train = rng.random((N, D))
    targets = rng.random((M, D))
    target = targets[0:1]
    points = np.vstack((train, targets))
    order = rng.permutation(np.arange(N + M))
    train_ind = order[:N]
    target_ind = order[N:]

    np.savetxt(f"{root}/train_points.csv", train)
    np.savetxt(f"{root}/target_points.csv", targets)
    np.savetxt(f"{root}/train_indices.csv", train_ind, fmt="%d")
    np.savetxt(f"{root}/target_indices.csv", target_ind, fmt="%d")

    start = time.time()
    indices = cknn.select(train, target, kernel, S)
    np.savetxt(f"{root}/select_single.csv", indices, fmt="%d")
    print(f"time: {time.time() - start:.3f}")

    start = time.time()
    indices = cknn.select(train, targets, kernel, S)
    np.savetxt(f"{root}/select_mult.csv", indices, fmt="%d")
    print(f"time: {time.time() - start:.3f}")

    start = time.time()
    indices = cknn.nonadj_select(points, train_ind, target_ind, kernel, S)
    np.savetxt(f"{root}/select_nonadj.csv", indices, fmt="%d")
    print(f"time: {time.time() - start:.3f}")

    start = time.time()
    indices = cknn.chol_select(points, train_ind, target_ind, kernel, S)
    np.savetxt(f"{root}/select_chol.csv", indices, fmt="%d")
    print(f"time: {time.time() - start:.3f}")
