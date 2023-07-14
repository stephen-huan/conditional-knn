import matplotlib.pyplot as plt
import numpy as np
import sklearn.gaussian_process.kernels as kernels

from KoLesky import cknn
from KoLesky.typehints import Kernel, Points

N = 5  # number of training points
M = 1  # number of testing points
D = 1  # dimension

# trials
TRIALS = 1000

# set random seed
rng = np.random.default_rng(1)


def select(x_train: Points, x_test: Points, kernel: Kernel, s: int) -> tuple:
    """
    Greedily select the s entries minimizing conditional covariance.

    This is a (re-)implementation of cknn.select to keep track of statistics.
    """
    n, m = x_train.shape[0], x_test.shape[0]
    points = np.vstack((x_train, x_test))
    # initialization
    indexes = np.zeros(min(n, s), dtype=np.int64)
    candidates = list(range(n))
    factors = np.zeros((n, s))
    factors_pr = np.zeros((n + m, s + m))
    cond_var = kernel.diag(x_train)
    cond_var_pr = np.copy(cond_var)
    # pre-condition on the m prediction points
    for i in range(m):
        cov_k = kernel(points, [points[n + i]]).flatten()
        cknn.__chol_update(  # pyright: ignore
            cov_k, i, n + i, factors_pr, cond_var_pr
        )
    factors_pr = factors_pr[:n]

    information_gain = cond_var / cond_var_pr
    less, total = 0, 0
    for i in range(indexes.shape[0]):
        # pick best entry
        k = max(candidates, key=lambda j: cond_var[j] / cond_var_pr[j])
        indexes[i] = k
        candidates.remove(k)
        # update Cholesky factors
        cov_k = kernel(x_train, [x_train[k]]).flatten()
        cknn.__chol_update(cov_k, i, k, factors, cond_var)  # pyright: ignore
        cknn.__chol_update(  # pyright: ignore
            cov_k, i + m, k, factors_pr, cond_var_pr
        )
        # patch: check submodularity
        new_gain = np.copy(information_gain)
        new_gain[candidates] = cond_var[candidates] / cond_var_pr[candidates]
        less += np.sum(new_gain[candidates] <= information_gain[candidates])
        total += len(candidates)
        information_gain = new_gain

    return indexes, less, total


if __name__ == "__main__":
    kernel = kernels.Matern(length_scale=1, nu=5 / 2)

    num_less, num_total = 0, 0
    for _ in range(TRIALS):
        X_train = rng.random((N, D))
        X_test = rng.random((M, D))

        indexes, less, total = select(X_train, X_test, kernel, N)
        # assert np.allclose(
        #     indexes, cknn.select(X_train, X_test, kernel, N)
        # ), "methods don't agree"

        num_less += less
        num_total += total

    print(f"percent submodular: {num_less/num_total:.3%}")

    trials = 1000
    dims = np.arange(1, 32 + 1)
    y = []
    for dim in dims:
        num_less, num_total = 0, 0
        for _ in range(trials):
            X_train = rng.random((N, dim))
            X_test = rng.random((M, dim))

            _, less, total = select(X_train, X_test, kernel, N)
            num_less += less
            num_total += total
        y.append(num_less / num_total)

    plt.style.use("seaborn-whitegrid")
    lightblue = "#a1b4c7"

    plt.plot(dims, y, color=lightblue)
    plt.title("Submodularity percentage with dimension")
    plt.xlabel("Dimension")
    plt.ylabel("% submodular")
    plt.tight_layout()
    plt.savefig("misc/submodular.png")
    plt.clf()
