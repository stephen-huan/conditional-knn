import time
from functools import partial

import numpy as np
import sklearn.gaussian_process.kernels as kernels
from sklearn.model_selection import train_test_split

from KoLesky import cholesky, cknn
from KoLesky import gp_regression as gp_regr
from KoLesky.gp_regression import coverage, grid, rmse
from KoLesky.typehints import CholeskyFactor, Empty, Kernel, Points, Select

# fmt: off
D = 3      # dimension of points
N = 2**11  # number of total points
N = 100
TTS = 0.1  # percentage of testing points

K = 10     # number of nearest neighbors
P = 1      # tuning parameter, maximin ordering robustness

TRIALS = 10**3
# fmt: on

# set random seed
rng = np.random.default_rng(1)


def get_sample(points: Points, kernel: Kernel) -> tuple:
    """Generate y labels from the features."""
    sample = gp_regr.sample_chol(rng, kernel(points))  # type: ignore
    y = sample(TRIALS).T
    # randomly split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        points, y, test_size=TTS, random_state=1
    )
    return X_train, X_test, y_train, y_test


def laGP(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    kernel: Kernel,
    s: int,
    select: Select = cknn.select,
) -> tuple:
    """Select and then directly apply Gaussian process regression."""
    n = X_test.shape[0]
    mu_pred = np.zeros((n, TRIALS))
    var_pred = np.zeros(n)
    for i in range(n):
        target = X_test[i : i + 1]
        indices = select(X_train, target, kernel, s)
        mu_pred[i, :], var_pred[i], _ = gp_regr.estimate(
            X_train, y_train, target, kernel, indices
        )
    return np.array(mu_pred), np.array(var_pred)


def chol(
    X_train: np.ndarray,
    X_test: np.ndarray,
    kernel: Kernel,
    s: int,
    p: int = 1,
    test_cond: bool = True,
) -> CholeskyFactor:
    """Gaussian process regression by approximation of joint covariance."""
    # ordering doesn't matter as we will use full candidates
    x = np.vstack((X_test, X_train))
    n, m = x.shape[0], X_test.shape[0]
    order = np.arange(n)
    # no aggregation
    groups = [[i] for i in range(n)]
    # reference sparsity is just used for number of nonzeros
    # additional nonzero since sparsity pattern includes diagonal
    ref_sparsity = {0: Empty((n - s) * (s + 1) + s * (s + 1) // 2)}
    # full candidate sparsity
    candidate_sparsity = (
        {i: np.arange(i + 1, n) for i in range(n)}
        if test_cond
        else {i: np.arange(max(i + 1, m), n) for i in range(n)}
    )
    L = cholesky.__cholesky_subsample(  # pyright: ignore
        x, kernel, ref_sparsity, candidate_sparsity, groups, cknn.select
    )
    return L, order


def test_regr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    regr,
) -> tuple:
    """Evaluate inverse Cholesky for Gaussian process regression."""
    start = time.time()
    mu_pred, var_pred, *_ = regr(X_train, y_train, X_test, kernel)
    time_regr = time.time() - start
    # ~np.sqrt(np.mean(var_pred)) because unbiased estimator
    loss = np.mean(rmse(y_test, mu_pred))
    emperical_coverage = np.mean(coverage(y_test, mu_pred, var_pred))

    return loss, emperical_coverage, time_regr


if __name__ == "__main__":
    kernel = kernels.Matern(length_scale=1, nu=1 / 2)

    points = grid(N, 0, 1, d=D)
    X_train, X_test, y_train, y_test = get_sample(points, kernel)

    inv_chol = partial(chol, s=K, p=P)
    methods = {
        "exact": gp_regr.estimate,
        "knn": partial(laGP, s=K, select=cknn.knn_select),
        "laGP": partial(laGP, s=K),
        "chol": partial(gp_regr.estimate_chol_joint, chol=inv_chol),
    }

    inv_chol_laGP = partial(chol, s=K, p=P, test_cond=False)
    chol_regr = partial(gp_regr.estimate_chol_joint, chol=inv_chol_laGP)

    mu_la, var_la, *_ = methods["laGP"](X_train, y_train, X_test, kernel)
    mu_chol, var_chol, *_ = chol_regr(X_train, y_train, X_test, kernel)
    assert np.allclose(mu_la, mu_chol), "laGP and chol not the same"

    for name, method in methods.items():
        loss, cov, time_regr = test_regr(
            X_train, y_train, X_test, y_test, method
        )
        print(f"{name:>5} loss: {loss:.3e} {cov:.2%} {time_regr:.3f}s")
