import numpy as np
import sklearn.gaussian_process.kernels as kernels

from KoLesky import gp_regression as gp_regr
from KoLesky.gp_regression import grid

# fmt: off
D = 3  # dimension of points
N = 14 # number of training points
M = 5  # number of prediction points
S = 2  # number of entries to pick

# number of samples for emperical covariance
TRIALS = 10**5
# allowed relative error
RTOL = 1e-1
# fmt: on

# display settings
np.set_printoptions(precision=3, suppress=True)

# set random seed
rng = np.random.default_rng(1)

if __name__ == "__main__":
    ### testing sampling methods

    # random points

    kernel = kernels.Matern(length_scale=1, nu=5 / 2)
    points = rng.random((N, D))
    true_cov: np.ndarray = kernel(points)  # type: ignore

    sample = gp_regr.sample(rng, true_cov)
    empirical_cov = gp_regr.empirical_covariance(sample, TRIALS)

    assert np.allclose(
        true_cov, empirical_cov, rtol=RTOL
    ), "numpy sample wrong on random"

    sample = gp_regr.sample_chol(rng, true_cov)
    empirical_cov = gp_regr.empirical_covariance(sample, TRIALS)

    assert np.allclose(
        true_cov, empirical_cov, rtol=RTOL
    ), "Cholesky sample wrong on random"

    # points on torus

    n, true_cov = gp_regr.torus(kernel, N, d=D)

    sample = gp_regr.sample(rng, true_cov)
    empirical_cov = gp_regr.empirical_covariance(sample, TRIALS)

    assert np.allclose(
        true_cov, empirical_cov, rtol=RTOL
    ), "numpy sample wrong on torus"

    sample = gp_regr.sample_chol(rng, true_cov)
    empirical_cov = gp_regr.empirical_covariance(sample, TRIALS)

    assert np.allclose(
        true_cov, empirical_cov, rtol=RTOL
    ), "Cholesky sample wrong on torus"

    sample = gp_regr.sample_circulant(rng, true_cov, n, D)
    empirical_cov = gp_regr.empirical_covariance(sample, TRIALS)

    assert np.allclose(
        true_cov, empirical_cov, rtol=RTOL
    ), "circulant sample wrong on torus"

    # points on unit grid

    points = grid(N, d=D)
    true_cov: np.ndarray = kernel(points)  # type: ignore

    sample = gp_regr.sample(rng, true_cov)
    empirical_cov = gp_regr.empirical_covariance(sample, TRIALS)

    assert np.allclose(
        true_cov, empirical_cov, rtol=RTOL
    ), "numpy sample wrong on grid"

    sample = gp_regr.sample_chol(rng, true_cov)
    empirical_cov = gp_regr.empirical_covariance(sample, TRIALS)

    assert np.allclose(
        true_cov, empirical_cov, rtol=RTOL
    ), "Cholesky sample wrong on grid"

    sample = gp_regr.sample_grid(rng, kernel, N, d=D)
    empirical_cov = gp_regr.empirical_covariance(sample, TRIALS)

    assert np.allclose(
        true_cov, empirical_cov, rtol=RTOL
    ), "grid sample wrong on grid"
