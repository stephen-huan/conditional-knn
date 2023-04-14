from typing import Callable

import numpy as np
import scipy.linalg
import scipy.sparse as sparse
import scipy.stats as stats
from numpy.fft import fftn, ifftn
from sklearn.gaussian_process.kernels import Kernel

import cholesky
from cholesky import chol, inv_order, logdet, prec_logdet
from ordering import reverse_maximin

# number of samples to take for empirical covariane
TRIALS = 1000
# size of the symmetric confidence interval
CONFIDENCE = 0.9

Sample = Callable[[int], np.ndarray]

### helper methods


def rmse(u: np.ndarray, v: np.ndarray) -> float:
    """Root mean squared error between u and v."""
    return np.sqrt(np.mean((u - v) ** 2, axis=0))


def coverage(
    y_test: np.ndarray,
    mu_pred: np.ndarray,
    var_pred: np.ndarray,
    alpha: float = CONFIDENCE,
) -> float:
    """Emperical coverage of ground truth by predicted mean and variance."""
    std = np.sqrt(var_pred)
    # symmetric coverage centered around mean
    delta = stats.norm.ppf((1 + alpha) / 2) * std
    count = (mu_pred.T - delta <= y_test.T) & (y_test.T <= mu_pred.T + delta)
    return np.average(count, axis=1)


def inv_chol(x: np.ndarray, kernel: Kernel) -> tuple:
    """Cholesky factor for the precision, theta^{-1}."""
    n = x.shape[0]
    U = np.flip(chol(kernel(x[::-1])))
    Linv = scipy.linalg.solve_triangular(U, np.identity(n), lower=False).T
    return Linv, np.arange(n)


def joint_inv_chol(
    x_train: np.ndarray, x_test: np.ndarray, kernel: Kernel
) -> tuple:
    """Cholesky factor for the joint precision."""
    return inv_chol(np.vstack((x_test, x_train)), kernel)


def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the system Ax = b for symmetric positive definite A."""
    return scipy.linalg.solve(A, b, assume_a="pos")


def solve_triangular(
    L: sparse.csc_matrix, b: np.ndarray, lower: bool = True
) -> np.ndarray:
    """Solve the system Lx = b for sparse lower triangular L."""
    return (
        scipy.linalg.solve_triangular(L, b, lower=lower)
        if isinstance(L, np.ndarray)
        else sparse.linalg.spsolve_triangular(
            sparse.csr_matrix(L), b, lower=lower
        )
    )


### point generation methods


def grid(n: int, a: float = 0, b: float = 1, d: int = 2) -> np.ndarray:
    """Generate n points evenly spaced in a [a, b]^d hypercube."""
    spaced = np.linspace(a, b, round(n ** (1 / d)))
    cube = (spaced,) * d
    return np.stack(np.meshgrid(*cube), axis=-1).reshape(-1, d)


def perturbed_grid(
    rng: np.random.Generator,
    n: int,
    a: float = 0,
    b: float = 1,
    d: int = 2,
    delta: float = None,
) -> np.ndarray:
    """Generate n points roughly evenly spaced in a [a, b]^d hypercube."""
    points = grid(n, a, b, d)
    # compute level of perturbation as half width
    if delta is None:
        # one point, width ill-defined
        width = (b - a) / (n ** (1 / d) - 1) if n ** (1 / d) > 1 else 0
        delta = 1 / 2 * width
    return points + rng.uniform(-delta, delta, points.shape)


def __torus(
    kernel: Kernel,
    n: int,
    a: float = 0,
    b: float = 1,
    d: int = 2,
    row: int = 0,
) -> np.ndarray:
    """Generate a row of the covariance matrix on a d-dimensional torus."""
    # make 3^d grid with original grid in the center
    points = grid(n, a, b, d)
    cube = (np.arange(-1, 2),) * d
    spaced, width = np.linspace(a, b, round(n ** (1 / d)), retstep=True)
    N = spaced.shape[0] ** d

    shifts = np.stack(np.meshgrid(*cube), axis=-1).reshape(-1, d)
    copies = [points + (b - a + width) * shift for shift in shifts]

    # make covariance matrix taking into account copies
    theta = np.zeros(N)
    for j in range(N):
        cov = kernel(points[row], [grid[j] for grid in copies])
        theta[j] = np.max(cov)

    return spaced.shape[0], theta


def torus(
    kernel: Kernel, n: int, a: float = 0, b: float = 1, d: int = 2
) -> np.ndarray:
    """Generate the covariance matrix on a d-dimensional torus."""
    size, _ = __torus(kernel, n, a, b, d)
    return size, np.vstack(
        [__torus(kernel, n, a, b, d, row)[1] for row in range(size**d)]
    )


def sphere(
    rng: np.random.Generator,
    n: int,
    r: float = 1,
    d: int = 2,
    delta: float = None,
) -> np.ndarray:
    """Generate n points evenly spaced in a radius r hypersphere."""
    points = perturbed_grid(rng, n, a=-r, b=r, d=d, delta=delta)
    return np.array([point for point in points if np.linalg.norm(point) <= r])


def maximin(
    rng: np.random.Generator,
    k: int,
    n: int,
    a: float = 0,
    b: float = 1,
    d: int = 2,
    delta: float = None,
) -> np.ndarray:
    """Take the first n points from a maximin ordering of the unit grid."""
    points = perturbed_grid(rng, n, a, b, d, delta)
    order, _ = reverse_maximin(points)
    return points[order[::-1][:k]]


### sampling methods


def sample(
    rng: np.random.Generator, sigma: np.ndarray, mu: np.ndarray = None
) -> Sample:
    """Centered multivariate normal distribution with given covariance."""
    if mu is None:
        mu = np.zeros(sigma.shape[0])

    def sample(samples: int = 1) -> np.ndarray:
        """Return samples number of samples from the distribution."""
        y = rng.multivariate_normal(mu, sigma, samples)
        return y

    return sample


def sample_chol(
    rng: np.random.Generator, sigma: np.ndarray, mu: np.ndarray = None
) -> Sample:
    """Sample by Cholesky factorization."""
    n = sigma.shape[0]
    if mu is None:
        mu = np.zeros(n)

    # store Cholesky factor
    L = chol(sigma)

    def sample(samples: int = 1) -> np.ndarray:
        """Return samples number of samples from the distribution."""
        z = rng.standard_normal((n, samples))
        y = (L @ z).T + mu
        return y

    return sample


def sample_circulant(
    rng: np.random.Generator,
    sigma: np.ndarray,
    n: int,
    d: int = 2,
    mu: np.ndarray = None,
) -> Sample:
    """Sample by circulant embedding."""
    # base for block circulant matrix
    if sigma.ndim == 2:
        sigma = sigma[0]
    N = sigma.shape[0]
    if mu is None:
        mu = np.zeros(N)
    c = sigma.reshape((n,) * d)
    # force positive definiteness
    eigs = np.sqrt(fftn(c)).real

    def sample(samples: int = 1) -> np.ndarray:
        """Return samples number of samples from the distribution."""
        z = rng.standard_normal((samples, *(n,) * d))
        y = fftn(eigs * ifftn(z)).reshape((samples, N)) + mu
        return y.real

    return sample


def sample_grid(
    rng: np.random.Generator,
    kernel: Kernel,
    n: int,
    a: float = 0,
    b: float = 1,
    d: int = 2,
    mu: np.ndarray = None,
) -> Sample:
    """Sample from the hypercube by marginalization of torus."""
    spaced, width = np.linspace(a, b, round(n ** (1 / d)), retstep=True)
    n = spaced.shape[0] ** d
    # make 3^d grid with original grid in the center
    N, sigma = __torus(kernel, (3**d) * n, 3 * a - width, 3 * b + width, d)
    sample_torus = sample_circulant(rng, sigma, N, d)

    if mu is None:
        mu = np.zeros(n)

    mask = []
    for i in range(N**d):
        coords = ((i // (N ** np.arange(d - 1, -1, -1))) % N) // spaced.shape[
            0
        ]
        if np.all(coords == 1):
            mask.append(i)
    mask = np.array(mask)

    def sample(samples: int = 1) -> np.ndarray:
        """Return samples number of samples from the distribution."""
        return sample_torus(samples)[:, mask] + mu

    return sample


def empirical_covariance(sample: Sample, trials: int = TRIALS) -> np.ndarray:
    """Compute the empirical covariance from a sampling method."""
    X = sample(trials)
    # rows of X are observations, so covariance is X^T X
    return X.T @ X / trials


### estimation methods


def __estimate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    kernel: Kernel,
) -> tuple:
    """Estimate y_test with direct Gaussian process regression."""
    # O(n^3 + n m^2)
    K_TT = kernel(x_train)
    K_PP = kernel(x_test)
    K_TP = kernel(x_train, x_test)
    K_PT_TT = solve(K_TT, K_TP).T

    mu_pred = K_PT_TT @ y_train
    cov_pred = K_PP - K_PT_TT @ K_TP
    var_pred = kernel.diag(x_test) - np.sum(K_PT_TT.T * K_TP, axis=0)
    assert np.allclose(
        np.diag(cov_pred), var_pred
    ), "variance not diagonal of covariance"
    return mu_pred, var_pred, logdet(cov_pred)


def estimate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    kernel: Kernel,
    indexes: list = slice(None),
) -> tuple:
    """Estimate y_test according to the given sparsity pattern."""
    return __estimate(x_train[indexes], y_train[indexes], x_test, kernel)


def estimate_chol(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    kernel: Kernel,
    chol=inv_chol,
) -> tuple:
    """Estimate y_test with Cholesky factorization of training covariance."""
    L, order = chol(x_train, kernel)
    K_PT = kernel(x_test, x_train[order])
    L_P = L.T.dot(K_PT.T)

    mu_pred = K_PT @ L.dot(L.T.dot(y_train[order]))
    cov_pred = kernel(x_test) - L_P.T @ L_P
    var_pred = kernel.diag(x_test) - np.sum(L_P * L_P, axis=0)
    assert np.allclose(
        np.diag(cov_pred), var_pred
    ), "variance not diagonal of covariance"
    return mu_pred, var_pred, logdet(cov_pred), L


def estimate_chol_joint(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    kernel: Kernel,
    chol=inv_chol,
) -> tuple:
    """Estimate y_test with Cholesky factorization of joint covariance."""
    n, m = x_train.shape[0], x_test.shape[0]
    L, order = chol(x_train, x_test, kernel)
    inv_test_order, train_order = inv_order(order[:m]), order[m:] - m

    L11 = L[:m, :m]
    L21 = L[m:, :m]

    mu_pred = -solve_triangular(
        L11.T, L21.T.dot(y_train[train_order]), lower=False
    )
    # cov_pred = np.linalg.inv(L11.toarray()).T@np.linalg.inv(L11.toarray())
    e_i = solve_triangular(L11, np.identity(m), lower=True)
    var_pred = np.sum(e_i * e_i, axis=0)
    # assert np.allclose(np.diag(cov_pred), var_pred), \
    #     "variance not diagonal of covariance"
    return (
        mu_pred[inv_test_order],
        var_pred[inv_test_order],
        prec_logdet(L11),
        L,
    )
