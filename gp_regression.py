from typing import Callable
import numpy as np
from numpy.fft import fftn, ifftn
import scipy.linalg
from sklearn.gaussian_process.kernels import Kernel

# number of samples to take for empirical covariane
TRIALS = 1000

Sample = Callable[[int], np.ndarray]

### helper methods

def rmse(u: np.ndarray, v: np.ndarray) -> float:
    """ Root mean squared error between u and v. """
    return np.sqrt(np.mean((u - v)**2))

def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ Solve the system Ax = b for symmetric positive definite A. """
    return scipy.linalg.solve(A, b, assume_a="pos")

### point generation methods

def grid(n: int, a: float=0, b: float=1, d: int=2) -> np.ndarray:
    """ Generate n points evenly spaced in a [a, b]^d hypercube. """
    spaced = np.linspace(a, b, round(n**(1/d)))
    cube = (spaced,)*d
    return np.stack(np.meshgrid(*cube), axis=-1).reshape(-1, d)

def __torus(kernel: Kernel, n: int,
            a: float=0, b: float=1, d: int=2, row: int=0) -> np.ndarray:
    """ Generate a row of the covariance matrix on a d-dimensional torus. """
    # make 3^d grid with original grid in the center
    points = grid(n, a, b, d)
    cube = (np.arange(-1, 2),)*d
    spaced, width = np.linspace(a, b, round(n**(1/d)), retstep=True)
    N = spaced.shape[0]**d

    shifts = np.stack(np.meshgrid(*cube), axis=-1).reshape(-1, d)
    copies = [points + (b - a + width)*shift for shift in shifts]

    # make covariance matrix taking into account copies
    theta = np.zeros(N)
    for j in range(N):
        cov = kernel(points[row], [grid[j] for grid in copies])
        theta[j] = np.max(cov)

    return spaced.shape[0], theta

def torus(kernel: Kernel, n: int,
          a: float=0, b: float=1, d: int=2) -> np.ndarray:
    """ Generate the covariance matrix on a d-dimensional torus. """
    size, _ = __torus(kernel, n, a, b, d)
    return size, np.vstack([__torus(kernel, n, a, b, d, row)[1]
                            for row in range(size**d)])

### sampling methods

def sample(rng: np.random.Generator, sigma: np.ndarray,
           mu: np.ndarray=None) -> Sample:
    """ Centered multivariate normal distribution with given covariance. """
    if mu is None:
        mu = np.zeros(sigma.shape[0])

    def sample(samples: int=1) -> np.ndarray:
        """ Return samples number of samples from the distribution. """
        y = rng.multivariate_normal(mu, sigma, samples)
        return y

    return sample

def sample_chol(rng: np.random.Generator, sigma: np.ndarray,
                mu: np.ndarray=None) -> Sample:
    """ Sample by Cholesky factorization. """
    n = sigma.shape[0]
    if mu is None:
        mu = np.zeros(n)

    # store Cholesky factor
    L = np.linalg.cholesky(sigma)

    def sample(samples: int=1) -> np.ndarray:
        """ Return samples number of samples from the distribution. """
        z = rng.standard_normal((n, samples))
        y = (L@z).T + mu
        return y

    return sample

def sample_circulant(rng: np.random.Generator, sigma: np.ndarray,
                     n: int, d: int=2, mu: np.ndarray=None) -> Sample:
    """ Sample by circulant embedding. """
    # base for block circulant matrix
    if sigma.ndim == 2:
        sigma = sigma[0]
    N = sigma.shape[0]
    if mu is None:
        mu = np.zeros(N)
    c = sigma.reshape((n,)*d)
    eigs = np.sqrt(fftn(c))

    def sample(samples: int=1) -> np.ndarray:
        """ Return samples number of samples from the distribution. """
        z = rng.standard_normal((samples, *(n,)*d))
        y = fftn(eigs*ifftn(z)).reshape((samples, N)) + mu
        return y.real

    return sample

def sample_grid(rng: np.random.Generator, kernel: Kernel,
                n: int, a: float=0, b: float=1, d: int=2,
                mu: np.ndarray=None) -> Sample:
    """ Sample from the hypercube by marginalization of torus. """
    spaced, width = np.linspace(a, b, int(n**(1/d)), retstep=True)
    n = spaced.shape[0]**d
    # make 3^d grid with original grid in the center
    N, sigma = __torus(kernel, (3**d)*n, 3*a - width, 3*b + width, d)
    sample_torus = sample_circulant(rng, sigma, N, d)

    if mu is None:
        mu = np.zeros(n)

    mask = []
    for i in range(N**d):
        coords = ((i//(N**np.arange(d - 1, -1, -1))) % N)//spaced.shape[0]
        if np.all(coords == 1):
            mask.append(i)
    mask = np.array(mask)

    def sample(samples: int=1) -> np.ndarray:
        """ Return samples number of samples from the distribution. """
        return sample_torus(samples)[:, mask] + mu

    return sample

def empirical_covariance(sample: Sample, trials: int=TRIALS) -> np.ndarray:
    """ Compute the empirical covariance from a sampling method. """
    X = sample(trials)
    # rows of X are observations, so covariance is X^T X
    return X.T@X/trials

### estimation methods

def __estimate(x_train: np.ndarray, y_train: np.ndarray,
               x_test: np.ndarray, kernel: Kernel) -> np.ndarray:
    """ Estimate y_test with direct Gaussian process regression. """
    # O(n^3 + n m^2)
    K_TT = kernel(x_train)
    K_PP = kernel(x_test)
    K_TP = kernel(x_train, x_test)
    K_PT_TT = solve(K_TT, K_TP).T

    mu_pred = K_PT_TT@y_train
    var_pred = K_PP - K_PT_TT@K_TP
    return mu_pred, var_pred

def estimate(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
             kernel: Kernel, indexes: list=slice(None)) -> np.ndarray:
    """ Estimate y_test according to the given sparsity pattern. """
    return __estimate(x_train[indexes], y_train[indexes], x_test, kernel)

