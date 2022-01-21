import numpy as np
import sklearn.gaussian_process.kernels as kernels
import cholesky

D = 3    # dimension of points
N = 100  # number of points
M = 5    # number of columns to aggregate
S = 20   # number of entries to pick

# display settings
np.set_printoptions(precision=3, suppress=True)

# set random seed
rng = np.random.default_rng(1)

if __name__ == "__main__":
    X = rng.random((N, D))
    # generate a symmetric positive definite matrix
    kernel = kernels.Matern(length_scale=1, nu=5/2)
    theta = kernel(X)
    theta_inv = cholesky.inv(theta)
    theta_det = cholesky.logdet(theta)

    # sensor placement methods

    indexes = cholesky.naive_entropy(X, kernel, S)
    answer = cholesky.entropy(X, kernel, S)
    assert indexes == answer, "entropy indexes mismatch"

    indexes = cholesky.naive_mi(X, kernel, S)
    answer = cholesky.mi(X, kernel, S)
    assert indexes == answer, "mutual information indexes mismatch"

    # Cholesky methods

    # single column with no aggregation
    L = cholesky.naive_cholesky(theta, S)
    L2 = cholesky.cholesky(theta, S)
    assert np.allclose(L.toarray(), L2.toarray()), "cholesky mismatch"

    # KL(theta, (L@L.T)^{-1}) = KL(L@L.T, theta^{-1})
    print(f"{cholesky.sparse_kl_div(L2) - theta_det:.3f}")

    # multiple column with aggregation

    # make M adjacent columns into the same group
    indexes = list(range(N))
    groups = [indexes[M*i: M*(i + 1)] for i in range(int(np.ceil(N/M)))]
    L = cholesky.naive_mult_cholesky(theta, S, groups)
    L2 = cholesky.cholesky(theta, S, groups)
    assert np.allclose(L.toarray(), L2.toarray()), "cholesky mismatch"

    print(f"{cholesky.sparse_kl_div(L2) - theta_det:.3f}")

