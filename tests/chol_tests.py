import numpy as np
import sklearn.gaussian_process.kernels as kernels
import cholesky

D = 3    # dimension of points
N = 50   # number of points
M = 5    # number of columns to aggregate
S = 10   # number of entries to pick
R = 2    # tuning parameter, number of nonzero entries
G = 1.5  # tuning parameter, size of groups

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

    ### sensor placement methods

    indexes = cholesky.naive_entropy(X, kernel, S)
    answer = cholesky.entropy(X, kernel, S)
    assert indexes == answer, "entropy indexes mismatch"

    indexes = cholesky.naive_mi(X, kernel, S)
    answer = cholesky.mi(X, kernel, S)
    assert indexes == answer, "mutual information indexes mismatch"

    ### Cholesky methods

    ## single column with no aggregation

    L = cholesky.naive_cholesky(theta, S)
    L2 = cholesky.cholesky_select(X, kernel, S)
    assert np.allclose(L.toarray(), L2.toarray()), "single cholesky mismatch"

    # KL(theta, (L@L.T)^{-1}) = KL(L@L.T, theta^{-1})
    print(f"single kl div: {cholesky.sparse_kl_div(L2, theta_det):.3f}")
    print(f"nonzeros: {L2.nnz}")

    ## multiple columns with aggregation

    # make M adjacent columns into the same group
    indexes = list(range(N))
    groups = [indexes[M*i: M*(i + 1)] for i in range(int(np.ceil(N/M)))]
    L = cholesky.naive_mult_cholesky(theta, S, groups)
    L2 = cholesky.cholesky_select(X, kernel, S, groups)
    assert np.allclose(L.toarray(), L2.toarray()), "mult cholesky mismatch"

    print(f"mult kl div: {cholesky.sparse_kl_div(L2, theta_det):.3f}")
    print(f"nonzeros: {L2.nnz}")

    ## multiple non-adjacent columns with aggregation

    indexes = list(range(N))
    # make groups non-adjacent
    rng.shuffle(indexes)
    groups = [indexes[M*i: M*(i + 1)] for i in range(int(np.ceil(N/M)))]
    L = cholesky.naive_nonadj_cholesky(theta, S, groups)
    L2 = cholesky.cholesky_nonadj_select(X, kernel, S, groups)
    assert np.allclose(L.toarray(), L2.toarray()), \
        "non-adj mult cholesky mismatch"

    print(f"non-adj mult kl div: {cholesky.sparse_kl_div(L2, theta_det):.3f}")

    ### KL algorithm

    ## single column with no aggregation

    L, order = cholesky.naive_cholesky_kl(X, kernel, R)
    L2, order2 = cholesky.cholesky_kl(X, kernel, R)
    assert np.allclose(order, order2), "ordering mismatch"
    assert np.allclose(L.toarray(), L2.toarray()), "kl cholesky mismatch"

    # determinant unchanged after permutation of matrix
    print(f"kl kl div: {cholesky.sparse_kl_div(L2, theta_det):.3f}")
    print(f"nonzeros: {L2.nnz}")

    ## multiple columns with aggregation

    L, order = cholesky.naive_cholesky_kl(X, kernel, R, G)
    L2, order2 = cholesky.cholesky_kl(X, kernel, R, G)
    assert np.allclose(order, order2), "ordering mismatch"
    assert np.allclose(L.toarray(), L2.toarray()), "mult kl cholesky mismatch"

    print(f"mult kl kl div: {cholesky.sparse_kl_div(L2, theta_det):.3f}")
    print(f"nonzeros: {L2.nnz}")

    ## subsampling

    L2, order = cholesky.cholesky_subsample(X, kernel, 2, R, G)

    print(f"sample kl div: {cholesky.sparse_kl_div(L2, theta_det):.3f}")
    print(f"nonzeros: {L2.nnz}")

