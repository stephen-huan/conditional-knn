import numpy as np
import sklearn.gaussian_process.kernels as kernels

from KoLesky import cknn, gp_kernels
from KoLesky.gp_regression import estimate, sample

# fmt: off
D = 3    # dimension of points
N = 100  # number of training points
M = 5    # number of prediction points
S = 20   # number of entries to pick
# fmt: on

# display settings
# np.set_printoptions(precision=3, suppress=True)

# set random seed
rng = np.random.default_rng(1)

if __name__ == "__main__":
    # data matrix is each *row* is point, gram matrix X X^T
    X = rng.random((N + M, D))
    X_train, X_test = X[:N], X[N:]

    # generate arbitrary p.d. matrix
    L = rng.random((N + M, N + M))
    index_points, matrix_kernel = gp_kernels.matrix_kernel(L @ L.T)

    funcs = [
        (X_train, X_test, kernels.Matern(length_scale=0.5, nu=1 / 2)),
        (X_train, X_test, kernels.Matern(length_scale=0.9, nu=3 / 2)),
        (X_train, X_test, kernels.Matern(length_scale=0.1, nu=5 / 2)),
        (index_points[:N], index_points[N:], matrix_kernel),
    ]

    for X_train, X_test, kernel in funcs:
        # generate sample
        X = np.vstack((X_train, X_test))
        sampler = sample(rng, kernel(X))
        y = sampler().flatten()  # type: ignore
        y_train, y_test = y[:N], y[N:]

        if isinstance(kernel, kernels.Matern):
            params = kernel.get_params()
            print(
                f"Matern(nu={params['nu']}, "
                f"length_scale={params['length_scale']})"
            )
        else:
            print("Matrix kernel")
        # single point case
        answer = cknn.naive_select(X_train, X_test[0:1], kernel, S)
        indexes = cknn.prec_select(X_train, X_test[0:1], kernel, S)
        assert np.allclose(indexes, answer), "prec single indexes mismatch"
        indexes = cknn.chol_single_select(X_train, X_test[0:1], kernel, S)
        assert np.allclose(indexes, answer), "chol single indexes mismatch"
        indexes = cknn.select(X_train, X_test[0:1], kernel, S)
        assert np.allclose(indexes, answer), "single select indexes mismatch"

        if isinstance(kernel, kernels.Matern):
            answer = cknn.knn_select(X_train, X_test[0:1], kernel, S)
            selected = cknn.knn_select(
                X_train,
                X_test[0:1],
                cknn.euclidean,  # type: ignore
                S,
            )
            assert np.allclose(selected, answer), "knn mismatch"

        # multiple point case
        answer = cknn.naive_mult_select(X_train, X_test, kernel, S)
        indexes = cknn.prec_mult_select(X_train, X_test, kernel, S)
        assert np.allclose(indexes, answer), "prec multiple indexes mismatch"
        indexes = cknn.chol_mult_select(X_train, X_test, kernel, S)
        assert np.allclose(indexes, answer), "chol multiple indexes mismatch"
        indexes = cknn.select(X_train, X_test, kernel, S)
        assert np.allclose(indexes, answer), "mult select indexes mismatch"

        # predictions
        mu_pred, var_pred, det = estimate(X_train, y_train, X_test, kernel)
        print(y_test, mu_pred)
        print(det)

        # approximate
        mu_pred, var_pred, det = estimate(
            X_train, y_train, X_test, kernel, indexes
        )
        print(y_test, mu_pred)
        print(det)

        indexes = cknn.knn_select(X_train, X_test, kernel, S)
        mu_pred, var_pred, det = estimate(
            X_train, y_train, X_test, kernel, indexes
        )
        print(y_test, mu_pred)
        print(det)

        print(answer)
        print(indexes)

        print()
