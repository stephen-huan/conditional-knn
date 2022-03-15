import numpy as np
import sklearn.gaussian_process.kernels as kernels
import gp_kernels
import cknn

D = 3    # dimension of points
N = 100  # number of training points
M = 5    # number of prediction points
S = 20   # number of entries to pick

# display settings
# np.set_printoptions(precision=3, suppress=True)

# set random seed
rng = np.random.default_rng(1)

if __name__ == "__main__":
    # generate input data and corresponding output
    # data matrix is each *row* is point, gram matrix X X^T
    X = rng.random((N, D))
    y = np.zeros(N)
    w = np.array([2, -1, -1])
    f = lambda x: w.dot(x.T)
    for i in range(N):
        y[i] = f(X[i])

    x_test = rng.random((M, D))
    y_test = np.zeros(M)
    for i in range(M):
        y_test[i] = f(x_test[i])

    # generate arbitrary p.d. matrix
    L = rng.random((N + M, N + M))
    index_points, matrix_kernel = gp_kernels.matrix_kernel(L@L.T)

    funcs = [(X, x_test, kernels.Matern(length_scale=0.5, nu=1/2)),
             (X, x_test, kernels.Matern(length_scale=0.9, nu=3/2)),
             (X, x_test, kernels.Matern(length_scale=0.1, nu=5/2)),
             (index_points[:N], index_points[N:], matrix_kernel),
            ]

    for X, x_test, kernel in funcs:
        if isinstance(kernel, kernels.Matern):
            params = kernel.get_params()
            print(f"Matern(nu={params['nu']}, \
length_scale={params['length_scale']})")
        else:
            print("Matrix kernel")
        # single point case
        answer = cknn.__naive_select(X, x_test[0:1], kernel, S)
        indexes = cknn.__prec_select(X, x_test[0:1], kernel, S)
        assert np.allclose(indexes, answer), "prec single indexes mismatch"
        indexes = cknn.__chol_select(X, x_test[0:1], kernel, S)
        assert np.allclose(indexes, answer), "chol single indexes mismatch"
        indexes = cknn.select(X, x_test[0:1], kernel, S)
        assert np.allclose(indexes, answer), "single select indexes mismatch"

        if isinstance(kernel, kernels.Matern):
            answer = cknn.knn_select(X, x_test[0:1], kernel, S)
            selected = cknn.knn_select(X, x_test[0:1], cknn.euclidean, S)
            assert np.allclose(selected, answer), "knn mismatch"

        # multiple point case
        answer = cknn.__naive_mult_select(X, x_test, kernel, S)
        indexes = cknn.__prec_mult_select(X, x_test, kernel, S)
        assert np.allclose(indexes, answer), "prec multiple indexes mismatch"
        indexes = cknn.__chol_mult_select(X, x_test, kernel, S)
        assert np.allclose(indexes, answer), "chol multiple indexes mismatch"
        indexes = cknn.select(X, x_test, kernel, S)
        assert np.allclose(indexes, answer), "mult select indexes mismatch"

        # predictions
        mu_pred, var_pred = cknn.estimate(X, y, x_test, kernel)
        print(y_test, mu_pred)
        print(cknn.logdet(var_pred))

        # approximate
        mu_pred, var_pred = cknn.estimate(X, y, x_test, kernel, indexes)
        print(y_test, mu_pred)
        print(cknn.logdet(var_pred))

        indexes = cknn.knn_select(X, x_test, kernel, S)
        mu_pred, var_pred = cknn.estimate(X, y, x_test, kernel, indexes)
        print(y_test, mu_pred)
        print(cknn.logdet(var_pred))

        print(answer)
        print(indexes)

        print()

