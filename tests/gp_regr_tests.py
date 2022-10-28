import time
import numpy as np
import sklearn.gaussian_process.kernels as kernels
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import cknn
import cholesky
import gp_regression as gp_regr
from gp_regression import estimate, grid, rmse, coverage

D = 3     # dimension of points
N = 2**11 # number of training points
M = 2**10 # number of prediction points
S = 2**5  # number of entries to pick

RHO = 2      # tuning parameter, number of nonzero entries
RHO_S = 2    # tuning parameter, factor larger to make rho in subsampling
LAMBDA = 1.5 # tuning parameter, size of groups

TRIALS = 10**3
RTOL = 1e-1

# display settings
np.set_printoptions(precision=3, suppress=True)

# set random seed
rng = np.random.default_rng(1)

if __name__ == "__main__":
    ### GP regression

    # generate all points together
    kernel = kernels.Matern(length_scale=1, nu=5/2)

    points = rng.random((N + M, D))
    # points = grid(N + M, 0, 1)

    # sample from covariance
    sample = gp_regr.sample_chol(rng, kernel(points))
    y = sample(TRIALS).T

    # randomly split into training and testing
    X_train, X_test, y_train, y_test = \
        train_test_split(points, y, train_size=N, test_size=M, random_state=1)

    # predictions

    print("exact")
    start = time.time()
    mu_pred, var_pred, det = estimate(X_train, y_train, X_test, kernel)
    print(f"    time: {time.time() - start:.3f}")
    true_loss = np.mean(np.log(rmse(y_test, mu_pred)))
    print(f"    loss: {true_loss:.3f}")
    print(f"  logdet: {det:.3f}")
    print(f"coverage: {np.mean(coverage(y_test, mu_pred, var_pred)):.3f}")
    print()

    ## selection

    print("approximate")
    indexes = cknn.select(X_train, X_test, kernel, S)
    mu_pred, var_pred, det = estimate(X_train, y_train, X_test,
                                      kernel, indexes)
    print(f"    time: {time.time() - start:.3f}")
    print(f"    loss: {np.mean(np.log(rmse(y_test, mu_pred))):.3f}")
    print(f"  logdet: {det:.3f}")
    print(f"coverage: {np.mean(coverage(y_test, mu_pred, var_pred)):.3f}")
    print()

    print("knn")
    start = time.time()
    indexes = cknn.knn_select(X_train, X_test, kernel, S)
    mu_pred, var_pred, det = estimate(X_train, y_train, X_test,
                                      kernel, indexes)
    print(f"    time: {time.time() - start:.3f}")
    print(f"    loss: {np.mean(np.log(rmse(y_test, mu_pred))):.3f}")
    print(f"  logdet: {det:.3f}")
    print(f"coverage: {np.mean(coverage(y_test, mu_pred, var_pred)):.3f}")
    print()

    ## Cholesky factorization

    # direct

    funcs = [
        ("inv chol", gp_regr.inv_chol),
        ("KL", lambda x, kernel: cholesky.cholesky_kl(x, kernel, RHO)),
        ("select",
         lambda x, kernel: cholesky.cholesky_subsample(x, kernel, RHO_S, RHO)),
        ("KL (agg)",
         lambda x, kernel: cholesky.cholesky_kl(x, kernel, RHO, LAMBDA)),
        ("select (agg)",
         lambda x, kernel: cholesky.cholesky_subsample(x, kernel,
                                                       RHO_S, RHO, LAMBDA),),
    ]

    for name, chol in funcs:
        print(f"direct {name}")
        start = time.time()
        mu_pred, var_pred, det, *L = \
            gp_regr.estimate_chol(X_train, y_train, X_test, kernel, chol=chol)
        loss = np.mean(np.log(rmse(y_test, mu_pred)))
        print(f"    time: {time.time() - start:.3f}")
        print(f"    loss: {np.mean(np.log(rmse(y_test, mu_pred))):.3f}")
        print(f"  logdet: {det:.3f}")
        print(f"coverage: {np.mean(coverage(y_test, mu_pred, var_pred)):.3f}")
        print()

        if name == "inv chol":
            assert np.isclose(true_loss, loss), "direct wrong"

    # joint

    funcs = [
        ("inv chol", gp_regr.joint_inv_chol),
        ("KL", lambda x_train, x_test, kernel: \
         cholesky.cholesky_joint(x_train, x_test, kernel, RHO)),
        ("select", lambda x_train, x_test, kernel: \
         cholesky.cholesky_joint_subsample(x_train, x_test, kernel,
                                           RHO_S, RHO)),
        ("KL (agg)", lambda x_train, x_test, kernel: \
         cholesky.cholesky_joint(x_train, x_test, kernel, RHO, LAMBDA)),
        ("select (agg)", lambda x_train, x_test, kernel: \
         cholesky.cholesky_joint_subsample(x_train, x_test, kernel,
                                           RHO_S, RHO, LAMBDA)),
    ]

    for name, chol in funcs:
        print(f"joint {name}")
        start = time.time()
        mu_pred, var_pred, det, *L = \
            gp_regr.estimate_chol_joint(X_train, y_train, X_test,
                                        kernel, chol=chol)
        loss = np.mean(np.log(rmse(y_test, mu_pred)))
        print(f"    time: {time.time() - start:.3f}")
        print(f"    loss: {loss:.3f}")
        print(f"  logdet: {det:.3f}")
        print(f"coverage: {np.mean(coverage(y_test, mu_pred, var_pred)):.3f}")
        print()

        if name == "inv chol":
            assert np.isclose(true_loss, loss), "joint wrong"

