import copy
import os
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import sklearn.gaussian_process.kernels as kernels
from scipy.sparse.linalg import LinearOperator

from KoLesky import cholesky, hlibpro  # pyright: ignore
from KoLesky.linalg import operator_norm
from KoLesky.typehints import Kernel, Matrix, Ordering, Points, Sparse, Vector

from . import (
    avg_results__,
    darkorange,
    darkrust,
    lightblue,
    load_data__,
    orange,
    plot__,
    rust,
    save_data__,
    seagreen,
    silver,
)

# make folders
ROOT = "experiments/cg"
os.makedirs(f"{ROOT}/data", exist_ok=True)
load_data = lambda *args, **kwargs: load_data__(*args, **kwargs, root=ROOT)
save_data = lambda *args, **kwargs: save_data__(*args, **kwargs, root=ROOT)
plot = lambda *args, **kwargs: plot__(*args, **kwargs, root=ROOT)

# fmt: off
D = 3         # dimension of points
N = 2**10     # number of points
RHO = 4       # tuning parameter, number of nonzero entries
S = 2         # tuning parameter, factor larger to make rho in subsampling
LAMBDA = 1.5  # tuning parameter, size of groups
P = 2         # tuning parameter, maximin ordering robustness

KL = True     # compute true KL divergence (requires computing logdet)

MAX_ITERS = 10**5
RTOL = 10**-12 # relative tolerance for conjugate gradient
EPS = 1e-2     # precision on binary search

TRIALS = 1    # number of trials
avg_results = lambda f, trials=TRIALS: avg_results__(f, trials)

GENERATE_N = True  # generate data for n
PLOT_N     = True  # plot data for n

GENERATE_NNZ = True # generate data for nonzeros
PLOT_NNZ     = True # plot data for nonzeros

GENERATE_RHO = True # generate data for rho
PLOT_RHO     = True # plot data for rho
# fmt: on

iters = []


### conjugate gradient helper methods


def avg_iters(iters: list) -> Vector:
    """Return the average value along each slice of the list."""
    n = max(map(len, iters))
    out = np.zeros(n)
    for i in range(n):
        data = [iters[d][i] for d in range(len(iters)) if len(iters[d]) > i]
        out[i] = np.average(data)
    return out


def cholesky_linearoperator(L: Sparse) -> LinearOperator:
    """Return a LinearOperator object from a Cholesky factor."""

    def matvec(v: Vector) -> Vector:
        """Compute L (L^T v)"""
        return L.dot(L.T.dot(v))

    return LinearOperator(L.shape, matvec=matvec, rmatvec=matvec)


def permutation_linearoperator(
    linop: LinearOperator, order: Ordering
) -> LinearOperator:
    """Return a LinearOperator with re-ordering."""
    order_inv = cholesky.inv_order(order)

    def matvec(x: Vector) -> Vector:
        """Apply an ordering to x."""
        return linop(x[order_inv])[order]

    return LinearOperator(linop.shape, matvec=matvec, rmatvec=matvec)


def hlib_size(
    kernel: Kernel, points: Points, eps: float, inverse: bool = True
) -> int:
    """Return the size corresponding to the provided accuracy."""
    _, done, size = hlibpro.gram(kernel, points, inverse=inverse, eps=eps)
    done()
    return size


### experiment


def setup() -> tuple[
    Points,
    Kernel,
    tuple[LinearOperator, Callable[[], None]],
    Vector,
    Vector,
]:
    """Generate a matrix and a vector to solve."""
    kernel = kernels.Matern(length_scale=1, nu=1 / 2)
    points = rng.random((N, D))

    # multiply i.i.d. normal by covariance matrix to smoothen
    # this gives better results than generating the right hand side directly
    x = rng.standard_normal(N)
    if N <= 1 << 14:
        theta: Matrix = kernel(points)  # type: ignore
        linop = LinearOperator(
            (N, N), matvec=lambda x: theta @ x, rmatvec=lambda x: theta @ x
        )
        done = lambda: None
    else:
        matvec, done, _ = hlibpro.gram(kernel, points, eps=1e-6)
        linop = LinearOperator((N, N), matvec=matvec, rmatvec=matvec)
    y: np.ndarray = linop(x)  # type: ignore

    return points, kernel, (linop, done), x, y


def solve(
    theta: Matrix | LinearOperator,
    y: Vector,
    preconditioner: LinearOperator,
    callback: Callable[[Vector], None],
) -> tuple[Vector, Vector]:
    """Solve theta x = y by conjugate gradient with preconditioning."""
    iters.append([])
    x0 = np.zeros(N)
    xp, _ = sparse.linalg.cg(  # type: ignore
        theta,
        y,
        rtol=RTOL,
        atol=0,
        maxiter=MAX_ITERS,
        x0=x0,
        M=preconditioner,
        callback=callback,
    )
    # if len(iters[-1]) >= MAX_ITERS:
    #     exit("Conjugate gradient hit maximum number of iterations.")
    return xp, iters[-1]


def test_chol(
    chol, *args
) -> tuple[float, float, float, int, int, float, float, float]:
    """Runs a cg test with the Cholesky factorization method."""
    # generate matrix and right hand side
    points, kernel, (linop, done), x, y = setup()

    # compute preconditioner
    start = time.time()
    L, order = chol(points, kernel, *args, p=P)
    time_chol = time.time() - start

    preconditioner = cholesky_linearoperator(L)
    theta = permutation_linearoperator(linop, order)

    def cg_callback(xk: Vector) -> None:
        """Callback called by conjugate gradient after each iteration."""
        iters[-1].append(np.linalg.norm(x[order] - xk))

    # solve system with conjugate gradient
    start = time.time()
    xp, run = solve(theta, y[order], preconditioner, cg_callback)
    time_cg = time.time() - start

    if COMPUTE_OPNORM:
        diff = LinearOperator(
            (N, N),
            matvec=lambda x: x - theta(preconditioner(x)),
            rmatvec=lambda x: x - preconditioner(theta(x)),
        )
        op_norm = operator_norm(rng, diff, rtol=1e-2)
    else:
        op_norm = 0

    done()

    # kl_div = cholesky.sparse_kl_div(L, theta)
    kl_div = 1.0
    residual = np.linalg.norm(x[order] - xp)

    return (
        kl_div,
        op_norm,
        residual,  # type: ignore
        len(run),
        L.nnz,
        time_chol,
        time_cg,
        time_chol + time_cg,
    )


def test_hlib(
    rho: float | None = None, eps: float | None = None
) -> tuple[float, float, float, int, int, float, float, float]:
    """Runs a cg test with the Cholesky factorization method."""
    # generate matrix and right hand side
    points, kernel, (linop, done), x, y = setup()

    # compute preconditioner
    eps = eps if eps is not None else 10.0 ** (-(rho + 1.5))  # type: ignore
    start = time.time()
    inv_matvec, inv_done, size = hlibpro.gram(
        kernel, points, inverse=True, eps=eps
    )
    time_chol = time.time() - start

    preconditioner = LinearOperator(
        (N, N), matvec=inv_matvec, rmatvec=inv_matvec
    )
    theta = linop

    def cg_callback(xk: Vector) -> None:
        """Callback called by conjugate gradient after each iteration."""
        iters[-1].append(np.linalg.norm(x - xk))

    # solve system with conjugate gradient
    start = time.time()
    xp, run = solve(theta, y, preconditioner, cg_callback)
    time_cg = time.time() - start

    if COMPUTE_OPNORM:
        diff = LinearOperator(
            (N, N),
            matvec=lambda x: x - theta(preconditioner(x)),
            rmatvec=lambda x: x - preconditioner(theta(x)),
        )
        op_norm = operator_norm(rng, diff, rtol=1e-2)
    else:
        op_norm = 0

    inv_done()
    done()

    # kl_div = cholesky.sparse_kl_div(L, theta)
    kl_div = 1.0
    residual = np.linalg.norm(x - xp)

    return (
        kl_div,
        op_norm,
        residual,  # type: ignore
        len(run),
        size / 24,
        time_chol,
        time_cg,
        time_chol + time_cg,
    )


def binary_search(method, max_iters: int, eps: float = EPS) -> tuple:
    """Find rho such that conjugate gradient converges in < max_iters."""
    global MAX_ITERS, RHO, rng

    # cut off conjugate gradient if not converged
    MAX_ITERS = max_iters + 1

    # double rho until satisfiable
    RHO = 2
    rng_original = copy.deepcopy(rng)
    rng = copy.deepcopy(rng_original)
    iters, nnz = method()[3:5]
    right_nnz = nnz
    while iters > max_iters:
        RHO *= 2
        rng = copy.deepcopy(rng_original)
        iters, right_nnz = method()[3:5]

    # binary search on range
    left, right = 2, RHO
    left_nnz = nnz
    while (
        abs(left_nnz - right_nnz) > eps * right_nnz and abs(left - right) > eps
    ):
        RHO = (left + right) / 2
        rng = copy.deepcopy(rng_original)
        iters, nnz = method()[3:5]
        # rho too small, increase
        if iters > max_iters:
            left, left_nnz = RHO, nnz
        # rho too large, decrease
        else:
            right, right_nnz = RHO, nnz

    # extract results of final rho
    RHO = right
    rng = copy.deepcopy(rng_original)
    return method()


def binary_search_hlib(max_iters: int, tol: float = EPS) -> tuple:
    """Find rho such that conjugate gradient converges in < max_iters."""
    global MAX_ITERS, rng

    # cut off conjugate gradient if not converged
    MAX_ITERS = max_iters + 1

    # binary search from both sides of stable eps
    eps = 3e-6
    # double eps until satisfiable
    rng_original = copy.deepcopy(rng)
    rng = copy.deepcopy(rng_original)
    starting_iters, starting_size = test_hlib(eps=eps)[3:5]
    iters, right_size = starting_iters, starting_size
    right_eps = eps
    # upper bound of 1 is arbitrary
    while iters < max_iters and right_eps <= 1:
        right_eps *= 2
        rng = copy.deepcopy(rng_original)
        iters, right_size = test_hlib(eps=right_eps)[3:5]
    # halve eps until satisfiable
    iters, left_size = starting_iters, starting_size
    left_eps = eps
    while iters >= max_iters:
        left_eps /= 2
        rng = copy.deepcopy(rng_original)
        iters, left_size = test_hlib(eps=left_eps)[3:5]
    # binary search on range
    left, right = left_eps, right_eps
    while (
        abs(left_size - right_size) > tol * left_size
        and abs(left_size - right_size) > tol * right_size
        # and abs(left - right) > tol
    ):
        eps = (left + right) / 2
        rng = copy.deepcopy(rng_original)
        iters, size = test_hlib(eps=eps)[3:5]
        if iters < max_iters:
            left, left_size = eps, size
        else:
            right, right_size = eps, size

    # extract results of final acc
    eps = left
    rng = copy.deepcopy(rng_original)
    return test_hlib(eps=eps)


if __name__ == "__main__":
    methods = [
        ("KL", lightblue, lambda: test_chol(cholesky.cholesky_kl, RHO)),
        (
            "select",
            orange,
            lambda: test_chol(cholesky.cholesky_subsample, S, RHO),
        ),
        # (
        #     "select-global",
        #     darkorange,
        #     lambda: test_chol(cholesky.cholesky_global, S, RHO),
        # ),
        (
            "select-KNN",
            silver,
            lambda: test_chol(cholesky.cholesky_knn, RHO),
        ),
        (
            "KL (agg)",
            seagreen,
            lambda: test_chol(cholesky.cholesky_kl, RHO, LAMBDA),
        ),
        (
            "select (agg)",
            rust,
            lambda: test_chol(cholesky.cholesky_subsample, S, RHO, LAMBDA),
        ),
        # (
        #     "select-global (agg)",
        #     darkrust,
        #     lambda: test_chol(cholesky.cholesky_global, S, RHO, LAMBDA),
        # ),
        ("hlib", darkrust, lambda: test_hlib(RHO)),
    ]
    names, colors, funcs = zip(*methods)

    y = [
        ("kl_div", "KL divergence"),
        ("op_norm", "Operator norm"),
        ("res", "Residual ($\\ell_2$-norm)"),
        ("iter", "Iterations"),
        ("nnz", "Nonzeros"),
        ("time_chol", "Time for Cholesky factorization (seconds)"),
        ("time_cg", "Time for conjugate gradient (seconds)"),
        ("time_tot", "Total Wall-clock Time (seconds)"),
    ]
    y_names, y_labels = zip(*y)

    d = 0
    iter_data = []

    ### changing n

    COMPUTE_OPNORM = True
    data = [[[] for _ in range(len(funcs))] for _ in range(len(y))]
    kl_div, op_norm, res, num_iters, nnzs, times_chol, times_cg, times = data

    RHO = 4
    S = 2
    sizes = 2 ** np.arange(18)

    if GENERATE_N:
        for N in sizes:
            iter_data = []
            for i, f in enumerate(funcs):
                # reset random seed so all methods get the same seed
                rng = np.random.default_rng(1)
                for d, result in enumerate(avg_results(f)):
                    data[d][i].append(result)

                if d == len(y) - 1:
                    print(f"{N:5} {names[i]:12} {data[d][i][-1]:.3f}")

                iter_res = avg_iters(iters)
                iter_data.append(iter_res)
                iters = []

        save_data(data, sizes, "n", y_names, names)

        for i in range(len(iter_data)):
            fname = f"{ROOT}/data/n_iter-res_{names[i]}.csv"
            iter_res = np.array(iter_data[i])
            table = np.array([np.arange(len(iter_res)), iter_res]).T
            np.savetxt(fname, table, delimiter=" ")
    elif PLOT_N:
        data = load_data("n", y_names, names)
        kl_div, op_norm, res, num_iters, nnzs, times_chol, times_cg, times = (
            data
        )

        iter_data = [None for _ in range(len(names))]
        for i in range(len(names)):
            fname = f"{ROOT}/data/n_iter-res_{names[i]}.csv"
            # fmt: off
            iter_data[i] = ( # type: ignore
                np.loadtxt(fname, delimiter=" ")[:, 1]
            )
            # fmt: on

    ## plot n to each y-axis parameter

    if PLOT_N:
        for y_data, y_name, y_label in zip(data, y_names, y_labels):

            def plot_callback():
                plt.title(
                    f"{y_label.split()[0]} with increasing $N$ "
                    f"($\\rho$ = {RHO}, $s$ = {S}, $\\lambda$ = {LAMBDA})"
                )
                plt.xlabel("$N$")
                plt.ylabel(y_label)

                if y_name in ["kl_div", "res"]:
                    plt.yscale("log")

            plot(sizes, y_data, names, colors, "n", y_name, plot_callback)

    ## custom plots

    # number of nonzero entries per column
    if PLOT_N:
        for y_value, name, color in zip(data[4], names, colors):
            plt.plot(sizes, y_value / sizes, label=name, color=color)

        plt.title(
            f"Nonzero entries per column with increasing $N$ "
            f"($\\rho$ = {RHO}, $s$ = {S}, $\\lambda$ = {LAMBDA})"
        )
        plt.xlabel("$N$")
        plt.ylabel("Number of Nonzeros")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{ROOT}/n_nnz_col.png")
        plt.clf()

    # progress of conjugate gradient with iteration number
    if PLOT_N:
        for y_value, name, color in zip(iter_data, names, colors):
            plt.plot(
                np.arange(len(y_value)),  # type: ignore
                y_value,
                label=name,
                color=color,
            )

        plt.yscale("log")

        plt.title(
            f"Residual with Iterations "
            f"($\\rho$ = {RHO}, $s$ = {S}, $\\lambda$ = {LAMBDA})"
        )
        plt.xlabel("Iterations")
        plt.ylabel("Residual ($\\ell_2$-norm)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{ROOT}/n_iter-res.png")
        plt.clf()

    # number of nonzero entries per column to maintain constant iterations
    COMPUTE_OPNORM = False
    data = [[[] for _ in range(len(funcs))] for _ in range(len(y))]
    kl_div, op_norm, res, num_iters, nnzs, times_chol, times_cg, times = data

    old_max_iters = MAX_ITERS
    max_iters = 50
    if GENERATE_NNZ:
        for N in sizes:
            iter_data = []
            for i, f in enumerate(funcs):
                # reset random seed so all methods get the same seed
                rng = np.random.default_rng(1)
                g = (
                    (lambda: binary_search(f, max_iters))
                    if names[i] != "hlib"
                    else (lambda: binary_search_hlib(max_iters))
                )
                for d, result in enumerate(avg_results(g)):
                    data[d][i].append(result)

                if d == len(y) - 1:
                    print(f"{N:5} {names[i]:12} {data[d][i][-1]:.3f}")

                iter_res = avg_iters(iters)
                iter_data.append(iter_res)

        save_data(data, sizes, "nnz", y_names, names)
    elif PLOT_NNZ:
        data = load_data("nnz", y_names, names)
        kl_div, op_norm, res, num_iters, nnzs, times_chol, times_cg, times = (
            data
        )

    if PLOT_NNZ:
        for y_value, name, color in zip(data[4], names, colors):
            plt.plot(sizes, y_value / sizes, label=name, color=color)

        plt.title(
            "Nonzero entries per column with increasing $N$ "
            f"(iters = {max_iters}, $s$ = {S}, $\\lambda$ = {LAMBDA})"
        )
        plt.xlabel("$N$")
        plt.ylabel("Number of Nonzeros")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{ROOT}/nnz_const.png")
        plt.clf()

    MAX_ITERS = old_max_iters

    ### changing rho

    COMPUTE_OPNORM = True
    data = [[[] for _ in range(len(funcs))] for _ in range(len(y))]
    kl_div, op_norm, res, num_iters, nnzs, times_chol, times_cg, times = data

    N = 2**17
    S = 2
    rhos = np.arange(1, 5)

    if GENERATE_RHO:
        for RHO in rhos:
            for i, f in enumerate(funcs):
                # reset random seed so all methods get the same seed
                rng = np.random.default_rng(1)
                if "agg" in names[i] and RHO >= 6:
                    results = np.zeros(7)
                else:
                    results = avg_results(f)
                for d, result in enumerate(results):
                    data[d][i].append(result)

                if d == len(y) - 1:
                    print(f"{RHO:5} {names[i]:12} {data[d][i][-1]:.3f}")

                iters = []

        save_data(data, rhos, "rho", y_names, names)
    elif PLOT_RHO:
        data = load_data("rho", y_names, names)
        kl_div, op_norm, res, num_iters, nnzs, times_chol, times_cg, times = (
            data
        )

    ## plot rho to each y-axis parameter

    if PLOT_RHO:
        for y_data, y_name, y_label in zip(data, y_names, y_labels):

            def plot_callback():
                plt.title(
                    f"{y_label.split()[0]} with increasing $\\rho$ "
                    f"($N$ = {N}, $s$ = {S}, $\\lambda$ = {LAMBDA})"
                )
                plt.xlabel("$\\rho$")
                plt.ylabel(y_label)

                if (
                    y_name in ["kl_div", "op_norm", "res", "iter"]
                    or "time" in y_name
                ):
                    plt.yscale("log")

            x_data = rhos
            if y_name == "iter":
                # remove rho = 1 which skews the iteration plot
                # x_data, y_data = rhos[1:], y_data[:, 1:]
                ...

            plot(x_data, y_data, names, colors, "rho", y_name, plot_callback)

    hlibpro.finish()
