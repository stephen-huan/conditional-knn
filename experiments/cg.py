import copy
import os
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import sklearn.gaussian_process.kernels as kernels
from pbbfmm3d import gram
from pbbfmm3d.kernels import Kernel as FMMKernel
from pbbfmm3d.kernels import from_sklearn
from scipy.sparse.linalg import LinearOperator

from KoLesky import cholesky
from KoLesky.typehints import Kernel, Matrix, Points, Sparse, Vector

from . import (
    avg_results__,
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
EPS = 1e-3     # precision on binary search

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


### experiment


def setup() -> tuple[Points, Kernel, FMMKernel, Vector, Vector]:
    """Generate a matrix and a vector to solve."""
    kernel = kernels.Matern(length_scale=1, nu=1 / 2)
    points = rng.random((N, D))
    fmm_kernel = from_sklearn(kernel)
    fmm_kernel.init(L=1, tree_level=4, interpolation_order=5, eps=1e-6)

    # multiply i.i.d. normal by covariance matrix to smoothen
    # this gives better results than generating the right hand side directly
    x = rng.standard_normal(N)
    if N <= 1 << 14:
        theta: Matrix = kernel(points)  # type: ignore
        y = theta @ x
    else:
        matvec = gram(fmm_kernel, points)
        y = matvec(x)

    return points, kernel, fmm_kernel, x, y


def solve(
    theta: Matrix | LinearOperator,
    y: Vector,
    linearop: LinearOperator,
    callback: Callable[[Vector], None],
) -> tuple[Vector, Vector]:
    """Solve theta x = y by conjugate gradient with preconditioner."""
    iters.append([])
    x0 = np.zeros(N)
    xp, _ = sparse.linalg.cg(  # type: ignore
        theta,
        y,
        rtol=RTOL,
        atol=0,
        maxiter=MAX_ITERS,
        x0=x0,
        M=linearop,
        callback=callback,
    )
    return xp, iters[-1]


def test_chol(
    chol, *args
) -> tuple[float, float, int, int, float, float, float]:
    """Runs a cg test with the Cholesky factorization method."""
    # generate matrix and right hand side
    points, kernel, fmm_kernel, x, y = setup()

    # compute preconditioner
    start = time.time()
    L, order = chol(points, kernel, *args, p=P)
    time_chol = time.time() - start

    linearop = cholesky_linearoperator(L)
    if N <= 1 << 14:
        theta = kernel(points[order])
    else:
        matvec = gram(fmm_kernel, points[order])
        theta = LinearOperator((N, N), matvec=matvec, rmatvec=matvec)

    def cg_callback(xk: Vector) -> None:
        """Callback called by conjugate gradient after each iteration."""
        iters[-1].append(np.linalg.norm(x[order] - xk))

    # solve system with conjugate gradient
    start = time.time()
    xp, run = solve(theta, y[order], linearop, cg_callback)
    time_cg = time.time() - start

    # kl_div = cholesky.sparse_kl_div(L, theta)
    kl_div = 1.0
    residual = np.linalg.norm(x[order] - xp)

    return (
        kl_div,
        residual,  # type: ignore
        len(run),
        L.nnz,
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
    while method()[2] > max_iters:
        RHO *= 2
        rng = copy.deepcopy(rng_original)

    # binary search on range
    left, right = 2, RHO
    while abs(left - right) > eps:
        RHO = (left + right) / 2
        rng = copy.deepcopy(rng_original)
        iters = method()[2]
        # rho too small, increase
        if iters > max_iters:
            left = RHO
        # rho too large, decrease
        else:
            right = RHO

    # extract results of final rho
    RHO = right
    rng = copy.deepcopy(rng_original)
    return method()


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
    ]
    names, colors, funcs = zip(*methods)

    y = [
        ("kl_div", "KL divergence"),
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

    data = [[[] for _ in range(len(funcs))] for _ in range(len(y))]
    kl_div, res, num_iters, nnzs, times_chol, times_cg, times = data

    RHO = 4
    S = 2
    sizes = 2 ** np.arange(17)

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
        kl_div, res, num_iters, nnzs, times_chol, times_cg, times = data

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
        for y_value, name, color in zip(data[3], names, colors):
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
    data = [[[] for _ in range(len(funcs))] for _ in range(len(y))]
    kl_div, res, num_iters, nnzs, times_chol, times_cg, times = data

    old_max_iters = MAX_ITERS
    max_iters = 50
    if GENERATE_NNZ:
        for N in sizes:
            iter_data = []
            for i, f in enumerate(funcs):
                # reset random seed so all methods get the same seed
                rng = np.random.default_rng(1)
                g = lambda: binary_search(f, max_iters)
                for d, result in enumerate(avg_results(g)):
                    data[d][i].append(result)

                if d == len(y) - 1:
                    print(f"{N:5} {names[i]:12} {data[d][i][-1]:.3f}")

                iter_res = avg_iters(iters)
                iter_data.append(iter_res)

        save_data(data, sizes, "nnz", y_names, names)
    elif PLOT_NNZ:
        data = load_data("nnz", y_names, names)
        kl_div, res, num_iters, nnzs, times_chol, times_cg, times = data

    if PLOT_NNZ:
        for y_value, name, color in zip(data[3], names, colors):
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

    data = [[[] for _ in range(len(funcs))] for _ in range(len(y))]
    kl_div, res, num_iters, nnzs, times_chol, times_cg, times = data

    N = 2**16
    S = 2
    rhos = np.arange(1, 9)

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
        kl_div, res, num_iters, nnzs, times_chol, times_cg, times = data

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

                if y_name in ["kl_div", "res", "iter"] or "time" in y_name:
                    plt.yscale("log")

            x_data = rhos
            if y_name == "iter":
                # remove rho = 1 which skews the iteration plot
                # x_data, y_data = rhos[1:], y_data[:, 1:]
                ...

            plot(x_data, y_data, names, colors, "rho", y_name, plot_callback)
