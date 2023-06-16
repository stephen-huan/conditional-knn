import os
import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn.gaussian_process.kernels as kernels

from KoLesky import cholesky
from KoLesky import gp_regression as gp_regr
from KoLesky import ordering
from KoLesky.typehints import InvChol, Kernel, Points

from . import (
    avg_results__,
    lightblue,
    load_data__,
    orange,
    plot__,
    rng,
    rust,
    save_data__,
    seagreen,
    silver,
)

# make folders
ROOT = "experiments/cholesky"
os.makedirs(f"{ROOT}/data", exist_ok=True)
load_data = lambda *args, **kwargs: load_data__(*args, **kwargs, root=ROOT)
save_data = lambda *args, **kwargs: save_data__(*args, **kwargs, root=ROOT)
plot = lambda *args, **kwargs: plot__(*args, **kwargs, root=ROOT)

# fmt: off
N = 2**16     # number of points
D = 2         # dimension of points
RHO = 2       # tuning parameter, number of nonzero entries
S = 2         # tuning parameter, factor larger to make rho in subsampling
LAMBDA = 1.5  # tuning parameter, size of groups
P = 1         # tuning parameter, maximin ordering robustness

KL = True     # compute true KL divergence (requires computing logdet)

# TRIALS = 10   # number of samples per point
TRIALS = 1    # number of samples per point
avg_results = lambda f, trials=TRIALS: avg_results__(f, trials)

GENERATE_N   = True  # generate data for n
PLOT_N       = True  # plot data for n

GENERATE_RHO = True  # generate data for rho
PLOT_RHO     = True  # plot data for rho

GENERATE_S   = True  # generate data for s
PLOT_S       = True  # plot data for s
# fmt: on

cache = {}

### experiment


def get_points(n: int, d: int) -> Points:
    """Return a n points of dimension d."""
    # return rng.random((n, d))
    width = 1 / (n ** (1 / d) - 1) if n ** (1 / d) > 1 else 0
    return gp_regr.perturbed_grid(rng, n, 0, 1, d, 1 / 3 * width)


def test_chol(
    points: Points,
    kernel: Kernel,
    logdet_theta: float,
    inv_chol: InvChol,
    *args,
) -> tuple[float, int, float]:
    """Test the Cholesky factorization."""
    reference = None
    if inv_chol == cholesky.cholesky_kl:
        order, lengths = ordering.p_reverse_maximin(points, p=P)
        x = points[order]
        key = args
        args = args if len(args) > 1 else (*args, None)
        sparsity, groups = cholesky.__cholesky_kl(  # pyright: ignore
            x, lengths, *args
        )
        cache[key] = sparsity, groups
    else:
        key = args if inv_chol == cholesky.cholesky_knn else args[1:]
        reference = cache[key]

    # compute Cholesky factorization
    start = time.time()
    L, _ = (
        inv_chol(points, kernel, *args, p=P)  # pyright: ignore
        if reference is None
        else inv_chol(
            points, kernel, *args, p=P, reference=reference  # pyright: ignore
        )
    )
    time_chol = time.time() - start

    kl_div = cholesky.sparse_kl_div(L, logdet_theta)

    return kl_div, L.nnz, time_chol


if __name__ == "__main__":
    kernel = kernels.Matern(length_scale=1, nu=5 / 2)

    ### plotting

    methods = [
        (
            "KL",
            lightblue,
            lambda: (cholesky.cholesky_kl, RHO),
        ),
        (
            "select",
            orange,
            lambda: (cholesky.cholesky_subsample, S, RHO),
        ),
        # (
        #     "select-global",
        #     darkorange,
        #     lambda: (cholesky.cholesky_global, S, RHO),
        # ),
        (
            "select-KNN",
            silver,
            lambda: (cholesky.cholesky_knn, RHO),
        ),
        (
            "KL (agg)",
            seagreen,
            lambda: (cholesky.cholesky_kl, RHO, LAMBDA),
        ),
        (
            "select (agg)",
            rust,
            lambda: (cholesky.cholesky_subsample, S, RHO, LAMBDA),
        ),
        # (
        #     "select-global (agg)",
        #     darkrust,
        #     lambda: (cholesky.cholesky_global, S, RHO, LAMBDA),
        # ),
    ]
    names, colors, funcs = zip(*methods)

    y = [
        ("kl_div", "KL divergence"),
        ("nnz", "Nonzeros"),
        ("time", "Time (seconds)"),
    ]
    y_names, y_labels = zip(*y)

    test = lambda inv_chol: test_chol(x, kernel, logdet_theta, *inv_chol())

    ### changing n

    data = [[[] for _ in range(len(funcs))] for _ in range(len(y))]
    kl_div, nonzeros, times = data

    sizes = 2 ** np.arange(round(np.log2(N)) + 1)

    if GENERATE_N:
        for n in sizes:
            x = get_points(n, D)
            logdet_theta = (
                cholesky.logdet(kernel(x)) if KL else 0  # type: ignore
            )
            for i, f in enumerate(funcs):
                for d, result in enumerate(avg_results(lambda: test(f))):
                    data[d][i].append(result)

                    if d == len(y) - 1:
                        print(f"{n:5} {names[i]:12} {data[d][i][-1]:.3f}")

        save_data(data, sizes, "n", y_names, names)
        data = np.array(data)
    elif PLOT_N:
        data = load_data("n", y_names, names)
        kl_div, nonzeros, times = data

    ## plot n to each y-axis parameter

    if PLOT_N:
        for y_data, y_name, y_label in zip(data, y_names, y_labels):

            def plot_callback():
                plt.title(
                    f"{y_label.split()[0]} with increasing $N$ "
                    f"($\\rho$ = {RHO}, $\\rho_s$ = {S}, "
                    f"$\\lambda$ = {LAMBDA})"
                )
                plt.xlabel("$N$")
                plt.ylabel(y_label)

            plot(sizes, y_data, names, colors, "n", y_name, plot_callback)

    ## custom plots

    # number of nonzeros to time
    if PLOT_N:
        for y_nnz, y_time, name, color in zip(nonzeros, times, names, colors):
            plt.plot(y_nnz, y_time, label=name, color=color)

        plt.title(
            f"Time with increasing $N$ against nonzeros "
            f"($\\rho$ = {RHO}, $\\rho_s$ = {S}, $\\lambda$ = {LAMBDA})"
        )
        plt.xlabel("Nonzeros")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{ROOT}/n_nnz_time.png")
        plt.clf()

    ### changing rho

    data = [[[] for _ in range(len(funcs))] for _ in range(len(y))]
    kl_div, nonzeros, times = data

    S = 2
    rhos = np.linspace(1, 8, 8)

    if GENERATE_RHO:
        x = get_points(N, D)
        logdet_theta = cholesky.logdet(kernel(x)) if KL else 0  # type: ignore
        for RHO in rhos:
            for i, f in enumerate(funcs):
                for d, result in enumerate(avg_results(lambda: test(f))):
                    data[d][i].append(result)

                    if d == len(y) - 1:
                        print(f"{RHO:5} {names[i]:12} {data[d][i][-1]:.3f}")

        save_data(data, rhos, "rho", y_names, names)
        data = np.array(data)
    elif PLOT_RHO:
        data = load_data("rho", y_names, names)
        kl_div, nonzeros, times = data

    ## plot rho to each y-axis parameter

    if PLOT_RHO:
        for y_data, y_name, y_label in zip(data, y_names, y_labels):

            def plot_callback():
                plt.title(
                    f"{y_label.split()[0]} with increasing $\\rho$ "
                    f"($N$ = {N}, $\\rho_s$ = {S}, $\\lambda$ = {LAMBDA})"
                )
                plt.xlabel("$\\rho$")
                plt.ylabel(y_label)

                if y_name == "kl_div" or y_name == "time":
                    plt.yscale("log")

            plot(rhos, y_data, names, colors, "rho", y_name, plot_callback)

    ## custom plots

    # time to accuracy
    if PLOT_RHO:
        for y_kl, y_time, name, color in zip(kl_div, times, names, colors):
            plt.plot(y_time, y_kl, label=name, color=color)

        plt.title(
            f"Accuracy with increasing $\\rho$ against time "
            f"($N = {N}, \\rho_s = {S}, \\lambda = {LAMBDA}$)"
        )
        plt.xlabel("Time (seconds)")
        plt.ylabel("KL divergence")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{ROOT}/rho_time_kl-div.png")
        plt.clf()

    ### changing s

    data = [[[] for _ in range(len(funcs))] for _ in range(len(y))]
    kl_div, nonzeros, times = data

    RHO = 4
    ss = np.linspace(1, 8, 8)

    if GENERATE_S:
        x = get_points(N, D)
        logdet_theta = cholesky.logdet(kernel(x)) if KL else 0  # type: ignore
        for S in ss:
            for i, f in enumerate(funcs):
                for d, result in enumerate(avg_results(lambda: test(f))):
                    data[d][i].append(result)

                    if d == len(y) - 1:
                        print(f"{S:5} {names[i]:12} {data[d][i][-1]:.3f}")

        save_data(data, ss, "s", y_names, names)
        data = np.array(data)
    elif PLOT_S:
        data = load_data("s", y_names, names)
        kl_div, nonzeros, times = data

    ## plot s to each y-axis parameter

    if PLOT_S:
        for y_data, y_name, y_label in zip(data, y_names, y_labels):

            def plot_callback():
                plt.title(
                    f"{y_label.split()[0]} with increasing $\\rho_s$ "
                    f"($N$ = {N}, $\\rho$ = {RHO}, $\\lambda$ = {LAMBDA})"
                )
                plt.xlabel("$\\rho_s$")
                plt.ylabel(y_label)

                if y_name == "kl_div" or y_name == "time":
                    plt.yscale("log")

            plot(ss, y_data, names, colors, "s", y_name, plot_callback)
