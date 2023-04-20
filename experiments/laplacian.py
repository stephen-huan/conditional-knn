import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

from KoLesky import cholesky, gp_kernels
from KoLesky.typehints import Matrix

from . import (
    Laplacian,
    avg_results__,
    darkorange,
    darkrust,
    get_laplacian,
    lightblue,
    load_data__,
    orange,
    plot__,
    rust,
    save_data__,
    seagreen,
)

# make folders
ROOT = "experiments/laplacian"
os.makedirs(f"{ROOT}/data", exist_ok=True)
load_data = lambda *args, **kwargs: load_data__(*args, **kwargs, root=ROOT)
save_data = lambda *args, **kwargs: save_data__(*args, **kwargs, root=ROOT)
plot = lambda *args, **kwargs: plot__(*args, **kwargs, root=ROOT)

DATASET = "iris_dataset_30NN"
# DATASET = "mice_10NN"

# fmt: off
RHO = 64     # tuning parameter, number of nonzero entries
S = 2        # tuning parameter, factor larger to make rho in subsampling
LAMBDA = 1.5 # tuning parameter, size of groups

GENERATE_RHO = True  # generate data for rho
PLOT_RHO     = True  # plot data for rho
# fmt: on

TRIALS = 1  # number of trials
avg_results = lambda f, trials=TRIALS: avg_results__(f, trials)

### Laplacian experiment


def setup_laplacian() -> tuple[Laplacian, Matrix]:
    """Generate a Laplacian and its Cholesky factor."""
    laplacian = get_laplacian(DATASET)
    # add multiple of identity to force p.d.
    laplacian += 1 * sparse.identity(laplacian.shape[0])
    L = np.linalg.cholesky(laplacian.toarray())

    return laplacian, L


def recover_lapacian(laplacian: sparse.coo_matrix, chol, *args) -> tuple:
    """Reconstruct a Laplacian given its inverse."""
    laplacian_inv = cholesky.inv(laplacian.toarray())
    points, kernel = gp_kernels.matrix_kernel(laplacian_inv)
    factor, order = chol(points, kernel, *args)

    return factor, order


def test_laplacian(chol, *args) -> tuple:
    """Test a selction method's ability to reconstruct a Laplacian."""
    laplacian, L = setup_laplacian()
    start = time.time()
    factor, order = recover_lapacian(laplacian, chol, *args)
    recover_time = time.time() - start

    # re-order laplacian
    laplacian = laplacian[np.ix_(order, order)]  # type: ignore

    # check condition of resulting preconditioner
    precond = sparse.linalg.spsolve_triangular(  # type: ignore
        factor.tocsr(), np.identity(factor.shape[0]), lower=True
    )
    precond = sparse.csc_matrix(precond)
    # cond = np.linalg.cond(laplacian.toarray())
    # M ~= L L^T, L^{-1} M L^{-T} ~= I
    laplacian_precond = precond @ laplacian @ precond.T
    precond_cond = np.linalg.cond(laplacian_precond.toarray())

    kl_div = cholesky.sparse_kl_div(factor, cholesky.inv(laplacian.toarray()))

    # recovery percentage
    row_ind, col_ind, _ = sparse.find(L)
    sel_row_ind, sel_col_ind, _ = sparse.find(factor)
    indices = set(zip(col_ind, row_ind))
    sel_indices = set(zip(sel_col_ind, sel_row_ind))
    # |intersection|/|union|
    score = len(indices & sel_indices) / len(indices | sel_indices)

    return kl_div, precond_cond, score, factor.nnz, recover_time


if __name__ == "__main__":
    methods = [
        ("KL", lightblue, lambda: test_laplacian(cholesky.cholesky_kl, RHO)),
        (
            "select",
            orange,
            lambda: test_laplacian(cholesky.cholesky_subsample, S, RHO),
        ),
        (
            "select-global",
            darkorange,
            lambda: test_laplacian(cholesky.cholesky_global, S, RHO),
        ),
        (
            "KL (agg)",
            seagreen,
            lambda: test_laplacian(cholesky.cholesky_kl, RHO, LAMBDA),
        ),
        (
            "select (agg)",
            rust,
            lambda: test_laplacian(
                cholesky.cholesky_subsample, S, RHO, LAMBDA
            ),
        ),
        (
            "select-global (agg)",
            darkrust,
            lambda: test_laplacian(cholesky.cholesky_global, S, RHO, LAMBDA),
        ),
    ]
    names, colors, funcs = zip(*methods)

    y = [
        ("kl_div", "KL divergence"),
        ("cond", "Condition Number"),
        ("score", "Accuracy (IOU)"),
        ("nnz", "nonzeros"),
        ("time", "Time (seconds)"),
    ]
    y_names, y_labels = zip(*y)

    ### changing rho

    data = [[[] for _ in range(len(funcs))] for _ in range(len(y))]
    kl_div, conds, scores, nnzs, times = data

    S = 2
    LAMBDA = 1.5

    if DATASET == "iris_dataset_30NN":
        rhos = np.arange(1, 64)
    elif DATASET == "mice_10NN":
        rhos = np.arange(1, 9)
    else:
        rhos = np.arange(1, 9)

    x_name = f"{DATASET}_rho"

    if GENERATE_RHO:
        for RHO in rhos:
            for i, f in enumerate(funcs):
                # reset random seed so all methods get the same seed
                rng = np.random.default_rng(1)
                for d, result in enumerate(avg_results(f)):
                    data[d][i].append(result)

                    if d == len(y) - 1:
                        print(f"{RHO:5} {names[i]:12} {data[d][i][-1]:.3f}")

        save_data(data, rhos, x_name, y_names, names)
    elif PLOT_RHO:
        data = load_data(x_name, y_names, names)
        kl_div, conds, scores, nnzs, times = data

    ## plot n to each y-axis parameter

    if PLOT_RHO:
        for y_data, y_name, y_label in zip(data, y_names, y_labels):

            def plot_callback():
                plt.title(
                    f"{y_label.split()[0]} with increasing $\\rho$ "
                    f"(dataset = {DATASET}, $s$ = {S}, $\\lambda$ = {LAMBDA})"
                )
                plt.xlabel("$\\rho$")
                plt.ylabel(y_label)

                if y_name == "kl_div":  # or y_name == "cond":
                    plt.yscale("log")

            plot(rhos, y_data, names, colors, x_name, y_name, plot_callback)
