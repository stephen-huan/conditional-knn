import cknn
import cholesky
import gp_kernels
from . import *

# make folders
ROOT = "experiments/recover"
os.makedirs(f"{ROOT}/data", exist_ok=True)
load_data = lambda *args, **kwargs: load_data__(*args, **kwargs, root=ROOT)
save_data = lambda *args, **kwargs: save_data__(*args, **kwargs, root=ROOT)
plot = lambda *args, **kwargs: plot__(*args, **kwargs, root=ROOT)

N = 2**10  # number of points
S = 2**5   # number of entries to pick
NOISE = 0  # noise level
# value filled in the diagonal of the Cholesky factor
# value must be sufficiently large to control
# the spectrum of the associated p.d. matrix!
DIAG_FILL = 10

TRIALS = 10 # number of trials
avg_results = lambda f, trials=TRIALS: avg_results__(f, trials)

GENERATE_N = True       # generate data for n
PLOT_N     = True       # plot data for n

GENERATE_S = True       # generate data for s
PLOT_S     = True       # plot data for s

GENERATE_NOISE = True   # generate data for noise
PLOT_NOISE = True       # plot data for noise

GENERATE_NOISE_S = True # generate data for noise
PLOT_NOISE_S = True     # plot data for noise

### experimental setup

def gen_noise(n: int, loc: float=0, scale: float=1) -> np.ndarray:
    """ Generate a nxn matrix of symmetric noise. """
    noise = np.zeros((n, n))
    # generate lower triangular
    ind = np.tril_indices(n, -1)
    noise[ind] += rng.normal(loc=loc, scale=scale, size=ind[0].shape[0])
    # copy to upper triangular
    noise += noise.T
    # generate diagonal
    noise[np.diag_indices(n)] = rng.normal(loc=loc, scale=scale, size=n)
    return noise

def setup() -> tuple:
    """ Generate a sparse Cholesky factor. """
    sparsity = [np.append(rng.choice(np.arange(i + 1, N),
                                     np.clip(S - 1, 0, N - i - 1),
                                     replace=False), i)
                for i in range(N)]
    col_ind = np.array([x for i in range(N) for x in [i]*len(sparsity[i])])
    row_ind = np.array([row for col in sparsity for row in col])
    data = rng.standard_normal(len(col_ind))
    factor = sparse.csc_matrix((data, (row_ind, col_ind)))

    L = factor.toarray()
    # maintain proper Cholesky factorization
    np.fill_diagonal(L, DIAG_FILL)

    return col_ind, row_ind, L

def recover(points: np.ndarray, kernel: Kernel, select) -> tuple:
    """ Reconstruct a sparse Cholesky factor given its covariance matrix. """
    # attempt to reconstruct
    sel_col_ind = []
    sel_row_ind = []
    for i in range(N):
        candidates = np.arange(i + 1, N)
        num = np.clip(S - 1, 0, N - i - 1)
        selected = select(points[candidates], points[i: i + 1], kernel, num)
        indices = [i] + list(candidates[selected])
        sel_col_ind += [i]*len(indices)
        sel_row_ind += indices

    return sel_col_ind, sel_row_ind

def test_recover(select, inv: bool=False) -> tuple:
    """ Test a selection method's ability to reconstruct a sparse factor. """
    # generate ground truth sparse Cholesky factor
    col_ind, row_ind, L = setup()

    # add symmetric noise
    std = np.sqrt(NOISE)
    theta = L@L.T
    theta += gen_noise(N, loc=0, scale=std)
    # print(sorted(np.linalg.eig(theta)[0])[:5])

    # give original matrix or inverse
    theta = cholesky.inv(theta) if inv else theta
    points, kernel = gp_kernels.matrix_kernel(theta)

    # attempt to recover sparsity pattern
    start = time.time()
    sel_col_ind, sel_row_ind = recover(points, kernel, select)
    recover_time = time.time() - start

    # compare recovered to ground truth
    indices = set(zip(col_ind, row_ind))
    sel_indices = set(zip(sel_col_ind, sel_row_ind))
    # |intersection|/|union|
    score = len(indices & sel_indices)/len(indices | sel_indices)

    # compute KL divergence of recovered factor
    sparsity = {i: [] for i in range(N)}
    for col, row in zip(sel_col_ind, sel_row_ind):
        sparsity[col].append(row)

    theta = cholesky.inv(L@L.T)
    points, kernel = gp_kernels.matrix_kernel(theta)
    factor = cholesky.__cholesky(points, kernel, sparsity)
    kl_div = cholesky.sparse_kl_div(factor, theta)

    return score, kl_div, recover_time

if __name__ == "__main__":
    methods = [
        ("random", silver, lambda: test_recover(cknn.random_select)),
        ("knn", lightblue, lambda: test_recover(cknn.knn_select)),
        ("corr", seagreen, lambda: test_recover(cknn.corr_select)),
        ("cknn", orange, lambda: test_recover(cknn.select, inv=True)),
    ]
    names, colors, funcs = zip(*methods)

    y = [
        ("score", "Accuracy (IOU)"),
        ("kl_div", "KL divergence"),
        ("time", "Time (seconds)"),
    ]
    y_names, y_labels = zip(*y)

    ### changing n

    data = [[[] for _ in range(len(funcs))] for _ in range(len(y))]
    scores, kl_div, times = data

    S = 2**5
    sizes = np.arange(1, 2**10, 1)

    # split region to make graph smoother
    # sizes = np.arange(1, 10)
    # sizes = np.append(sizes,  10*np.arange(1, 10))
    # sizes = np.append(sizes, 100*np.arange(1, 10))
    # sizes = np.sort(sizes)
    # sizes = sizes[sizes > 20]

    if GENERATE_N:
        for N in sizes:
            for i, f in enumerate(funcs):
                # reset random seed so all methods get the same seed
                rng = np.random.default_rng(1)
                for d, result in enumerate(avg_results(f)):
                    data[d][i].append(result)

                if d == len(y) - 1:
                    print(f"{N:5} {names[i]:12} {data[d][i][-1]:.3f}")

        save_data(data, sizes, "n", y_names, names)
    elif PLOT_N:
        data = load_data("n", y_names, names)
        scores, kl_div, times = data

    ## plot n to each y-axis parameter

    if PLOT_N:
        for y_data, y_name, y_label in zip(data, y_names, y_labels):

            def plot_callback():
                plt.title(f"{y_label.split()[0]} with increasing $N$ \
(s = {S})")
                plt.xlabel("$N$")
                plt.ylabel(y_label)

                if y_name == "kl_div":
                    plt.yscale("log")

            plot(sizes, y_data, names, colors, "n", y_name, plot_callback)

    ### changing s

    data = [[[] for _ in range(len(funcs))] for _ in range(len(y))]
    scores, kl_div, times = data

    # numerical error makes it hard to make this larger
    N = 2**8
    nums = np.arange(1, N, 1)

    # split region to make graph smoother
    # nums = 2**np.arange(8)
    # nums = np.append(nums, np.arange(0, 100, 10))
    # nums = np.append(nums, np.arange(100, N, 50))
    # nums = np.sort(nums)

    if GENERATE_S:
        for S in nums:
            for i, f in enumerate(funcs):
                # reset random seed so all methods get the same seed
                rng = np.random.default_rng(1)
                for d, result in enumerate(avg_results(f)):
                    data[d][i].append(result)

                if d == len(y) - 1:
                    print(f"{S:5} {names[i]:12} {data[d][i][-1]:.3f}")

        save_data(data, nums, "s", y_names, names)
    elif PLOT_S:
        data = load_data("s", y_names, names)
        scores, kl_div, times = data

    ## plot s to each y-axis parameter

    if PLOT_S:
        for y_data, y_name, y_label in zip(data, y_names, y_labels):

            def plot_callback():
                plt.title(f"{y_label.split()[0]} with increasing $s$ \
(N = {N})")
                plt.xlabel("$s$")
                plt.ylabel(y_label)

                if y_name == "kl_div":
                    plt.yscale("log")

            plot(nums, y_data, names, colors, "s", y_name, plot_callback)

    ### changing noise

    data = [[[] for _ in range(len(funcs))] for _ in range(len(y))]
    scores, kl_div, times = data

    N = 2**10
    S = 2**5
    DIAG_FILL = 10
    noises = np.linspace(0, 0.3, 10)

    if GENERATE_NOISE:
        for NOISE in noises:
            for i, f in enumerate(funcs):
                # reset random seed so all methods get the same seed
                rng = np.random.default_rng(1)
                for d, result in enumerate(avg_results(f)):
                    data[d][i].append(result)

                if d == len(y) - 1:
                    print(f"{NOISE:5} {names[i]:12} {data[d][i][-1]:.3f}")

        save_data(data, noises, "noise", y_names, names)
    elif PLOT_NOISE:
        data = load_data("noise", y_names, names)
        scores, kl_div, times = data

    ## plot noise to each y-axis parameter

    if PLOT_NOISE:
        for y_data, y_name, y_label in zip(data, y_names, y_labels):

            def plot_callback():
                plt.title(f"{y_label.split()[0]} with increasing $\sigma^2$ \
(N = {N}, S = {S})")
                plt.xlabel("$\sigma^2$")
                plt.ylabel(y_label)

                if y_name == "kl_div":
                    plt.yscale("log")

            plot(noises, y_data, names, colors, "noise", y_name, plot_callback)

    ### different noise with changing s

    noise_levels = [0, 0.01, 0.02, 0.04]
    methods = [
        ("cknn-0.00", orange, "solid",
         lambda: test_recover(cknn.select, inv=True)),
        ("cknn-0.01", orange, "dotted",
         lambda: test_recover(cknn.select, inv=True)),
        ("cknn-0.10", orange, "dashed",
         lambda: test_recover(cknn.select, inv=True)),
        ("cknn-0.20", orange, "dashdot",
         lambda: test_recover(cknn.select, inv=True)),
    ]
    names, colors, styles, funcs = zip(*methods)

    data = [[[] for _ in range(len(funcs))] for _ in range(len(y))]
    scores, kl_div, times = data

    N = 2**8
    nums = np.arange(1, N, 1)

    # split region to make graph smoother
    # nums = 2**np.arange(8)
    # nums = np.append(nums, np.arange(0, 100, 10))
    # nums = np.append(nums, np.arange(100, N, 50))
    # nums = np.sort(nums)

    if GENERATE_NOISE_S:
        for S in nums:
            for i, f in enumerate(funcs):
                NOISE = noise_levels[i]
                # reset random seed so all methods get the same seed
                rng = np.random.default_rng(1)
                for d, result in enumerate(avg_results(f)):
                    data[d][i].append(result)

        save_data(data, nums, "noise-s", y_names, names)
    elif PLOT_NOISE_S:
        data = load_data("noise-s", y_names, names)
        scores, times = data

    ## plot s to each y-axis parameter

    if PLOT_NOISE_S:
        for y_data, y_name, y_label in zip(data, y_names, y_labels):
            x, x_name = nums, "noise-s"
            for y, name, color, style in zip(y_data, names, colors, styles):
                plt.plot(x, y, label=name, color=color, linestyle=style)

            plt.title(f"{y_label.split()[0]} with increasing $s$ \
(N = {N})")
            plt.xlabel("$s$")
            plt.ylabel(y_label)

            if y_name == "kl_div":
                plt.yscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{ROOT}/{x_name}_{y_name}.png")
            plt.clf()

