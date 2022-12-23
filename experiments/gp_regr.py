import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import cknn
import cholesky
import gp_regression as gp_regr
from gp_regression import estimate, grid, rmse, coverage
from ordering import euclidean
from . import *

DATASET = "sarcos"
ROOT = f"experiments/gp/{DATASET}"

fnamey = f"{ROOT}/y.npy"
fnameL = f"{ROOT}/L.npy"

# make folders
os.makedirs(f"{ROOT}/data", exist_ok=True)
load_data = lambda *args, **kwargs: load_data__(*args, **kwargs, root=ROOT)
save_data = lambda *args, **kwargs: save_data__(*args, **kwargs, root=ROOT)
plot = lambda *args, **kwargs: plot__(*args, **kwargs, root=ROOT)

D = 3         # dimension of points
# D = 6         # dimension of points
N = 2**16     # number of total points
# N = 2**10
TTS = 0.1     # percentage of testing points

RHO = 2       # tuning parameter, number of nonzero entries
RHO_S = 2     # tuning parameter, factor larger to make rho in subsampling
LAMBDA = 1.5  # tuning parameter, size of groups

GEN_Y = True  # force regenerate samples
GRAPH = False # visualize dataset

TRIALS = 10**3
RTOL = 1e-1

GENERATE = True      # generate data

GENERATE_RHO = True  # generate data for rho
PLOT_RHO     = True  # plot data for rho

GENERATE_S   = True  # generate data for s
PLOT_S       = True  # plot data for s

### graphing helper methods

def scatter_points(points: np.ndarray, ax1: int, ax2: int, **kwargs) -> None:
    """ Scatter plot the feature marginalized points. """
    plt.scatter(points[:, ax1], points[:, ax2], **kwargs)

def chol_get_selected(L: sparse.csc_matrix, order: np.ndarray,
                      index: int) -> np.ndarray:
    """ Return the selected points for a index. """
    loc = np.arange(order.shape[0])[order == index][0]
    rows, cols = L[:, loc].nonzero()
    return order[rows]

def remove_duplicates(points: np.ndarray, eps: float=1e-9) -> np.ndarray:
    """ Remove (near) duplicate points from the list of points. """
    new_points = [points[0]]
    for i in range(1, points.shape[0]):
        if np.min(euclidean(points[:i], points[i: i + 1])) > eps:
            new_points.append(points[i])
    return np.array(new_points)

### experimental setup

def get_dataset(dataset: str) -> tuple:
    """ Return a dataset (X_train, y_train, X_test, y_test) from the name. """
    # certain datasets come with both training data and validation set
    m = 0
    # certain datasets come with labels
    y = None

    if dataset == "grid":
        points = grid(N, 0, 1, d=D)
    elif dataset == "perturbed-grid":
        points = gp_regr.perturbed_grid(rng, N, 0, 1, D)
    elif dataset == "random":
        # random looks more "clumped" than one would intuitively suspect
        points = rng.random((N, D))
    elif dataset == "s-curve":
        points, t = datasets.make_s_curve(10000, random_state=1)
    elif dataset == "swiss-roll":
        points, t = datasets.make_swiss_roll(10000, random_state=1)
    elif dataset == "circles":
        points, y = datasets.make_circles(1000, random_state=1)
    elif dataset == "moons":
        points, y = datasets.make_moons(1000, random_state=1)
    elif dataset == "digits":
        data = datasets.load_digits()
        points = data.data/16
        points = points[:N]
    elif dataset == "sarcos":
        root = "datasets/gpml"
        X_train = io.loadmat(f"{root}/sarcos_inv.mat")["sarcos_inv"]
        X_test = io.loadmat(f"{root}/sarcos_inv_test.mat")["sarcos_inv_test"]
        # https://gaussianprocess.org/gpml/data/
        # first 21 variables input data, 22nd column target variable
        target = 21
        # M = round(N*TTS)
        M = X_test.shape[0]
        X_train, y_train = X_train[:N, :target], X_train[:N, target]
        X_test, y_test = X_test[:M, :target], X_test[:M, target]
        points = np.vstack((X_test, X_train))
        y = np.concatenate((y_test[:, np.newaxis], y_train[:, np.newaxis]))
        y = None
        m = X_test.shape[0]
        # it turns out most of the testing are an exact duplicate of training
        # go through normal train-test-split with just training data
        points = X_train
        y = None
        # y = y_train[:, np.newaxis]
        m = 0
    elif dataset == "shuttle":
        root = "datasets/uci/shuttle"
        X_train = np.loadtxt(f"{root}/shuttle.trn")
        X_test = np.loadtxt(f"{root}/shuttle.tst")
        X_train = X_train[:N]
        X_test = X_test[:2**7]
        points = np.vstack((X_test, X_train))
        m = X_test.shape[0]
    elif dataset == "susy":
        root = "datasets/uci/susy"
        points = np.loadtxt(f"{root}/SUSY.csv.gz",
                            delimiter=",",
                            unpack=True,
                            max_rows=N).T
    elif dataset == "cod-rna":
        root = "datasets/libsvm/cod-rna"
        col = lambda s: float(s.decode().split(":")[-1])
        points = np.loadtxt(f"{root}/cod-rna",
                            converters={i: col for i in range(1, 9)},
                            unpack=True,
                            max_rows=N).T
        points[:, 1] /= 1000
        points[:, 2] /= 1000
        points = points[:, 2:]
    else:
        raise ValueError(f"invalid dataset: {dataset}")

    return points, y, m

def get_sample(points: np.ndarray, y: np.ndarray, m: int,
               kernel: cknn.Kernel) -> tuple:
    """ Generate y labels from the features. """
    if y is not None:
        pass
    elif os.path.exists(fnamey) and not GEN_Y:
        y = np.load(fnamey)
    # sample from covariance
    else:
        sample = gp_regr.sample_chol(rng, kernel(points))
        # a bit broken because of the lack of positive-definiteness
        # sample = gp_regr.sample_grid(rng, kernel, N, 0, 1, d=D)

        # true_cov = kernel(points)
        # emperical_cov = gp_regr.empirical_covariance(sample, trials=10000)
        # print(true_cov)
        # print()
        # print(emperical_cov)
        # assert np.allclose(true_cov, emperical_cov, rtol=0.1)
        # exit()

        y = sample(TRIALS).T
        # np.save(fnamey, y)

    # if dataset in ["sarcos", "shuttle"]:
    if m != 0:
        # first m points are testing points
        X_test, X_train = points[:m], points[m:]
        y_test, y_train = y[:m], y[m:]
    else:
        # randomly split into training and testing
        X_train, X_test, y_train, y_test = \
            train_test_split(points, y, test_size=TTS, random_state=1)

    return X_train, X_test, y_train, y_test

def test_regr(X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray, inv_chol) -> tuple:
    """ Evaluate inverse Cholesky for Gaussian process regression. """
    start = time.time()
    mu_pred, var_pred, det, L = \
        gp_regr.estimate_chol_joint(X_train, y_train, X_test,
                                    kernel, chol=inv_chol)
    time_regr = time.time() - start
    loss = np.mean(rmse(y_test, mu_pred))
    emperical_coverage = np.mean(coverage(y_test, mu_pred, var_pred))

    nnz = L.nnz if hasattr(L, "nnz") else np.product(L.shape)
    return loss, det, emperical_coverage, time_regr, nnz

if __name__ == "__main__":
    ### GP regression

    kernel = kernels.Matern(length_scale=1, nu=5/2)

    if GENERATE:
        points, y, m = get_dataset(DATASET)
        points = points[:, :D]
        print(f"original number of points: {len(points)}")
        points = remove_duplicates(points)
        print(f" cleaned number of points: {len(points)}")
        # length_scales = np.array([np.var(points[m:, d]) for d in range(D)])
        length_scales = np.array(
            [np.max(points[m:, d]) - np.min(points[m:, d]) for d in range(D)]
        )
        kernel = kernels.Matern(length_scale=1, nu=3/2)
        # kernel = kernels.Matern(length_scale=1e0*length_scales, nu=3/2)
        # generate all points together
        X_train, X_test, y_train, y_test = get_sample(points, y, m, kernel)
        points = np.vstack((X_test, X_train))
        n = points.shape[0]

    # print(kernel(X_train))
    # y_pred = kernel(X_test, X_train)@y_train
    # print(y_test.flatten())
    # print(y_pred.flatten())
    # print(np.sum((y_test - y_pred)**2))
    # print(np.sum(kernel(X_test, X_train), axis=1))
    # print(length_scales)
    # exit()

    ### graph points

    if GRAPH:
        ax1, ax2 = 0, 1
        scatter = lambda points, **kwargs: scatter_points(points,
                                                          ax1, ax2, **kwargs)

        scatter(X_train, color=lightblue)
        scatter(X_test, color=orange)

        test_ind = 0
        test_point = np.ascontiguousarray(X_test[test_ind: test_ind + 1])
        s = 10

        sel1 = cknn.knn_select(points, test_point, kernel, s)

        temp = np.vstack((points[:test_ind], points[test_ind + 1:]))
        temp = np.ascontiguousarray(temp)
        sel2 = cknn.select(temp, test_point, kernel, s - 1)
        sel2[sel2 >= test_ind] += 1

        # factor with sparse Cholesky to get interactions
        L1, order1 = \
            cholesky.cholesky_joint(X_train, X_test, kernel, RHO)
        L2, order2 = \
            cholesky.cholesky_joint_subsample(X_train, X_test, kernel,
                                              RHO_S, RHO)

        # sel1 = chol_get_selected(L1, order1, test_ind)
        # sel2 = chol_get_selected(L2, order2, test_ind)
        # print(sorted(sel1))
        # print(sorted(sel2))

        scatter(test_point, s=100, zorder=1000, color=rust)
        scatter(points[sel1], s=400, zorder=10, color=silver)
        scatter(points[sel2], s=100, zorder=100, color=seagreen)

        plt.axis("equal")
        plt.show()
        exit()

    if GENERATE:
        if os.path.exists(fnameL) and not GEN_Y:
            Linv = np.load(fnameL, allow_pickle=True).item()
            order = np.arange(n)
        # compute factor of joint precision and store
        else:
            Linv, order = gp_regr.joint_inv_chol(X_train, X_test, kernel)
            # np.save(fnameL, Linv, allow_pickle=True)

    # joint

    test = lambda inv_chol: test_regr(X_train, y_train, X_test, y_test,
                                      inv_chol)

    methods = [
        ("KL", lightblue, lambda x_train, x_test, kernel: \
         cholesky.cholesky_joint(x_train, x_test, kernel, RHO)),
        ("select", orange, lambda x_train, x_test, kernel: \
         cholesky.cholesky_joint_subsample(x_train, x_test, kernel,
                                           RHO_S, RHO)),
        ("select-KNN", silver, lambda x_train, x_test, kernel: \
         cholesky.cholesky_joint_subsample(x_train, x_test, kernel,
                                           RHO_S, RHO,
                                           select=cknn.knn_select)),
        # ("select-global", darkorange, lambda x_train, x_test, kernel: \
        #  cholesky.cholesky_joint_global(x_train, x_test, kernel, RHO_S, RHO)),
        ("KL (agg)", seagreen, lambda x_train, x_test, kernel: \
         cholesky.cholesky_joint(x_train, x_test, kernel, RHO, LAMBDA)),
        ("select (agg)", rust, lambda x_train, x_test, kernel: \
         cholesky.cholesky_joint_subsample(x_train, x_test, kernel,
                                           RHO_S, RHO, LAMBDA)),
        # ("select-global (agg)", rust, lambda x_train, x_test, kernel: \
        #  cholesky.cholesky_joint_global(x_train, x_test, kernel,
        #                                 RHO_S, RHO, LAMBDA)),
        ("exact", silver, lambda x_train, x_test, kernel: (Linv, order)),
    ]

    names, colors, funcs = zip(*methods)

    y = [
        ("loss", "RMSE"),
        ("logdet", "Log determinant"),
        ("coverage", "Coverage"),
        ("time", "Time (seconds)"),
        ("nnz", "Nonzeros"),
    ]
    y_names, y_labels = zip(*y)


    ### changing rho

    data = [[[] for _ in range(len(funcs))] for _ in range(len(y))]
    losses, logdets, emperical_coverages, times, nnz = data

    rhos = np.arange(1, 9)

    if GENERATE_RHO:
        for RHO in rhos:
            for i, f in enumerate(funcs):
                # reset random seed so all methods get the same seed
                rng = np.random.default_rng(1)
                for d, result in enumerate(test(f)):
                    data[d][i].append(result)

                    if d == len(y) - 2:
                        print(f"{RHO} {names[i]:12} {data[d][i][-1]:.3f}")

        save_data(data, rhos, "rho", y_names, names)
        data = np.array(data)
        losses, logdets, emperical_coverages, times, nnz = data
    elif PLOT_RHO:
        data = load_data("rho", y_names, names)
        losses, logdets, emperical_coverages, times, nnz = data

    ## plot rho to each y-axis parameter

    if PLOT_RHO:
        for y_data, y_name, y_label in zip(data, y_names, y_labels):

            def plot_callback():
                plt.title(f"{y_label.split()[0]} with increasing $\\rho$ \
($N$ = {N}, $\\rho_s$ = {RHO_S}, $\\lambda$ = {LAMBDA})")
                plt.xlabel("$\\rho$")
                plt.ylabel(y_label)

                if y_name in ["loss", "logdet", "time"]:
                    plt.yscale("log")

            # graph difference to ground truth
            if y_name == "loss" or y_name == "logdet":
                y_data = y_data[:-1] - y_data[-1]

            plot(rhos, y_data, names, colors, "rho", y_name, plot_callback)

    ## custom plots

    # time to accuracy
    if PLOT_RHO:
        for y_det, y_time, name, color in zip(logdets, times, names, colors):
            if name == "exact":
                continue
            plt.plot(y_time, y_det - logdets[-1], label=name, color=color)

        plt.title(f"Log determinant with increasing $\\rho$ against time \
($N = {N}, \\rho_s = {RHO_S}, \\lambda = {LAMBDA}$)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Log determinant")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{ROOT}/rho_time_logdet.png")
        plt.clf()

        for y_loss, y_time, name, color in zip(losses, times, names, colors):
            if name == "exact":
                continue
            plt.plot(y_time, y_loss - losses[-1], label=name, color=color)

        plt.title(f"Loss with increasing $\\rho$ against time \
($N = {N}, \\rho_s = {RHO_S}, \\lambda = {LAMBDA}$)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Loss")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{ROOT}/rho_time_loss.png")
        plt.clf()

    ### changing s

    data = [[[] for _ in range(len(funcs))] for _ in range(len(y))]
    losses, logdets, emperical_coverages, times, nnz = data

    RHO = 2
    ss = np.arange(1, 9)

    if GENERATE_S:
        for RHO_S in ss:
            for i, f in enumerate(funcs):
                # reset random seed so all methods get the same seed
                rng = np.random.default_rng(1)
                for d, result in enumerate(test(f)):
                    data[d][i].append(result)

                    if d == len(y) - 2:
                        print(f"{RHO_S} {names[i]:12} {data[d][i][-1]:.3f}")

        save_data(data, rhos, "s", y_names, names)
        data = np.array(data)
    elif PLOT_S:
        data = load_data("s", y_names, names)
        losses, logdets, emperical_coverages, times, nnz = data

    ## plot rho to each y-axis parameter

    if PLOT_S:
        for y_data, y_name, y_label in zip(data, y_names, y_labels):

            def plot_callback():
                plt.title(f"{y_label.split()[0]} with increasing $s$ \
($N$ = {N}, $\\rho$ = {RHO}, $\\lambda$ = {LAMBDA})")
                plt.xlabel("$s$")
                plt.ylabel(y_label)

                if y_name in ["loss", "logdet", "time"]:
                    plt.yscale("log")

            # graph difference to ground truth
            if y_name == "loss" or y_name == "logdet":
                y_data = y_data[:-1] - y_data[-1]

            plot(rhos, y_data, names, colors, "s", y_name, plot_callback)

