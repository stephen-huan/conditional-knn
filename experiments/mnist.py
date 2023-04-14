import scipy.stats
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import cknn

from . import *

# make folders
ROOT = "experiments/mnist"
os.makedirs(f"{ROOT}/data", exist_ok=True)
load_data = lambda *args, **kwargs: load_data__(*args, **kwargs, root=ROOT)
save_data = lambda *args, **kwargs: save_data__(*args, **kwargs, root=ROOT)
plot = lambda *args, **kwargs: plot__(*args, **kwargs, root=ROOT)

fnameX, fnamey = f"{ROOT}/X.npy", f"{ROOT}/y.npy"
# fmt: off
# POINTS = 1000      # number of data points to use
# SPLIT = 1/7        # train-test-split ratio
TRAIN_SIZE = 1000    # number of training points
TEST_SIZE = 100      # number of testing points
TRIALS = 100         # number of samples per point
K = 5                # number of training points to select
LENGTH_SCALE = 2**10 # length scale parameter of Matern kernel
NU = 3/2             # smoothness   parameter of Matern kernel

GENERATE_K = True    # generate data for k
PLOT_K     = True    # plot data for k

GENERATE_LS = True   # generate data for length scale
PLOT_LS     = True   # plot data for length scale
# fmt: on

### helper methods


def avg_results(f, trials: int = TRIALS) -> tuple:
    """Time a function trials times."""
    start = time.time()
    ys = []
    for i in range(trials):
        # re-sample every time for more robust accuracy estimate
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE, random_state=i
        )
        ys.append(accuracy_score(f(X_train, y_train, X_test), y_test))
    return (time.time() - start) / trials, np.average(ys)


### experimental setup


def knn(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int
) -> np.ndarray:
    """Classifies a list of points with k-th nearest neighbors (knn)."""
    kernel = cknn.euclidean  # equivalent to using Matern kernel
    return np.array(
        [
            scipy.stats.mode(
                y_train[cknn.knn_select(X_train, X_test[i : i + 1], kernel, k)]
            )[0]
            for i in range(len(X_test))
        ]
    )


def c_knn(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int
) -> np.ndarray:
    """Classifies a list of points with conditional knn."""
    kernel = kernels.Matern(length_scale=LENGTH_SCALE, nu=NU)
    return np.array(
        [
            scipy.stats.mode(
                y_train[cknn.select(X_train, X_test[i : i + 1], kernel, k)]
            )[0]
            for i in range(len(X_test))
        ]
    )


def scikit_knn(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int
) -> np.ndarray:
    """Classifies a list of points with scikit-learn's knn implementation."""
    # in high dimensions, brute force is much faster than ball/kd trees
    clf = KNeighborsClassifier(n_neighbors=k, algorithm="brute")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred


if __name__ == "__main__":
    # load mnist dataset
    if os.path.exists(fnameX) and os.path.exists(fnamey):
        # loading from numpy arrays is faster than fetch_openml,
        # even if fetch_openml has already downloaded the dataset
        X = np.load(fnameX, allow_pickle=True)
        y = np.load(fnamey, allow_pickle=True)
    else:
        # download mnist, caches to ~/scikit_learn_data
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        X, y = mnist["data"], mnist["target"]
        np.save(fnameX, X)
        np.save(fnamey, y)

    # X, y = X[:POINTS], y[:POINTS]
    # split 70,000 row dataset into train and test set, set seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE, random_state=1
    )

    ### plotting

    methods = [
        (
            "scikit-knn",
            lightblue,
            lambda X_train, y_train, X_test: scikit_knn(
                X_train, y_train, X_test, K
            ),
        ),
        (
            "conditional-knn",
            orange,
            lambda X_train, y_train, X_test: c_knn(
                X_train, y_train, X_test, K
            ),
        ),
    ]
    names, colors, funcs = zip(*methods)

    ys = [
        ("time", "Time (seconds)"),
        ("acc", "Accuracy (%)"),
    ]
    y_names, y_labels = zip(*ys)

    ### changing k

    data = [[[] for _ in range(len(funcs))] for _ in range(len(ys))]
    times, acc = data

    LENGTH_SCALE = 2**10
    ks = np.arange(1, 50)

    if GENERATE_K:
        for K in ks:
            print(K)
            for i, f in enumerate(funcs):
                for d, result in enumerate(avg_results(f)):
                    data[d][i].append(result)

        save_data(data, ks, "k", y_names, names)
        data = np.array(data)
    elif PLOT_K:
        data = load_data("k", y_names, names)
        times, accs = data

    ## plot k to each y-axis parameter

    if PLOT_K:
        for y_data, y_name, y_label in zip(data, y_names, y_labels):

            def plot_callback():
                plt.title(
                    f"{y_label.split()[0]} with increasing $k$ \
($n = {len(X_train)}, m = {len(X_test)}, l = {LENGTH_SCALE}, \\nu = {NU}$)"
                )
                plt.xlabel("$k$")
                plt.ylabel(y_label)

            k = 50
            plot(
                ks[:k],
                y_data[:, :k],
                names,
                colors,
                "k",
                y_name,
                plot_callback,
            )

    ### changing length scale

    data = [[[] for _ in range(len(funcs))] for _ in range(len(ys))]
    times, accs = data

    K = 5
    length_scales = 2 ** np.arange(0, 20)

    if GENERATE_LS:
        for l in length_scales:
            print(l)
            LENGTH_SCALE = l
            for i, f in enumerate(funcs):
                for d, result in enumerate(avg_results(f)):
                    data[d][i].append(result)

        save_data(data, length_scales, "l", y_names, names)
    elif PLOT_LS:
        data = load_data("l", y_names, names)
        times, accs = data

    ## plot length scale to each y-axis parameter

    if PLOT_LS:
        for y, name, color in zip(accs, names, colors):
            if name not in ["conditional-knn"]:
                continue
            plt.plot(length_scales, y, label=name, color=color)

        plt.title(
            f"Accuracy with increasing $l$ \
($n = {len(X_train)}, m = {len(X_test)}, k = {K}, \\nu = {NU}$)"
        )
        plt.xlabel("$l$")
        plt.xscale("log")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{ROOT}/l_acc.png")

    exit()

    ### experiments

    # sci-kit's knn
    start = time.time()
    y_pred = scikit_knn(X_train, y_train, X_test, K)
    print(f"accuracy: {accuracy_score(y_pred, y_test):.4f}")
    print(f"time: {time.time() - start:.3f}")

    # cknn implementation of knn
    start = time.time()
    y_pred = knn(X_train, y_train, X_test, K)
    print(f"accuracy: {accuracy_score(y_pred, y_test):.4f}")
    print(f"time: {time.time() - start:.3f}")

    # conditional knn
    start = time.time()
    y_pred = c_knn(X_train, y_train, X_test, K)
    print(f"accuracy: {accuracy_score(y_pred, y_test):.4f}")
    print(f"time: {time.time() - start:.3f}")
