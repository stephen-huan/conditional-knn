import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn.gaussian_process.kernels as kernels

from KoLesky import cknn
from KoLesky import gp_regression as gp_regr
from KoLesky.typehints import Indices, Kernel, Matrix, Points, Vector

from . import lightblue, orange, rng, rust, save_1d__, save__, seagreen, silver

ROOT = "figures/screen_select"
# make folders
os.makedirs(f"{ROOT}/data", exist_ok=True)
save = lambda *args, **kwargs: save__(*args, **kwargs, root=ROOT)
save_1d = lambda *args, **kwargs: save_1d__(*args, **kwargs, root=ROOT)

# fmt: off
D = 2    # dimension of points
R = 1    # range
L = 1    # length scale
NU1, NU2 = 5, 2 # nu = NU1/NU2

POINT_SIZE = 20 # point sizes
BIG_POINT  = 40 # large point
# fmt: on


def estimate(
    x_train: Points,
    y_train: Points,
    x_test: Points,
    kernel: Kernel,
    indexes: Indices | list[int] | slice = slice(None),
) -> tuple[Vector, Matrix]:
    """Estimate y_test with direct Gaussian process regression."""
    # O(n^3 + n m^2)
    x_train, y_train = x_train[indexes], y_train[indexes]
    K_TT = kernel(x_train)
    K_PP = kernel(x_test)
    K_TP = kernel(x_train, x_test)
    K_PT_TT = gp_regr.solve(K_TT, K_TP).T

    mu_pred = K_PT_TT @ y_train
    cov_pred = K_PP - K_PT_TT @ K_TP
    return mu_pred, cov_pred


if __name__ == "__main__":
    colors = [lightblue, orange, silver, seagreen, rust]

    # high smoothness has some interesting (but complex) structure
    kernel = kernels.Matern(length_scale=1, nu=1 / 2)

    ## kernel function

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(121, projection="3d")

    v = np.zeros((1, D))
    n = 21
    x = gp_regr.grid(n**2, -1, 1)
    s = 0.25
    points = np.array([[s, s], [s, -s], [-s, s], [-s, -s]])

    corr = lambda x, v: kernel(x, v).flatten() / np.sqrt(  # type: ignore
        np.diagonal(kernel(x)) * kernel(v).flatten()
    )

    points_corr, x_corr = corr(points, v), corr(x, v)

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points_corr,
        label="points",
        zorder=3,
        s=BIG_POINT,
        color=colors[1],
    )
    surf = ax.plot_wireframe(
        x[:, 0].reshape(n, n),
        x[:, 1].reshape(n, n),
        x_corr.reshape(n, n),
        color=colors[0],
        alpha=0.8,
    )

    ax.set_zlim3d(-0.1, 1)

    save(
        "matern_uncond_points.csv",
        1,
        (
            points[:, 0].flatten(),
            points[:, 1].flatten(),
            points_corr.flatten(),
        ),
    )
    save(
        "matern_uncond.csv",
        n,
        (x[:, 0].flatten(), x[:, 1].flatten(), x_corr.flatten()),
    )

    # conditional

    ax = fig.add_subplot(122, projection="3d")

    cond = lambda i, j, k: (
        kernel(i, j)
        - kernel(i, k) @ gp_regr.solve(kernel(k), kernel(k, j))  # type: ignore
    )

    cond_corr = lambda i, j, k: cond(i, j, k).flatten() / np.sqrt(
        np.diagonal(cond(i, i, k)) * np.diagonal(cond(j, j, k))
    )

    x_cond_corr = cond_corr(x, v, points)

    ax.scatter(
        points[:, 0],
        points[:, 1],
        np.zeros(len(points)),
        label="points",
        zorder=3,
        s=BIG_POINT,
        color=colors[1],
    )
    surf = ax.plot_wireframe(
        x[:, 0].reshape(n, n),
        x[:, 1].reshape(n, n),
        x_cond_corr.reshape(n, n),
        color=colors[-2],
    )
    ax.set_zlim3d(-0.1, 1)
    plt.savefig(f"{ROOT}/kernel.png")
    plt.clf()

    save(
        "matern_cond_points.csv",
        1,
        (
            points[:, 0].flatten(),
            points[:, 1].flatten(),
            np.zeros(len(points)),
        ),
    )
    save(
        "matern_cond.csv",
        n,
        (x[:, 0].flatten(), x[:, 1].flatten(), x_cond_corr.flatten()),
    )

    ## training and testing data

    D = 1  # dimension of points
    N = 8  # number of points
    M = 1  # number of testing points
    S = 2  # number of entries to pick

    POINT_SIZE = 512  # point sizes
    BIG_POINT = 1024  # large point
    LINEWIDTH = 8  # line width

    plt.rc("axes", labelsize=40)  # fontsize of the x and y labels
    plt.rc("legend", fontsize=27)  # fontsize of the legend

    # hand-craft points

    kernel = kernels.Matern(length_scale=1, nu=NU1 / NU2)

    x_train = np.array(
        [-0.9, -0.70, -0.57, 0.50, 0.56, 0.71, 0.8, 0.9]
    ).reshape(-1, 1)
    x_train += rng.uniform(-0.001, 0.001, x_train.shape)
    x_test = np.array([[0.0]])

    points = np.vstack((x_train, x_test))

    # generate realizations of Gaussian process
    # set random to a particular state to reproduce old code...
    rng.__setstate__(
        {
            "bit_generator": "PCG64",
            "state": {
                "state": 116693921324311315271032145974622854199,
                "inc": 194290289479364712180083596243593368443,
            },
            "has_uint32": 0,
            "uinteger": 0,
        }
    )

    mu = np.zeros(N + M)
    y = rng.multivariate_normal(mu, kernel(points))
    y_train, y_test = y[:N], y[N:]

    ## regression example

    cm = 1 / 2.54
    plt.figure(figsize=(20 * cm, 25 * cm))

    x = np.linspace(-R, R, 100).reshape(-1, 1)
    mu, sigma = estimate(x_train, y_train, x, kernel)
    print(mu, sigma)
    std = np.sqrt(np.diagonal(sigma))

    train = plt.scatter(
        x_train,
        y_train,
        label=r"$X_{Tr}$ train",
        s=POINT_SIZE,
        zorder=2.5,
        color=colors[0],
    )
    test = plt.scatter(
        x_test,
        y_test,
        label="$X_{Pr}$ predict",
        s=BIG_POINT,
        zorder=2.75,
        color=colors[1],
    )
    (mean,) = plt.plot(
        x, mu, label="conditional mean", color=colors[-1], linewidth=LINEWIDTH
    )
    sigma = plt.fill_between(
        x.flatten(),
        mu - 2 * std,
        mu + 2 * std,  # type: ignore
        label=r"$2 \sigma$",
        color=colors[-1],
        alpha=0.15,
    )
    plt.gca().set(xticklabels=[], yticklabels=[])
    plt.tick_params(axis="both", bottom=False, left=False)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend(handles=[train, test, mean, sigma])
    plt.tight_layout()
    plt.savefig(f"{ROOT}/predict_all.png")
    plt.clf()

    plt.axis("off")
    plt.figure(figsize=(20 * cm, 18 * cm))

    ## regression wrt. greedy sparse selection

    for s in range(1, S + 1):
        x = np.linspace(-R, R, 100).reshape(-1, 1)
        selected = cknn.knn_select(x_train, x_test, kernel, s)
        mu, sigma = estimate(x_train, y_train, x, kernel, selected)
        # standard deviation for each point (diagonal is variance)
        std = np.sqrt(np.diagonal(sigma))

        train = plt.scatter(
            x_train,
            y_train,
            label="training data",
            s=POINT_SIZE,
            zorder=2.5,
            color=colors[0],
        )
        test = plt.scatter(
            x_test,
            y_test,
            label="testing data",
            s=BIG_POINT,
            zorder=2.75,
            color=colors[1],
        )
        select = plt.scatter(
            x_train[selected],
            y_train[selected],
            label="selected",
            s=BIG_POINT,
            zorder=2.5,
            color=colors[3],
        )
        (mean,) = plt.plot(
            x,
            mu,
            label="conditional mean",
            color=colors[-1],
            linewidth=LINEWIDTH,
        )
        sigma = plt.fill_between(
            x.flatten(),
            mu - 2 * std,
            mu + 2 * std,  # type: ignore
            label=r"$2 \sigma$",
            color=colors[-1],
            alpha=0.15,
        )
        plt.axis((-1, 1, -1.5, 1.75))
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{ROOT}/predict_knn_{s}.png")
        plt.clf()

        save_1d(
            f"knn_selected_{s}.csv",
            (x_train[selected].flatten(), y_train[selected]),
        )
        save_1d(f"knn_mean_{s}.csv", (x.flatten(), mu))
        save_1d(f"knn_std_upper_{s}.csv", (x.flatten(), mu + 2 * std))
        save_1d(f"knn_std_lower_{s}.csv", (x.flatten(), mu - 2 * std))

    save_1d("train.csv", (x_train.flatten(), y_train))
    save_1d("test.csv", (x_test.flatten(), y_test))

    ## regression wrt. conditional sparse selection

    for s in range(1, S + 1):
        x = np.linspace(-R, R, 100).reshape(-1, 1)
        selected = cknn.select(x_train, x_test, kernel, s)
        mu, sigma = estimate(x_train, y_train, x, kernel, selected)
        std = np.sqrt(np.diagonal(sigma))

        train = plt.scatter(
            x_train,
            y_train,
            label="training data",
            s=POINT_SIZE,
            zorder=2.5,
            color=colors[0],
        )
        test = plt.scatter(
            x_test,
            y_test,
            label="testing data",
            s=BIG_POINT,
            zorder=2.75,
            color=colors[1],
        )
        select = plt.scatter(
            x_train[selected],
            y_train[selected],
            label="selected",
            s=BIG_POINT,
            zorder=2.5,
            color=colors[3],
        )
        (mean,) = plt.plot(
            x,
            mu,
            label="conditional mean",
            color=colors[-1],
            linewidth=LINEWIDTH,
        )
        sigma = plt.fill_between(
            x.flatten(),
            mu - 2 * std,
            mu + 2 * std,  # type: ignore
            label=r"$2 \sigma$",
            color=colors[-1],
            alpha=0.15,
        )
        plt.axis((-1, 1, -1.5, 1.75))
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{ROOT}/predict_cknn_{s}.png")
        plt.clf()

        save_1d(
            f"cknn_selected_{s}.csv",
            (x_train[selected].flatten(), y_train[selected]),
        )
        save_1d(f"cknn_mean_{s}.csv", (x.flatten(), mu))
        save_1d(f"cknn_std_upper_{s}.csv", (x.flatten(), mu + 2 * std))
        save_1d(f"cknn_std_lower_{s}.csv", (x.flatten(), mu - 2 * std))
