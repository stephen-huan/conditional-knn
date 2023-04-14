import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import scipy.sparse as sparse
import sklearn.gaussian_process.kernels as kernels
from sklearn.gaussian_process.kernels import Kernel

# display settings
np.set_printoptions(precision=3, suppress=True)

# set random seed
rng = np.random.default_rng(1)

### data generation methods

# Laplacians


def get_mm_matrix(fname: str) -> sparse.coo_matrix:
    """Load a Matrix Market formatted file."""
    return io.mmread(f"datasets/suitesparse/{fname}/{fname}.mtx")


def get_laplacian(fname: str) -> sparse.coo_matrix:
    """Return the Laplacian of an undirected weighted graph."""
    graph = get_mm_matrix(fname)
    laplacian = sparse.csgraph.laplacian(graph)
    return laplacian


### helper methods


def avg_results__(f, trials: int) -> np.ndarray:
    """Average the results from a function."""
    return np.average([f() for _ in range(trials)], axis=0)


def load_data__(
    x_name: str, y_names: list, names: list, root: str
) -> np.ndarray:
    """Load data from csv files."""
    data = [[None for _ in range(len(names))] for _ in range(len(y_names))]
    for d in range(len(y_names)):
        for i in range(len(names)):
            fname = f"{root}/data/{x_name}_{y_names[d]}_{names[i]}.csv"
            data[d][i] = np.loadtxt(fname, delimiter=" ")[:, 1]
    return np.array(data)


def save_data__(
    data: list,
    x: np.ndarray,
    x_name: str,
    y_names: list,
    names: list,
    root: str,
) -> None:
    """Save data into a csv file."""
    for d in range(len(data)):
        for i in range(len(data[d])):
            fname = f"{root}/data/{x_name}_{y_names[d]}_{names[i]}.csv"
            table = np.array([x, np.array(data[d][i])]).T
            np.savetxt(fname, table, delimiter=" ")


def plot__(
    x: np.ndarray,
    ys: np.ndarray,
    names: list,
    colors: list,
    x_name: str,
    y_name: str,
    callback=lambda x: x,
    root: str = "",
) -> None:
    """Plot the data, taking an optional callback."""
    for y, name, color in zip(ys, names, colors):
        plt.plot(x, y, label=name, color=color)

    callback()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{root}/{x_name}_{y_name}.png")
    plt.clf()


### graphing

# colors
# extract from latex with:
# \extractcolorspecs{lightblue}{\model}{\mycolor}
# \convertcolorspec{\model}{\mycolor}{HTML}\tmp\tmp
lightblue = "#a1b4c7"
orange = "#ea8810"
silver = "#b0aba8"
rust = "#b8420f"
seagreen = "#23553c"

lightsilver = "#e7e6e5"
darkorange = "#c7740e"
darksilver = "#96918f"
darklightblue = "#8999a9"
darkrust = "#9c380d"
darkseagreen = "#1e4833"

# plt.style.available for available themes
plt.style.use("seaborn-whitegrid")
