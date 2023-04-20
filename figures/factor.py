import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn.gaussian_process.kernels as kernels
from matplotlib.patches import Circle

from KoLesky import cholesky, ordering
from KoLesky.typehints import Kernel, Points, Sparse

from . import lightblue, orange, save_1d__, save__, seagreen, silver

ROOT = "figures/factor"
# make folders
os.makedirs(f"{ROOT}/data", exist_ok=True)
save = lambda *args, **kwargs: save__(*args, **kwargs, root=ROOT)
save_1d = lambda *args, **kwargs: save_1d__(*args, **kwargs, root=ROOT)

# fmt: off
D = 2         # dimension of points
N = 16        # number of points
M = 16        # number of columns in a group
S = 16        # number of entries to pick
COLS = N      # number of columns from the right
RHO = 2       # tuning parameter, number of nonzero entries
LAMBDA = 1.5  # tuning parameter, size of groups

SMALL_POINT = 10 # small point
POINT_SIZE = 20  # point sizes
BIG_POINT = 40   # large point
# fmt: on

# set random seed
rng = np.random.default_rng(3)


def save_points(fname: str, points: Points) -> None:
    """Write the points to the file."""
    save_1d(fname, (points[:, 0], points[:, 1]))


def get_factor(
    x: Points, kernel: Kernel, s: int, alg: str = "select"
) -> Sparse:
    """Factorize the set of points with the given algorithm."""
    theta = kernel(x)

    if alg == "select":
        factor = cholesky.cholesky_select(x, kernel, s)
    elif alg == "select-agg":
        indexes = list(range(N))
        groups = [
            indexes[M * i : M * (i + 1)] for i in range(int(np.ceil(N / M)))
        ]
        factor = cholesky.cholesky_select(x, kernel, M + s, groups)
    elif alg == "kl":
        factor, order = cholesky.cholesky_kl(x, kernel, RHO)
        theta = theta[np.ix_(order, order)]
    elif alg == "kl-agg":
        factor, order = cholesky.cholesky_kl(x, kernel, RHO, LAMBDA)
        theta = theta[np.ix_(order, order)]
    elif alg == "subsample":
        factor, order = cholesky.cholesky_subsample(x, kernel, s, RHO)
    elif alg == "subsample-agg":
        factor, order = cholesky.cholesky_subsample(x, kernel, s, RHO, LAMBDA)
    else:
        raise ValueError(f"Invalid algorithm {alg}.")

    return factor


def tikz_factor(fname: str, out: np.ndarray, col: int) -> None:
    """Render the Cholesky factor in TikZ."""
    indent = " " * 2
    N = out.shape[0]

    with open(fname, "w") as f:
        f.write(f"\\begin{{tikzpicture}}[scale=4/{COLS}]\n")

        # out = np.flip(out)
        # col = N - 1 - col

        # do special highlighted column last for z order
        for j in list(range(N - COLS, col)) + list(range(col + 1, N)) + [col]:
            # for i in range(j + 1):
            # lower triangular
            for i in range(j, N):
                ip, jp = i - (N - COLS), j - (N - COLS)
                coord1 = (jp, -ip)
                coord2 = (jp + 1, -(ip + 1))
                if j == col:
                    draw = "colborder"
                    fill = "selcolor" if out[i, j] else "candcolor"
                    if i == j:
                        fill = "targetcolor"
                else:
                    draw = "nnzborder" if out[i, j] else "zeroborder"
                    fill = "nnzcolor" if out[i, j] else "zerocolor"
                f.write(
                    f"{indent}\\filldraw[draw={draw}, fill={fill}] "
                    f"{coord1} rectangle {coord2};\n"
                )

        f.write("\\end{tikzpicture}\n")


def tikz_points_knn(
    fname: str, path: str, i: str, x: Points, radius: float
) -> None:
    """Render the points in TikZ."""
    p = str(tuple(x))
    with open(fname, "w") as f:
        f.write(
            f"""\
\\begin{{tikzpicture}}[baseline]
  \\begin{{axis}}[
    % calculated from Cholesky factor, exactly 16 cm x 16 cm
    width={{4cm}},
    height={{4cm}},
    axis lines={{none}},
    % force axis box to have exactly the right dimensions, ignoring labels
    scale only axis=true,
  ]
  % consistent size bounding box
  \\draw [white, line width=0] (-0.1, -0.1) -- (-0.1,  1.1);
  \\draw [white, line width=0] ( 1.1, -0.1) -- ( 1.1,  1.1);
  \\draw [white, line width=0] (-0.1, -0.1) -- (-1.1, -0.1);
  \\draw [white, line width=0] (-0.1,  1.1) -- (-1.1,  1.1);
  \\draw [seagreen!15, fill, radius={RHO*radius}] {p} circle;
  \\draw [seagreen, radius={RHO*radius}] {p} circle;
  \\draw [orange!25, fill, radius={radius}] {p} circle;
  \\draw [orange, radius={radius}] {p} circle;
  \\addplot [only marks, mark size=1, silver]    table
    {{{path}/all_points.csv}};
  \\addplot [only marks, mark size=2, lightblue] table
    {{{path}/candidates_{i}.csv}};
  \\addplot [only marks, mark size=4, seagreen]  table
    {{{path}/selected_{i}.csv}};
  \\addplot [only marks, mark size=4, orange]    table
    {{{path}/target_{i}.csv}};
  \\end{{axis}}
\\end{{tikzpicture}}
"""
        )


def tikz_points_cknn(fname: str, path: str, s: str) -> None:
    """Render the points in TikZ."""
    with open(fname, "w") as f:
        f.write(
            f"""\
\\begin{{tikzpicture}}[baseline]
  \\begin{{axis}}[
    % calculated from Cholesky factor, exactly 16 cm x 16 cm
    width={{4cm}},
    height={{4cm}},
    axis lines={{none}},
    % force axis box to have exactly the right dimensions, ignoring labels
    scale only axis=true,
  ]
  % consistent size bounding box
  \\draw [white, line width=0] (-0.1, -0.1) -- (-0.1,  1.1);
  \\draw [white, line width=0] ( 1.1, -0.1) -- ( 1.1,  1.1);
  \\draw [white, line width=0] (-0.1, -0.1) -- (-1.1, -0.1);
  \\draw [white, line width=0] (-0.1,  1.1) -- (-1.1,  1.1);
  \\addplot [only marks, mark size=1, silver]    table
    {{{path}/all_points.csv}};
  \\addplot [only marks, mark size=2, lightblue] table
    {{{path}/candidates.csv}};
  \\addplot [only marks, mark size=4, seagreen]  table
    {{{path}/selected_{s}.csv}};
  \\addplot [only marks, mark size=4, orange]    table
    {{{path}/target.csv}};
  \\end{{axis}}
\\end{{tikzpicture}}
"""
        )


if __name__ == "__main__":
    kernel = kernels.Matern(length_scale=1, nu=5 / 2)
    x = rng.random((N, D))

    ### knn

    name = "knn"
    path = f"figures/points_{name}"
    root = f"{ROOT}/data/{path}"
    os.makedirs(root, exist_ok=True)

    factor = get_factor(x, kernel, S, alg="kl")
    L = factor.toarray()
    out = 255 * (np.abs(L) > 0)

    order, lengths = ordering.reverse_maximin(x)
    x_old = x.copy()
    x = x[order]
    lengths[-1] = 2

    for i in range(N - 1, N - 1 - COLS, -1):
        i_str = f"{N - i:02}"

        # Cholesky factor

        # hide everything before ith column
        last_i = out.copy()
        last_i[:, :i] = 0
        tikz_factor(f"{root}/cholesky_factor_{i_str}.tex", last_i, i)

        # points

        selected = x[out[:, i] != 0]

        plt.scatter(
            x[:, 0],
            x[:, 1],
            label="all points",
            s=SMALL_POINT,
            zorder=2.5,
            color=silver,
        )
        plt.scatter(
            x[i:, 0],
            x[i:, 1],
            label="candidates",
            s=POINT_SIZE,
            zorder=2.75,
            color=lightblue,
        )
        plt.scatter(
            selected[:, 0],
            selected[:, 1],
            label="selected",
            s=BIG_POINT,
            zorder=3,
            color=seagreen,
        )
        plt.scatter(
            x[i : i + 1, 0],
            x[i : i + 1, 1],
            label="target",
            s=BIG_POINT,
            zorder=3.5,
            color=orange,
        )

        patches = [
            Circle(
                x[i],
                radius=lengths[i],
                edgecolor=orange + "ff",
                facecolor=orange + "40",
                linestyle="-",
                linewidth=2,
            ),
            Circle(
                x[i],
                radius=lengths[i] * RHO,
                edgecolor=seagreen + "ff",
                facecolor=seagreen + "20",
                linestyle="-",
                linewidth=2,
            ),
        ]
        for patch in patches:
            plt.gca().add_patch(patch)

        plt.axis("off")
        plt.axis("square")
        plt.axis((-0.1, 1.1, -0.1, 1.1))
        plt.tight_layout()
        plt.savefig(f"{root}/points_{i_str}.png")
        plt.clf()

        save_points(f"{path}/all_points.csv", x)
        save_points(f"{path}/candidates_{i_str}.csv", x[i:])
        save_points(f"{path}/selected_{i_str}.csv", selected)
        save_points(f"{path}/target_{i_str}.csv", x[i : i + 1])

        tikz_points_knn(
            f"{root}/selected_points_{i_str}.tex",
            path,
            i_str,
            x[i],
            lengths[i],
        )

    ### conditional knn

    name = "cknn"
    path = f"figures/points_{name}"
    root = f"{ROOT}/data/{path}"
    os.makedirs(root, exist_ok=True)
    # x = x_old

    # column to highlight
    # col = N - 13
    col = N - 14

    for s in range(1, N - col + 1):
        factor = get_factor(x, kernel, s, alg="select")
        L = factor.toarray()
        out = 255 * (np.abs(L) > 0)

        s_str = f"{s:02}"

        # Cholesky factor

        tikz_factor(f"{root}/cholesky_factor_{s_str}.tex", out, col)

        # points

        selected = x[out[:, col] != 0]

        plt.scatter(
            x[:, 0],
            x[:, 1],
            label="all points",
            s=SMALL_POINT,
            zorder=2.5,
            color=silver,
        )
        plt.scatter(
            x[col:, 0],
            x[col:, 1],
            label="candidates",
            s=POINT_SIZE,
            zorder=2.75,
            color=lightblue,
        )
        plt.scatter(
            selected[:, 0],
            selected[:, 1],
            label="selected",
            s=BIG_POINT,
            zorder=3,
            color=seagreen,
        )
        plt.scatter(
            x[col : col + 1, 0],
            x[col : col + 1, 1],
            label="target",
            s=BIG_POINT,
            zorder=3.5,
            color=orange,
        )

        plt.axis("off")
        plt.axis("square")
        plt.axis((-0.1, 1.1, -0.1, 1.1))
        plt.tight_layout()
        plt.savefig(f"{root}/points_{s_str}.png")
        plt.clf()

        save_points(f"{path}/all_points.csv", x)
        save_points(f"{path}/candidates.csv", x[col:])
        save_points(f"{path}/selected_{s_str}.csv", selected)
        save_points(f"{path}/target.csv", x[col : col + 1])

        tikz_points_cknn(f"{root}/selected_points_{s_str}.tex", path, s_str)
