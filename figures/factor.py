import cholesky
from . import *

ROOT = "figures/factor"
# make folders
os.makedirs(f"{ROOT}/data", exist_ok=True)
save = lambda *args, **kwargs: save__(*args, **kwargs, root=ROOT)
save_1d = lambda *args, **kwargs: save_1d__(*args, **kwargs, root=ROOT)

D = 2        # dimension of points
N = 16       # number of points
M = 16       # number of columns in a group
S = 7        # number of entries to pick
RHO = 2      # tuning parameter, number of nonzero entries
LAMBDA = 1.5 # tuning parameter, size of groups

SMALL_POINT = 10 # small point
POINT_SIZE  = 20 # point sizes
BIG_POINT   = 40 # large point

# set random seed
rng = np.random.default_rng(3)

def save_points(fname: str, points: np.ndarray) -> None:
    """ Write the points to the file. """
    save_1d(fname, (points[:, 0], points[:, 1]))


if __name__ == "__main__":
    kernel = kernels.Matern(length_scale=1, nu=5/2)
    x = rng.random((N, D))
    theta = kernel(x)

    alg = "select"

    if alg == "select":
        factor = cholesky.cholesky_select(x, kernel, S)
    if alg == "select-agg":
        indexes = list(range(N))
        groups = [indexes[M*i: M*(i + 1)] for i in range(int(np.ceil(N/M)))]
        factor = cholesky.cholesky_select(x, kernel, M + 2, groups)
    if alg == "kl":
        factor, order = cholesky.cholesky_kl(x, kernel, RHO)
        theta = theta[np.ix_(order, order)]
    if alg == "kl-agg":
        factor, order = cholesky.cholesky_kl(x, kernel, RHO, LAMBDA)
        theta = theta[np.ix_(order, order)]
    if alg == "subsample":
        factor, order = cholesky.cholesky_subsample(x, kernel, 2, RHO)
    if alg == "subsample-agg":
        factor, order = cholesky.cholesky_subsample(x, kernel, 2, RHO, LAMBDA)

    L = factor.toarray()
    out = 255*(np.abs(L) > 0)

    ### Cholesky factor

    col = 3

    with open(f"{ROOT}/data/cholesky_factor.tex", "w") as f:
        f.write("% outer triangular factor\n")
        f.write(f"\\fill[lightsilver] \
(0, 0) -- (0, {-N}) -- ({N}, {-N}) -- cycle;\n")

        f.write("\n% column rectangle\n")
        coord1 = (col, -col - 1)
        coord2 = (col + 1, -N)
        f.write(f"\draw[fill=lightblue] {coord1} rectangle {coord2};\n")

        f.write("\n% triangular factor\n")
        for i in range(len(L)):
            for j in range(len(L)):
                # lower triangular
                if i >= j:
                    coord1 = (j, -i)
                    coord2 = (j + 1, -(i + 1))
                    if out[i, j] != 0:
                        if j == col:
                            color = "orange" if i == j else "seagreen"
                        else:
                            color = "silver"
                        f.write(f"\\fill[{color}] \
{coord1} rectangle {coord2};\n");
                    if j == col:
                        f.write(f"\\draw \
{coord1} rectangle {coord2};\n");

        f.write("\n% column rectangle\n")
        coord1 = (col, -col - 1)
        coord2 = (col + 1, -N)
        f.write(f"\draw {coord1} rectangle {coord2};\n")

    ### points

    selected = x[out[:, col] != 0]

    plt.scatter(x[:, 0], x[:, 1], label="all points",
                s=SMALL_POINT, zorder=2.5, color=silver)
    plt.scatter(x[col:, 0], x[col:, 1], label="candidates",
                s=POINT_SIZE, zorder=2.75, color=lightblue)
    plt.scatter(selected[:, 0], selected[:, 1], label="selected",
                s=BIG_POINT, zorder=3, color=seagreen)
    plt.scatter(x[col: col + 1, 0], x[col: col + 1, 1], label="target",
                s=BIG_POINT, zorder=3.5, color=orange)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{ROOT}/points.png")

    save_points("all_points.csv", x)
    save_points("candidates.csv", x[col:])
    save_points("selected.csv", selected)
    save_points("target.csv", x[col: col + 1])

