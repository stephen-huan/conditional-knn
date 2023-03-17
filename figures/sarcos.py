import numpy as np
from experiments.gp_regr import get_dataset
from . import *

ROOT = "figures/sarcos"
# make folders
os.makedirs(f"{ROOT}/data", exist_ok=True)
save = lambda *args, **kwargs: save__(*args, **kwargs, root=ROOT)
save_1d = lambda *args, **kwargs: save_1d__(*args, **kwargs, root=ROOT)

D = 2    # dimension of points
N = 5000 # number of total points
POINT_SIZE = 1 # point sizes
BIG_POINT  = 40 # large point

def save_points(fname: str, points: np.ndarray) -> None:
    """ Write the points to the file. """
    save_1d(fname, (points[:, 0], points[:, 1]))

if __name__ == "__main__":
    dataset = "oco2"
    if dataset == "sarcos":
        points, y, m = get_dataset("sarcos_original")
        X_test = points[:m, :2]
        X_train = points[m: m + N, :2]
    elif dataset == "oco2":
        points, _, _ = get_dataset("oco2")
        X_train = points[0:len(points):len(points)//N + 1, :2]
    else:
        raise ValueError(f"Invalid datast {dataset}.")

    # selected = rng.choice(X_train, N, replace=False)
    # plt.scatter(selected[:, 0], selected[:, 1], label="points",
    #             s=POINT_SIZE, zorder=2.5, color=silver)
    plt.scatter(X_train[:, 0], X_train[:, 1], label="train",
                s=POINT_SIZE, zorder=2.5, color=silver)
    # plt.scatter(X_test[:, 0], X_test[:, 1], label="test",
    #             s=BIG_POINT, zorder=3.5, color=orange)
    # plt.axis("off")
    plt.axis("square")
    plt.tight_layout()
    plt.savefig(f"{ROOT}/points.png",
                bbox_inches="tight", pad_inches=0)
    plt.clf()

    save_points(f"points.csv", X_train)

