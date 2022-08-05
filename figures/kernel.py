import cknn
import gp_regression as gp_regr
from . import *
from matplotlib.animation import FuncAnimation

ROOT = "figures/kernel"
# make folders
os.makedirs(f"{ROOT}/data", exist_ok=True)
save = lambda *args, **kwargs: save__(*args, **kwargs, root=ROOT)
save_1d = lambda *args, **kwargs: save_1d__(*args, **kwargs, root=ROOT)

N = 51   # size of grid
S = 265  # number of points to select

SMALL_POINT = 10 # small point
POINT_SIZE =  20 # point sizes
BIG_POINT  =  40 # large point

ANIMATE = True   # draw animation
# number of points selected to animate to
UP_TO = N*N - 1

def save_points(fname: str, points: np.ndarray) -> None:
    """ Write the points to the file. """
    save_1d(fname, (points[:, 0], points[:, 1]))

def init() -> tuple:
    """ Initialize the animator. """
    plt.axis("off")
    plt.axis("square")
    plt.tight_layout()
    return all_plot, sel_plot, tgt_plot,

def update(frame: int) -> tuple:
    """ Callback on each new fraame of the animator. """
    sel = selected[:frame]
    sel_plot.set_data(x[sel, 0], x[sel, 1])
    return all_plot, sel_plot, tgt_plot,

if __name__ == "__main__":
    # x = gp_regr.perturbed_grid(rng, N*N, delta=0)
    x = gp_regr.perturbed_grid(rng, N*N, delta=1e-5)

    s = UP_TO if ANIMATE else S
    for i in range(4):
        point = (N + 1)*(N - 1)//2
        target = x[point: point + 1]
        train = np.vstack((x[:point], x[point + 1:]))
        # nearest neighbors
        if i == 0:
            selected = cknn.knn_select(train, target, cknn.euclidean, s)
            # kernel = kernels.Matern(length_scale=[1, 0.17], nu=1/2)*\
            #          kernels.Matern(length_scale=[0.17, 1], nu=1/2)
            # selected = cknn.corr_select(train, target, kernel, s)
        # different kernel functions
        else:
            kernel = [
                # isotropic kernel
                kernels.Matern(length_scale=1, nu=1/2),
                kernels.Matern(length_scale=1, nu=3/2),
                kernels.Matern(length_scale=1, nu=5/2),
                # additive kernel
                # kernels.Matern(length_scale=[1, 0.7], nu=5/2) + \
                # kernels.Matern(length_scale=[0.7, 1], nu=5/2),
                # product kernel
                # kernels.Matern(length_scale=[1, 0.17], nu=1/2)*\
                # kernels.Matern(length_scale=[0.17, 1], nu=1/2),
                # snowflake
                # kernels.Matern(length_scale=[1, 0.5], nu=1/2)*\
                # kernels.Matern(length_scale=[0.5, 1], nu=1/2),
                # flower
                # kernels.Matern(length_scale=[1.1, 0.17], nu=1/2)*\
                # kernels.Matern(length_scale=[0.17, 1.1], nu=1/2),
                # periodic kernel
                # kernels.ExpSineSquared(length_scale=1, periodicity=1),
                # kernels.DotProduct(),
            ][i - 1]
            selected = cknn.select(train, target, kernel, s)

        selected = [i if i < point else i + 1 for i in selected]

        plt.scatter(x[:, 0], x[:, 1], label="all points",
                    s=POINT_SIZE, zorder=2.5, color=silver)
        plt.scatter(x[selected[:S], 0], x[selected[:S], 1], label="selected",
                    s=POINT_SIZE, zorder=3, color=seagreen)
        plt.scatter(target[:, 0], target[:, 1], label="target",
                    s=POINT_SIZE, zorder=3.5, color=rust)
        plt.axis("off")
        plt.axis("square")
        plt.tight_layout()
        plt.savefig(f"{ROOT}/points_{i + 1}.png")
        plt.clf()

        save_points(f"points.csv", x)
        save_points(f"selected_{i + 1}", x[selected])
        save_points(f"target.csv", target)

        # animate the order of points being selected
        fig, ax = plt.subplots()

        all_plot, = plt.plot(x[:, 0], x[:, 1], "o ", label="all points",
                             markersize=np.sqrt(POINT_SIZE), color=silver,
                             zorder=2.5, animated=True)
        sel_plot, = plt.plot([], [], "o ", label="selected",
                             markersize=np.sqrt(POINT_SIZE), color=seagreen,
                             zorder=3, animated=True)
        tgt_plot, = plt.plot(target[:, 0], target[:, 1], "o ", label="target",
                             markersize=np.sqrt(POINT_SIZE), color=rust,
                             zorder=3.5, animated=True)

        if ANIMATE:
            anime = FuncAnimation(fig, update, frames=np.arange(0, s + 1, 1),
                                  init_func=init, blit=True, interval=10)
            anime.save(f"{ROOT}/points_{i + 1}.mp4", writer="ffmpeg", fps=60)
            # plt.show()

        plt.clf()

