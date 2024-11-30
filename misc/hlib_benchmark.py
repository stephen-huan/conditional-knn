import time

import numpy as np
import pbbfmm3d  # pyright: ignore
import sklearn.gaussian_process.kernels as kernels
from pbbfmm3d.kernels import from_sklearn  # pyright: ignore

from KoLesky import hlibpro  # pyright: ignore

# display settings
np.set_printoptions(precision=3, suppress=True)

# set random seed
rng = np.random.default_rng(1)


if __name__ == "__main__":
    D = 3  # dimension of points
    N = int(1e5)  # number of points
    n_cols = 10  # number of iterations

    X = rng.random((N, D))

    kernel = kernels.Matern(length_scale=1, nu=5 / 2)
    start = time.time()
    hlib_matvec, done, _ = hlibpro.gram(kernel, X)
    hlib_time = time.time() - start
    print(f"hlib preprocessing took {hlib_time:.3f}")
    start = time.time()
    fmm_kernel = from_sklearn(kernels.Matern(length_scale=1, nu=5 / 2))
    fmm_kernel.init(L=1, tree_level=4, interpolation_order=5)
    fmm_kernel.build()
    fmm_matvec = pbbfmm3d.gram(fmm_kernel, X)
    fmm_time = time.time() - start
    print(f"fmm preprocessing took {fmm_time:.3f}")
    for i in range(n_cols):
        y = rng.random((N,))
        start = time.time()
        hlib_matvec(y)
        hlib_time += time.time() - start
        print(
            f"{i + 1:3}th hlib iteration took {time.time() - start:.3f}"
        )
        start = time.time()
        fmm_matvec(y)
        fmm_time += time.time() - start
        print(
            f"{i + 1:3}th fmm iteration took {time.time() - start:.3f}"
        )
    print(f"hlib took {hlib_time:.3f} total")
    print(f"fmm took {fmm_time:.3f} total")
    done()
    hlibpro.finish()
