import numpy as np
import sklearn.gaussian_process.kernels as kernels

from KoLesky import hlibpro  # pyright: ignore

# display settings
np.set_printoptions(precision=3, suppress=True)

# set random seed
rng = np.random.default_rng(1)


if __name__ == "__main__":
    D = 3  # dimension of points
    N = 50  # number of points
    M = 25  # number of points

    smoothnesses = np.array([1 / 2, 3 / 2, 5 / 2])
    length_scales = 10.0 ** np.arange(-2, 3)

    X1 = rng.random((N, D))
    X2 = rng.random((M, D))
    y1 = rng.random((N,))
    y2 = rng.random((M,))

    for length_scale in length_scales:
        sklean_kernels = [
            kernels.Matern(length_scale=length_scale, nu=nu)
            for nu in smoothnesses
        ]
        for kernel in sklean_kernels:
            theta_11: np.ndarray = kernel(X1, X1)  # type: ignore
            theta_22: np.ndarray = kernel(X2, X2)  # type: ignore

            matvec, done, _ = hlibpro.gram(kernel, X1)
            assert np.allclose(
                matvec(y1), theta_11 @ y1
            ), f"hlibpro gram 1 wrong for {kernel}."
            done()
            matvec, done, _ = hlibpro.gram(kernel, X2)
            assert np.allclose(
                matvec(y2), theta_22 @ y2
            ), f"hlibpro gram 2 wrong for {kernel}."
            done()
            if length_scale > 10:
                continue
            matvec, done, _ = hlibpro.gram(kernel, X1, inverse=True)
            assert np.allclose(
                matvec(y1), np.linalg.inv(theta_11) @ y1, rtol=1e-4
            ), f"hlibpro gram 1 inverse wrong for {kernel}."
            done()
            matvec, done, _ = hlibpro.gram(kernel, X2, inverse=True)
            assert np.allclose(
                matvec(y2), np.linalg.inv(theta_22) @ y2, rtol=1e-4
            ), f"hlibpro gram 2 inverse wrong for {kernel}."
            done()

    hlibpro.finish()
