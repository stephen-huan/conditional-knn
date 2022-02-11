# cython: profile=False
from libc.math cimport sqrt, exp, INFINITY
cimport numpy as np
import numpy as np
cimport scipy.linalg.cython_blas as blas
import scipy.spatial.distance

### covariance functions

cdef double SQRT3 = sqrt(3)
cdef double SQRT5 = sqrt(5)

cdef double __matern12(double x):
    """ Matern Kernel function with v = 1/2 and with length scale l. """
    return exp(-x)

cdef double __matern32(double x):
    """ Matern Kernel function with v = 3/2 and with length scale l. """
    x *= SQRT3
    return (1 + x)*exp(-x)

cdef double __matern52(double x):
    """ Matern Kernel function with v = 5/2 and with length scale l. """
    x *= SQRT5
    return (1 + x + x*x/3)*exp(-x)

cdef (double (*)(double x)) __get_kernel(double nu):
    """ Get the Matern kernel with the given nu. """
    cdef double (*kernel)(double x)
    kernel = &__matern52
    if nu == 0.5:
        kernel = &__matern12
    elif nu == 1.5:
        kernel = &__matern32
    elif nu == 2.5:
        kernel = &__matern52
    return kernel

cdef void covariance_vector(double nu, double length_scale,
                            double[:, ::1] points, double[::1] point,
                            double *vector):
    """ Covariance between points and point. """
    cdef:
        int n, i, j
        double (*kernel)(double x)
        double dist, d
        double *start
        double *p

    kernel = __get_kernel(nu)
    n = points.shape[1]
    start = &points[0, 0]
    p = &point[0]
    for i in range(points.shape[0]):
        dist = 0
        for j in range(n):
            d = (start + i*n)[j] - p[j]
            dist += d*d
        vector[i] = kernel(sqrt(dist)/length_scale)

### selection methods

cdef void __chol_update(int n, int i, int k,
                        double[::1, :] factors, double[::1] cond_var):
    """ Updates the ith column of the Cholesky factor with column k. """
    cdef:
        char *TRANS
        int M, N, LDA, INCX, INCY, j
        double ALPHA, BETA
        double *A
        double *X
        double *Y

    # update Cholesky factors

    # factors[:, i] -= factors[:, :i]@factors[k, :i]
    TRANS = 'n'
    M = n
    N = i
    ALPHA = -1
    A = &factors[0, 0]
    LDA = factors.shape[0]
    X = &factors[k, 0]
    INCX = LDA
    BETA = 1
    Y = &factors[0, i]
    INCY = 1
    blas.dgemv(TRANS, &M, &N, &ALPHA, A, &LDA, X, &INCX, &BETA, Y, &INCY)

    # factors[:, i] /= np.sqrt(factors[k, i])
    ALPHA = 1/sqrt(factors[k, i])
    blas.dscal(&M, &ALPHA, Y, &INCY)

    # update conditional variance
    for j in range(cond_var.shape[0]):
        cond_var[j] -= factors[j, i]*factors[j, i]
    # clear out selected index
    if k < cond_var.shape[0]:
        cond_var[k] = INFINITY

def chol_update(double[::1] cov_k, int i, int k,
                double[::1, :] factors, double[::1] cond_var) -> None:
    """ Updates the ith column of the Cholesky factor with column k. """
    factors[:, i] = cov_k
    __chol_update(cov_k.shape[0], i, k, factors, cond_var)

cdef long[::1] __chol_select(double[:, ::1] x_train, double[:, ::1] x_test,
                             double nu, double length_scale, int s):
    """ Select the s most informative entries, storing a Cholesky factor. """
    # O(s*(n*s)) = O(n s^2)
    cdef:
        int n, i, j, k, INCX
        double best, ALPHA
        long[::1] indexes
        double[::1, :] factors
        double[::1] cond_var, cond_cov

    n = x_train.shape[0]
    # initialization
    indexes = np.zeros(min(n, s), dtype=np.int64)
    factors = np.zeros((n + 1, s), order="F")
    # cond_cov = kernel(x_train, x_test).flatten()
    cond_cov = np.zeros(n)
    covariance_vector(nu, length_scale, x_train, x_test[0], &cond_cov[0])
    # cond_var = kernel.diag(x_train)
    # Matern kernels have covariance one between a point and itself
    cond_var = np.ones(n)

    i = 0
    while i < indexes.shape[0]:
        # pick best entry
        k, best = 0, -INFINITY
        for j in range(n):
            if cond_var[j] != INFINITY and \
                    cond_cov[j]*cond_cov[j]/cond_var[j] > best:
                k = j
                best = cond_cov[j]*cond_cov[j]/cond_var[j]
        indexes[i] = k
        # update Cholesky factors
        covariance_vector(nu, length_scale, x_train, x_train[k],
                          &factors[0, i])
        covariance_vector(nu, length_scale, x_test, x_train[k],
                          &factors[n, i])
        __chol_update(n + 1, i, k, factors, cond_var)
        # update conditional covariance
        # cond_cov -= factors[:, i][:n]*factors[n, i]
        ALPHA = -factors[n, i]
        INCX = 1
        blas.daxpy(&n, &ALPHA, &factors[0, i], &INCX, &cond_cov[0], &INCX)
        i += 1

    return indexes

cdef long[::1] __chol_mult_select(double[:, ::1] x_train,
                                  double[:, ::1] x_test,
                                  double nu, double length_scale, int s):
    """ Greedily select the s entries minimizing conditional covariance. """
    # O(m*(n + m)*m + s*(n + m)*(s + m)) = O(n s^2 + n m^2 + m^3)
    cdef:
        int n, m, i, j, k
        double best
        long[::1] indexes
        double[::1, :] factors, factors_pr
        double[::1] cond_var, cond_var_pr, cov_k

    n, m = x_train.shape[0], x_test.shape[0]
    # initialization
    indexes = np.zeros(min(n, s), dtype=np.int64)
    factors = np.zeros((n, s), order="F")
    factors_pr = np.zeros((n + m, s + m), order="F")
    # cond_var = kernel.diag(x_train)
    # Matern kernels have covariance one between a point and itself
    cond_var = np.ones(n)
    cond_var_pr = np.copy(cond_var)
    # pre-condition on the m prediction points
    for i in range(m):
        covariance_vector(nu, length_scale, x_train, x_test[i],
                          &factors_pr[0, i])
        covariance_vector(nu, length_scale, x_test, x_test[i],
                          &factors_pr[n, i])
        __chol_update(n + m, i, n + i, factors_pr, cond_var_pr)

    i = 0
    while i < indexes.shape[0]:
        # pick best entry
        k, best = 0, INFINITY
        for j in range(n):
            if cond_var[j] != INFINITY and cond_var_pr[j]/cond_var[j] < best:
                k = j
                best = cond_var_pr[j]/cond_var[j]
        indexes[i] = k
        # update Cholesky factors
        covariance_vector(nu, length_scale, x_train, x_train[k],
                          &factors[0, i])
        factors_pr[:n, i + m] = factors[:, i]
        __chol_update(n, i, k, factors, cond_var)
        __chol_update(n, i + m, k, factors_pr, cond_var_pr)
        i += 1

    return indexes

def select(double[:, ::1] x_train, double[:, ::1] x_test,
           kernel, int s) -> list:
    """ Wrapper over various cknn selection methods. """
    cdef double nu, length_scale
    params = kernel.get_params()
    nu, length_scale = params["nu"], params["length_scale"]
    # single prediction point, use specialized function
    if x_test.shape[0] == 1:
        return list(__chol_select(x_train, x_test, nu, length_scale, s))
    else:
        return list(__chol_mult_select(x_train, x_test, nu, length_scale, s))

