# cython: profile=False
from libc.math cimport sqrt, exp, log, INFINITY
cimport numpy as np
import numpy as np
cimport scipy.linalg.cython_blas as blas

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
        double[::1] cond_var, cond_var_pr

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

### non-adjacent multiple point selection

cdef void __chol_insert(long[::1] order, int i, int index,
                        int k, double[::1, :] factors):
    """ Updates the ith column of the Cholesky factor with column k. """
    cdef:
        char *TRANS
        int M, N, LDA, INCX, INCY, last, col
        double ALPHA, BETA, dp
        double *A
        double *X
        double *Y

    last = factors.shape[1] - 1
    # condition covariance on previous variables
    # factors[:, last] -= factors[:, :index]@factors[k, :index]
    TRANS = 'n'
    M = factors.shape[0]
    N = index
    ALPHA = -1
    A = &factors[0, 0]
    LDA = factors.shape[0]
    X = &factors[k, 0]
    INCX = LDA
    BETA = 1
    Y = &factors[0, last]
    INCY = 1
    blas.dgemv(TRANS, &M, &N, &ALPHA, A, &LDA, X, &INCX, &BETA, Y, &INCY)
    # factors[:, index] /= sqrt(factors[k, index])
    ALPHA = 1/sqrt(factors[k, last])
    blas.dscal(&M, &ALPHA, Y, &INCY)

    # move columns over to make space at index
    for col in range(i, index, -1):
        # factors[:, col] = factors[:, col - 1]
        blas.dcopy(&M, &factors[0, col - 1], &INCY, &factors[0, col], &INCY)
    # copy conditional covariance from temporary storage to index
    # factors[:, index] = factors[:, last]
    blas.dcopy(&M, Y, &INCY, &factors[0, index], &INCY)

    # update downstream Cholesky factor by rank-one downdate
    for col in range(index + 1, i + 1):
        X = &factors[0, col]
        k = order[col]
        ALPHA, BETA = factors[k, col], Y[k]
        dp = sqrt(ALPHA*ALPHA - BETA*BETA)
        ALPHA, BETA = ALPHA/dp, -BETA/dp
        # factors[:, col] *= ALPHA
        blas.dscal(&M, &ALPHA, X, &INCY)
        # factors[:, col] += BETA*cov_k
        blas.daxpy(&M, &BETA, Y, &INCY, X, &INCY)
        # cov_k *= 1/ALPHA
        ALPHA = 1/ALPHA
        blas.dscal(&M, &ALPHA, Y, &INCY)
        # cov_k += BETA/ALPHA*factors[:, col]
        BETA *= ALPHA
        blas.daxpy(&M, &BETA, X, &INCY, Y, &INCY)

cdef int __insert_index(long[::1] order, long[::1] locations, int i, int k):
    """ Finds the index to insert index k into the order. """
    cdef int index = -1
    for index in range(i):
        # bigger than current value, insertion spot
        if locations[k] >= locations[order[index]]:
            return index
    return index + 1

cdef void __select_point(long[::1] order, long[::1] locations, int i, int k,
                         double[:, ::1] points, double nu, double length_scale,
                         double[::1, :] factors):
    """ Add the kth point to the Cholesky factor. """
    cdef int col, index, last

    index = __insert_index(order, locations, i, k)
    # shift values over to make room for k at index
    for col in range(i, index, -1):
        order[col] = order[col - 1]
    order[index] = k
    # insert covariance with k into Cholesky factor
    # use last column as temporary working space
    last = factors.shape[1] - 1
    covariance_vector(nu, length_scale, points, points[k], &factors[0, last])
    __chol_insert(order, i, index, k, factors)

cdef long[::1] __chol_nonadj_select(double[:, ::1] x,
                                    long[::1] train, long[::1] test,
                                    double nu, double length_scale, int s):
    """ Greedily select the s entries minimizing conditional covariance. """
    # O(m*(n + m)*m + s*(n + m)*(s + m)) = O(n s^2 + n m^2 + m^3)
    cdef:
        int n, m, i, j, k, best_k, col, index
        double best, key, cond_cov_k, cond_var_k, cond_var_j
        long[::1] indexes, order, locations
        double[:, ::1] points
        double[::1, :] factors
        double[::1] var, prefix

    n, m = train.shape[0], test.shape[0]
    locations = np.append(train, test)
    points = np.asarray(x)[locations]
    # initialization
    indexes = np.zeros(min(n, s), dtype=np.int64)
    order = np.zeros(indexes.shape[0] + test.shape[0], dtype=np.int64)
    factors = np.zeros((n + m, s + m + 1), order="F")
    # var = kernel.diag(x[train])
    # Matern kernels have covariance one between a point and itself
    var = np.ones(n)
    # pre-condition on the m prediction points
    for i in range(m):
        __select_point(order, locations, i, n + i,
                       points, nu, length_scale, factors)

    i = 0
    while i < indexes.shape[0]:
        best, best_k = INFINITY, 0
        # pick best entry
        for j in range(n):
            # selected already, don't consider as candidate
            if var[j] == INFINITY:
                continue

            cond_var_j = var[j]
            index = __insert_index(order, locations, m + i, j)
            # log(cond_var_j) is 0 for Matern kernel, no need to do casework
            key = 0

            for col in range(m + i):
                k = order[col]
                cond_var_k = factors[k, col]
                cond_cov_k = factors[j, col]*cond_var_k
                cond_var_k *= cond_var_k
                cond_cov_k *= cond_cov_k
                # remove spurious contribution from selected training point
                if k < n:
                    key -= log(cond_var_k - (cond_cov_k/cond_var_j
                                             if col >= index else 0))
                cond_var_j -= cond_cov_k/cond_var_k
                # remove spurious contribution of j
                if col + 1 == index:
                    key -= log(cond_var_j)
            # add logdet of entire covariance matrix
            key += log(cond_var_j)

            if key < best:
                best, best_k = key, j

        indexes[i] = best_k
        # mark as selected
        var[best_k] = INFINITY
        # update Cholesky factor
        __select_point(order, locations, i + m, best_k,
                       points, nu, length_scale, factors)
        i += 1

    return indexes

### wrapper functions

def chol_update(double[::1] cov_k, int i, int k,
                double[::1, :] factors, double[::1] cond_var) -> None:
    """ Updates the ith column of the Cholesky factor with column k. """
    factors[:, i] = cov_k
    __chol_update(cov_k.shape[0], i, k, factors, cond_var)

def select(double[:, ::1] x_train, double[:, ::1] x_test,
           kernel, int s) -> np.ndarray:
    """ Wrapper over various cknn selection methods. """
    cdef double nu, length_scale
    params = kernel.get_params()
    nu, length_scale = params["nu"], params["length_scale"]
    # single prediction point, use specialized function
    if x_test.shape[0] == 1:
        selected = __chol_select(x_train, x_test, nu, length_scale, s)
    else:
        selected = __chol_mult_select(x_train, x_test, nu, length_scale, s)
    return np.asarray(selected)

def nonadj_select(double[:, ::1] x, long[::1] train, long[::1] test,
                  kernel, int s) -> np.ndarray:
    """ Wrapper over various cknn selection methods. """
    cdef double nu, length_scale
    params = kernel.get_params()
    nu, length_scale = params["nu"], params["length_scale"]
    # single prediction point, use specialized function
    if len(test) == 1:
        points = np.asarray(x)
        selected = __chol_select(points[train], points[test],
                                 nu, length_scale, s)
    else:
        selected = __chol_nonadj_select(x, train, test, nu, length_scale, s)
    return np.asarray(selected)

