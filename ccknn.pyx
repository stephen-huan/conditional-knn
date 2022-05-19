# cython: profile=False
from libc.math cimport INFINITY, sqrt, log2
cimport numpy as np
# we rely on importing numpy to also load intel mkl libraries, see README.md
import numpy as np
cimport scipy.linalg.cython_blas as blas
cimport mkl
from c_kernels cimport Kernel, get_kernel, kernel_cleanup
from c_kernels cimport covariance_vector, variance_vector

### selection methods

cdef void __chol_update(int n, int i, int k,
                        double[::1, :] factors, double[::1] cond_var):
    """ Updates the ith column of the Cholesky factor with column k. """
    cdef:
        char *trans
        int M, N, lda, incx, incy, j
        double alpha, beta
        double *A
        double *x
        double *y

    # update Cholesky factors

    # factors[:, i] -= factors[:, :i]@factors[k, :i]
    trans = 'n'
    M = n
    N = i
    alpha = -1
    A = &factors[0, 0]
    lda = factors.shape[0]
    x = &factors[k, 0]
    incx = lda
    beta = 1
    y = &factors[0, i]
    incy = 1
    blas.dgemv(trans, &M, &N, &alpha, A, &lda, x, &incx, &beta, y, &incy)
    # factors[:, i] /= np.sqrt(factors[k, i])
    alpha = 1/sqrt(factors[k, i])
    blas.dscal(&M, &alpha, y, &incy)

    # update conditional variance
    for j in range(cond_var.shape[0]):
        alpha = factors[j, i]
        cond_var[j] -= alpha*alpha

    # clear out selected index
    if k < cond_var.shape[0]:
        cond_var[k] = INFINITY

cdef long[::1] __chol_select(double[:, ::1] x_train, double[:, ::1] x_test,
                             Kernel *kernel, int s):
    """ Select the s most informative entries, storing a Cholesky factor. """
    # O(s*(n*s)) = O(n s^2)
    cdef:
        int n, i, j, k, incx
        double best, alpha
        long[::1] indexes
        double[::1, :] factors
        double[::1] cond_var, cond_cov

    n = x_train.shape[0]
    # initialization
    indexes = np.zeros(min(n, s), dtype=np.int64)
    factors = np.zeros((n + 1, s), order="F")
    # cond_cov = kernel(x_train, x_test).flatten()
    cond_cov = np.zeros(n)
    covariance_vector(kernel, x_train, x_test[0], &cond_cov[0])
    # cond_var = kernel.diag(x_train)
    cond_var = np.zeros(n)
    variance_vector(kernel, x_train, &cond_var[0])

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
        covariance_vector(kernel, x_train, x_train[k], &factors[0, i])
        covariance_vector(kernel, x_test, x_train[k], &factors[n, i])
        __chol_update(n + 1, i, k, factors, cond_var)
        # update conditional covariance
        # cond_cov -= factors[:, i][:n]*factors[n, i]
        alpha = -factors[n, i]
        incx = 1
        blas.daxpy(&n, &alpha, &factors[0, i], &incx, &cond_cov[0], &incx)
        i += 1

    return indexes

cdef long[::1] __chol_mult_select(double[:, ::1] x_train,
                                  double[:, ::1] x_test,
                                  Kernel *kernel, int s):
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
    cond_var = np.zeros(n)
    variance_vector(kernel, x_train, &cond_var[0])
    cond_var_pr = np.copy(cond_var)
    # pre-condition on the m prediction points
    for i in range(m):
        covariance_vector(kernel, x_train, x_test[i], &factors_pr[0, i])
        covariance_vector(kernel, x_test, x_test[i], &factors_pr[n, i])
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
        covariance_vector(kernel, x_train, x_train[k], &factors[0, i])
        factors_pr[:n, i + m] = factors[:, i]
        __chol_update(n, i, k, factors, cond_var)
        __chol_update(n, i + m, k, factors_pr, cond_var_pr)
        i += 1

    return indexes

### non-adjacent multiple point selection

cdef unsigned long long MANTISSA_MASK = (1 << 52) - 1
cdef unsigned long long MANTISSA_BASE = ((1 << 10) - 1) << 52
cdef unsigned long long EXPONENT_MASK = ((1 << 11) - 1) << 52

cdef double __get_mantissa(double x):
    """ Get the mantissa component of a double. """
    cdef unsigned long long value = (<unsigned long long *> &x)[0]
    new_value = MANTISSA_BASE + (value & MANTISSA_MASK)
    return (<double*> &new_value)[0]

cdef int __get_exponent(double x):
    """ Get the exponent component of a double. """
    cdef unsigned long long value = (<unsigned long long *> &x)[0]
    return ((value & EXPONENT_MASK) >> 52) - 1023

cdef double __log_product(int n, double *x, int incx):
    """ Return the log of the product of the entries of a vector. """
    cdef:
        unsigned long long value
        double mantissa
        int i, exponent

    mantissa = 1
    exponent = 0
    for i in range(n):
        value = (<unsigned long long *> (x + i*incx))[0]
        exponent += ((value & EXPONENT_MASK) >> 52) - 1023
        value = MANTISSA_BASE + (value & MANTISSA_MASK)
        mantissa *= (<double *> &value)[0]
        # prevent underflow by periodic normalization
        if i & 512:
            exponent += __get_exponent(mantissa)
            mantissa = __get_mantissa(mantissa)

    return log2(mantissa) + exponent

cdef void __chol_insert(long[::1] order, int i, int index,
                        int k, double[::1, :] factors):
    """ Updates the ith column of the Cholesky factor with column k. """
    cdef:
        char *trans
        int m, n, lda, incx, incy, last, col
        double alpha, beta, dp
        double *A
        double *x
        double *y

    last = factors.shape[1] - 1
    # condition covariance on previous variables
    # factors[:, last] -= factors[:, :index]@factors[k, :index]
    trans = 'n'
    m = factors.shape[0]
    n = index
    alpha = -1
    A = &factors[0, 0]
    lda = factors.shape[0]
    x = &factors[k, 0]
    incx = lda
    beta = 1
    y = &factors[0, last]
    incy = 1
    blas.dgemv(trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy)
    # factors[:, index] /= sqrt(factors[k, index])
    alpha = 1/sqrt(factors[k, last])
    blas.dscal(&m, &alpha, y, &incy)

    # move columns over to make space at index
    for col in range(i, index, -1):
        # factors[:, col] = factors[:, col - 1]
        blas.dcopy(&m, &factors[0, col - 1], &incy, &factors[0, col], &incy)
    # copy conditional covariance from temporary storage to index
    # factors[:, index] = factors[:, last]
    blas.dcopy(&m, y, &incy, &factors[0, index], &incy)

    # update downstream Cholesky factor by rank-one downdate
    for col in range(index + 1, i + 1):
        x = &factors[0, col]
        k = order[col]
        alpha, beta = factors[k, col], y[k]
        dp = sqrt(alpha*alpha - beta*beta)
        alpha, beta = alpha/dp, -beta/dp
        # factors[:, col] = alpha*factors[:, col] + beta*cov_k
        mkl.cblas_daxpby(m, beta, y, incy, alpha, x, incy)
        # cov_k = beta/alpha*factors[:, col] + 1/alpha*cov_k
        mkl.cblas_daxpby(m, beta/alpha, x, incy, 1/alpha, y, incy)

cdef int __insert_index(long[::1] order, long[::1] locations, int i, int k):
    """ Finds the index to insert index k into the order. """
    cdef int index = -1
    for index in range(i):
        # bigger than current value, insertion spot
        if locations[k] >= locations[order[index]]:
            return index
    return index + 1

cdef void __select_point(long[::1] order, long[::1] locations, int i, int k,
                         double[:, ::1] points, Kernel *kernel,
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
    covariance_vector(kernel, points, points[k], &factors[0, last])
    __chol_insert(order, i, index, k, factors)

cdef long[::1] __chol_nonadj_select(double[:, ::1] x, long[::1] train,
                                    long[::1] test, Kernel *kernel, int s):
    """ Greedily select the s entries minimizing conditional covariance. """
    # O(m*(n + m)*m + s*(n + m)*(s + m)) = O(n s^2 + n m^2 + m^3)
    cdef:
        int n, m, i, j, k, best_k, col, index
        double best, key, cond_cov_k, cond_var_k, cond_var_j
        long[::1] indexes, order, locations
        double[:, ::1] points
        double[::1, :] factors
        double[::1] var, keys

    n, m = train.shape[0], test.shape[0]
    locations = np.append(train, test)
    points = np.asarray(x)[locations]
    # initialization
    indexes = np.zeros(min(n, s), dtype=np.int64)
    order = np.zeros(indexes.shape[0] + test.shape[0], dtype=np.int64)
    factors = np.zeros((n + m, s + m + 1), order="F")
    # var = kernel.diag(x[train])
    var = np.zeros(n)
    variance_vector(kernel, points[:n], &var[0])
    keys = np.ones(s + m + 2)
    # pre-condition on the m prediction points
    for i in range(m):
        __select_point(order, locations, i, n + i, points, kernel, factors)

    i = 0
    while i < indexes.shape[0]:
        best, best_k = -INFINITY, 0
        # pick best entry
        for j in range(n):
            # selected already, don't consider as candidate
            if var[j] == INFINITY:
                continue

            cond_var_j = var[j]
            index = __insert_index(order, locations, m + i, j)
            keys[m + i] = 1

            for col in range(index):
                k = order[col]
                cond_var_k = factors[k, col]
                cond_cov_k = factors[j, col]*cond_var_k
                cond_var_k *= cond_var_k
                cond_cov_k *= cond_cov_k
                # remove spurious contribution from selected training point
                if k < n:
                    keys[col] = cond_var_k
                cond_var_j -= cond_cov_k/cond_var_k

            # remove spurious contribution of j
            keys[m + i] = cond_var_j

            for col in range(index, m + i):
                k = order[col]
                cond_var_k = factors[k, col]
                cond_cov_k = factors[j, col]*cond_var_k
                cond_var_k *= cond_var_k
                cond_cov_k *= cond_cov_k
                # remove spurious contribution from selected training point
                if k < n:
                    keys[col] = cond_var_k - cond_cov_k/cond_var_j
                cond_var_j -= cond_cov_k/cond_var_k

            # add logdet of entire covariance matrix
            keys[m + i + 1] = 1/cond_var_j

            key = __log_product(m + i + 2, &keys[0], 1)
            if key > best:
                best, best_k = key, j

        indexes[i] = best_k
        # mark as selected
        var[best_k] = INFINITY
        # update Cholesky factor
        __select_point(order, locations, i + m, best_k,
                       points, kernel, factors)
        i += 1

    return indexes

cdef long[::1] __budget_select(double[:, ::1] x, long[::1] train,
                               long[::1] test, Kernel *kernel, int s):
    """ Greedily select the s entries minimizing conditional covariance. """
    # O(m*(n + m)*m + s*(n + m)*(s + m)) = O(n s^2 + n m^2 + m^3)
    cdef:
        int n, m, budget, max_sel, i, j, k, best_k, col, index
        double best, key, cond_cov_k, cond_var_k, cond_var_j
        long[::1] indexes, order, locations
        double[:, ::1] points
        double[::1, :] factors
        double[::1] var, keys

    n, m = train.shape[0], test.shape[0]
    locations = np.append(train, test)
    points = np.asarray(x)[locations]
    # allow each selected point to condition all the prediction points
    budget = m*s
    max_sel = min(n, budget)
    # initialization
    indexes = np.zeros(max_sel, dtype=np.int64)
    order = np.zeros(indexes.shape[0] + test.shape[0], dtype=np.int64)
    factors = np.zeros((n + m, max_sel + m + 1), order="F")
    # var = kernel.diag(x[train])
    var = np.zeros(n)
    variance_vector(kernel, points[:n], &var[0])
    keys = np.ones(max_sel + m + 2)
    # pre-condition on the m prediction points
    for i in range(m):
        __select_point(order, locations, i, n + i, points, kernel, factors)

    i = 0
    while i < indexes.shape[0] and budget > 0:
        best, best_k = -INFINITY, 0
        # pick best entry
        for j in range(n):
            # selected already, don't consider as candidate
            if var[j] == INFINITY:
                continue

            cond_var_j = var[j]
            index = __insert_index(order, locations, m + i, j)
            keys[m + i] = 1

            for col in range(index):
                k = order[col]
                cond_var_k = factors[k, col]
                cond_cov_k = factors[j, col]*cond_var_k
                cond_var_k *= cond_var_k
                cond_cov_k *= cond_cov_k
                # remove spurious contribution from selected training point
                if k < n:
                    keys[col] = cond_var_k
                cond_var_j -= cond_cov_k/cond_var_k

            # remove spurious contribution of j
            keys[m + i] = cond_var_j

            for col in range(index, m + i):
                k = order[col]
                cond_var_k = factors[k, col]
                cond_cov_k = factors[j, col]*cond_var_k
                cond_var_k *= cond_var_k
                cond_cov_k *= cond_cov_k
                # remove spurious contribution from selected training point
                if k < n:
                    keys[col] = cond_var_k - cond_cov_k/cond_var_j
                cond_var_j -= cond_cov_k/cond_var_k

            # add logdet of entire covariance matrix
            keys[m + i + 1] = 1/cond_var_j

            key = __log_product(m + i + 2, &keys[0], 1)
            if key > best:
                best, best_k = key, j

        # subtract number conditioned from budget
        index = train[best_k]
        for j in test:
            budget -= j < index
        if budget < 0:
            break
        # budget -= 1
        indexes[i] = best_k
        # mark as selected
        var[best_k] = INFINITY
        # update Cholesky factor
        __select_point(order, locations, i + m, best_k,
                       points, kernel, factors)
        i += 1

    return indexes[:i]

### wrapper functions

def chol_update(double[::1] cov_k, int i, int k,
                double[::1, :] factors, double[::1] cond_var) -> None:
    """ Updates the ith column of the Cholesky factor with column k. """
    factors[:, i] = cov_k
    __chol_update(cov_k.shape[0], i, k, factors, cond_var)

def select(double[:, ::1] x_train, double[:, ::1] x_test,
           kernel_object, int s) -> np.ndarray:
    """ Wrapper over various cknn selection methods. """
    cdef Kernel *kernel = get_kernel(kernel_object)
    # single prediction point, use specialized function
    if x_test.shape[0] == 1:
        selected = __chol_select(x_train, x_test, kernel, s)
    else:
        selected = __chol_mult_select(x_train, x_test, kernel, s)
    kernel_cleanup(kernel)
    return np.asarray(selected)

def nonadj_select(double[:, ::1] x, long[::1] train, long[::1] test,
                  kernel_object, int s) -> np.ndarray:
    """ Wrapper over various cknn selection methods. """
    cdef Kernel *kernel = get_kernel(kernel_object)
    # single prediction point, use specialized function
    if test.shape[0] == 1:
        points = np.asarray(x)
        selected = __chol_select(points[train], points[test], kernel, s)
    else:
        selected = __chol_nonadj_select(x, train, test, kernel, s)
    kernel_cleanup(kernel)
    return np.asarray(selected)

def chol_select(double[:, ::1] x, long[::1] train, long[::1] test,
                kernel_object, int s) -> np.ndarray:
    """ Wrapper over selection specialized to Cholesky factorization. """
    cdef Kernel *kernel = get_kernel(kernel_object)
    # single prediction point, use specialized function
    if test.shape[0] == 1:
        points = np.asarray(x)
        selected = __chol_select(points[train], points[test], kernel, s)
    else:
        selected = __budget_select(x, train, test, kernel, s)
    kernel_cleanup(kernel)
    return np.asarray(selected)

