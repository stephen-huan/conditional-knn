# cython: profile=False
from libc.math cimport INFINITY, sqrt, log2
cimport numpy as np
# we rely on importing numpy to also load intel mkl libraries, see README.md
import numpy as np
cimport scipy.linalg.cython_blas as blas
cimport mkl
from c_kernels cimport Kernel, get_kernel, kernel_cleanup
from c_kernels cimport covariance_vector, variance_vector
cimport maxheap
from maxheap cimport Heap
cimport sequence
from sequence cimport Sequence

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

cdef void __select_update(double[:, ::1] x_train, double[:, ::1] x_test,
                          Kernel *kernel, int i, int k, double[::1, :] factors,
                          double[::1] cond_var, double[::1] cond_cov):
    """ Update the selection data structures after selecting a point. """
    cdef:
        int n, incx
        double alpha

    n = x_train.shape[0]
    # update Cholesky factors
    covariance_vector(kernel, x_train, x_train[k], &factors[0, i])
    covariance_vector(kernel, x_test, x_train[k], &factors[n, i])
    __chol_update(n + 1, i, k, factors, cond_var)
    # update conditional covariance
    # cond_cov -= factors[:, i][:n]*factors[n, i]
    alpha = -factors[n, i]
    incx = 1
    blas.daxpy(&n, &alpha, &factors[0, i], &incx, &cond_cov[0], &incx)

cdef long[::1] __chol_select(double[:, ::1] x_train, double[:, ::1] x_test,
                             Kernel *kernel, int s):
    """ Select the s most informative entries, storing a Cholesky factor. """
    # O(s*(n*s)) = O(n s^2)
    cdef:
        int n, i, j, k
        double best
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
        # update data structures
        __select_update(x_train, x_test, kernel, i, k,
                        factors, cond_var, cond_cov)
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
        unsigned long long *y
        unsigned long long value
        double mantissa
        int i, exponent

    y = <unsigned long long *> x
    mantissa = 1
    exponent = 0
    for i in range(n):
        value = y[i*incx]
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

### global selection

cdef long[:, ::1] __global_select(double[:, ::1] points,
                                  Kernel *kernel, int nonzeros,
                                  list group_candidates, list ref_groups):
    """ Construct a sparsity pattern from a candidate set over all columns. """
    cdef:
        int n, i, j, k, total_candidates, entries_left, max_sel, entry, lda, s
        long[::1] group_sizes, candidate_sizes, group, candidates, ids
        long[:, ::1] indexes
        double key, var
        double[::1] group_var, values, cond_var, cond_cov
        double[::1, :] factor
        Sequence *group_list
        Sequence *candidate_list
        Sequence *factors
        Sequence *cond_covs
        Sequence *cond_vars
        Heap heap

    n = points.shape[0]
    x = np.asarray(points)

    group_list = sequence.from_list(ref_groups)
    group_sizes = sequence.size_list(ref_groups)
    candidate_list = sequence.from_list(group_candidates)
    candidate_sizes = sequence.size_list(group_candidates)

    total_candidates = 0
    for i in range(candidate_sizes.shape[0]):
        total_candidates += candidate_sizes[i]
    entries_left = nonzeros
    for i in range(group_sizes.shape[0]):
        j = group_sizes[i]
        entries_left -= j*(j + 1)//2
    max_sel = 3*(entries_left//n)

    # reserve last entry for number selected
    indexes = np.zeros((group_list.size, max_sel + 1), dtype=np.int64)

    # initialize data structures
    factors = sequence.new_sequence(group_list.size)
    cond_covs = sequence.new_sequence(group_list.size)
    cond_vars = sequence.new_sequence(group_list.size)
    for i in range(group_list.size):
        group = <long[:group_sizes[i]:1]> group_list.data[i]
        lda = candidate_sizes[i]
        if lda == 0:
            continue
        candidates = <long[:lda:1]> candidate_list.data[i]
        sequence.add_item(factors, i, (lda + group.shape[0])*max_sel)
        sequence.add_item(cond_covs, i, lda)
        covariance_vector(kernel, x[candidates], x[group[0]],
                          <double *> cond_covs.data[i])
        sequence.add_item(cond_vars, i, lda)
        variance_vector(kernel, x[candidates], <double *> cond_vars.data[i])

    group_var = np.zeros(group_list.size)
    variance_vector(kernel, x[[group[0] for group in ref_groups]],
                    &group_var[0])

    # add candidates to max heap
    values = np.zeros(total_candidates)
    ids = np.zeros(total_candidates, dtype=np.int64)
    k = 0
    for i in range(group_list.size):
        var = group_var[i]
        lda = candidate_sizes[i]
        if lda == 0:
            continue
        cond_cov = <double[:lda:1]> cond_covs.data[i]
        cond_var = <double[:lda:1]> cond_vars.data[i]
        for j in range(candidate_sizes[i]):
            values[k] = (cond_cov[j]*cond_cov[j]/cond_var[j])/var
            ids[k] = n*j + i
            k += 1

    heap = Heap(values, ids)
    while heap.size > 0 and entries_left > 0:
        entry = maxheap.__get_id(heap.__pop())
        k, i = entry//n, entry % n
        # do not select if group already has enough entries
        s = indexes[i, max_sel]
        if s >= max_sel:
            continue
        lda = candidate_sizes[i]
        candidates = <long[:lda:1]> candidate_list.data[i]
        cond_cov = <double[:lda:1]> cond_covs.data[i]
        cond_var = <double[:lda:1]> cond_vars.data[i]
        group = <long[:group_sizes[i]:1]> group_list.data[i]
        lda = candidates.shape[0] + group.shape[0]
        factor = <double[:lda:1, :max_sel]> factors.data[i]
        # add entry to sparsity pattern
        indexes[i, s] = candidates[k]
        # update data structures
        group_var[i] -= cond_cov[k]*cond_cov[k]/cond_var[k]
        __select_update(x[candidates], x[group], kernel, s, k,
                        factor, cond_var, cond_cov)
        # update affected candidates
        for j in range(candidates.shape[0]):
            # if hasn't been selected already
            if cond_var[j] != INFINITY:
                key = (cond_cov[j]*cond_cov[j]/cond_var[j])/group_var[i]
                heap.__update_key(n*j + i, key)
        indexes[i, max_sel] += 1
        entries_left -= group_sizes[i]

    sequence.cleanup(group_list, &group_sizes[0])
    sequence.cleanup(candidate_list, &candidate_sizes[0])
    sequence.cleanup(factors)
    sequence.cleanup(cond_covs)
    sequence.cleanup(cond_vars)
    return indexes

def global_select(double[:, ::1] x, kernel_object, dict ref_sparsity,
                  dict candidate_sparsity, list ref_groups) -> dict:
    """ Construct a sparsity pattern from a candidate set over all columns. """
    cdef:
        int i, nonzeros, max_sel
        Kernel *kernel
        long[:, ::1] indexes

    nonzeros = sum(map(len, ref_sparsity.values()))
    groups = [np.array(group, dtype=np.int64) for group in ref_groups]
    group_candidates = [np.array(list(
        {j for i in group for j in candidate_sparsity[i]} - set(group)
    ), dtype=np.int64) for group in ref_groups]

    kernel = get_kernel(kernel_object)
    indexes = __global_select(x, kernel, nonzeros, group_candidates, groups)
    kernel_cleanup(kernel)

    max_sel = indexes.shape[1] - 1
    sparsity = {group[0]: sorted(group + \
                                 list(indexes[i, :indexes[i, max_sel]]))
                for i, group in enumerate(ref_groups)}
    return sparsity

