from cpython.mem cimport PyMem_Free, PyMem_Malloc
from libc.math cimport exp, sqrt
from libc.stdio cimport printf

import numpy as np
import sklearn.gaussian_process.kernels as kernels

from .hpro cimport (
    HPRO_ADM_STD_MIN,
    HPRO_BSP_AUTO,
    HPRO_LRAPX_ACAPLUS,
    HPRO_MATOP_NORM,
    HPRO_NO_ERROR,
    hpro_acc_fixed_eps,
    hpro_acc_t,
    hpro_admcond_free,
    hpro_admcond_geom,
    hpro_admcond_s,
    hpro_bct_build,
    hpro_bct_free,
    hpro_blockclustertree_s,
    hpro_clt_build_bsp,
    hpro_clt_free,
    hpro_clt_perm_e2i,
    hpro_clt_perm_i2e,
    hpro_clustertree_s,
    hpro_coord_dim,
    hpro_coord_free,
    hpro_coord_get,
    hpro_coord_import,
    hpro_coord_ncoord,
    hpro_coord_s,
    hpro_d_linearoperator_apply,
    hpro_d_linearoperator_domain_vector,
    hpro_d_linearoperator_free,
    hpro_d_linearoperator_range_vector,
    hpro_d_linearoperator_s,
    hpro_d_matrix_add_identity,
    hpro_d_matrix_build_coeff,
    hpro_d_matrix_bytesize,
    hpro_d_matrix_copy,
    hpro_d_matrix_free,
    hpro_d_matrix_ll_inv,
    hpro_d_matrix_s,
    hpro_d_perm_linearoperator,
    hpro_d_vector_build,
    hpro_d_vector_export,
    hpro_d_vector_free,
    hpro_d_vector_import,
    hpro_d_vector_permute,
    hpro_d_vector_s,
    hpro_done,
    hpro_error_desc,
    hpro_init,
    hpro_is_init,
    hpro_permutation_s,
)


cdef char buf[1024]


cdef struct Kernel:
    hpro_coord_s *coord
    double nu
    double length_scale


cdef struct Data:
    hpro_clustertree_s *ct
    hpro_d_matrix_s *A
    hpro_d_linearoperator_s *linop
    hpro_d_vector_s *x
    hpro_d_vector_s *y
    int inverse


cdef void check_info(int info):
    """Check the status code for errors."""
    if info != HPRO_NO_ERROR:
        hpro_error_desc(buf, 1024)
        printf("\n%s\n\n", buf)
        exit(1)


cdef double **to_coord(double[:, ::1] points):
    """Convert an array of points to coordinates."""
    cdef:
        int info
        size_t n, d, i, j
        double **x

    n, d = points.shape[0], points.shape[1]
    x = <double **> PyMem_Malloc(n * sizeof(double *))
    for i in range(n):
        x[i] = <double *> PyMem_Malloc(d * sizeof(double))
        for j in range(d):
            x[i][j] = points[i, j]
    return x


cdef void free_coord(hpro_coord_s *coord, double **x):
    """Free memory associated to the coordinates."""
    cdef:
        int info
        size_t n, i

    n = hpro_coord_ncoord(coord, &info)
    check_info(info)
    hpro_coord_free(coord, &info)
    check_info(info)
    for i in range(n):
        PyMem_Free(x[i])
    PyMem_Free(x)


cdef hpro_d_linearoperator_s *safe_ll_inv(
    hpro_d_matrix_s *A, const hpro_acc_t acc, int verbose
):
    """Inverse Cholesky factorization with restarting."""
    cdef:
        int info
        double eps
        hpro_d_matrix_s *B
        hpro_d_linearoperator_s *linop

    # eps = sqrt(1e-15)
    eps = 1e-1
    B = hpro_d_matrix_copy(A, &info)
    linop = hpro_d_matrix_ll_inv(B, acc, &info)
    while info != HPRO_NO_ERROR and eps < 1e5:
        if verbose:
            printf("Restarting with eps = %e\n", eps)
            # hpro_error_desc(buf, 1024)
            # printf("\n%s\n\n", buf)
        hpro_d_matrix_free(B, &info)
        check_info(info)
        B = hpro_d_matrix_copy(A, &info)
        check_info(info)
        hpro_d_matrix_add_identity(B, eps, &info)
        check_info(info)
        linop = hpro_d_matrix_ll_inv(B, acc, &info)
        # eps *= 10
        eps *= 1e2
    check_info(info)
    return linop


cdef double matern(
    size_t d, double *x, double *y, double nu, double length_scale
) noexcept:
    """Matern covariance between points x and y."""
    cdef:
        size_t i
        double r, diff

    r = 0
    for i in range(d):
        diff = (x[i] - y[i]) / length_scale
        r += diff * diff
    r = sqrt(r)
    if nu == 0.5:
        return exp(-r)
    elif nu == 1.5:
        r *= sqrt(3)
        return (1 + r) * exp(-r)
    elif nu == 2.5:
        r *= sqrt(5)
        return (1 + r + r * r / 3) * exp(-r)
    else:
        return exp(-r * r)


cdef void kernel_function(
    const size_t n,
    const int *rowidx,
    const size_t m,
    const int *colidx,
    double *matrix,
    void *arg,
) noexcept:
    """Kernel function callback."""
    cdef:
        int info
        size_t d, i, j
        Kernel *kernel
        double *x
        double *y

    kernel = <Kernel *> arg
    d = hpro_coord_dim(kernel.coord, &info)
    check_info(info)
    for j in range(m):
        y = hpro_coord_get(kernel.coord, colidx[j], &info)
        check_info(info)
        for i in range(n):
            x = hpro_coord_get(kernel.coord, rowidx[i], &info)
            check_info(info)
            matrix[j * n + i] = matern(d, x, y, kernel.nu, kernel.length_scale)


cdef Data *__init_matvec(
    double nu,
    double length_scale,
    double[:, ::1] points,
    double eps,
    int inverse
):
    """Initialize the matrix-vector product."""
    cdef:
        int info
        size_t n, d
        Data *data
        Kernel *kernel
        double **x
        hpro_coord_s *coord
        hpro_admcond_s *adm
        hpro_blockclustertree_s *bct
        hpro_acc_t acc

    data = <Data *> PyMem_Malloc(sizeof(Data))
    data.inverse = inverse
    n, d = points.shape[0], points.shape[1]
    x = to_coord(points)
    coord = hpro_coord_import(n, d, x, NULL, &info)
    check_info(info)
    data.ct = hpro_clt_build_bsp(coord, HPRO_BSP_AUTO, 20, &info)
    check_info(info)
    adm = hpro_admcond_geom(HPRO_ADM_STD_MIN, 2.0, &info)
    check_info(info)
    bct = hpro_bct_build(data.ct, data.ct, adm, &info)

    acc = hpro_acc_fixed_eps(eps);
    kernel = <Kernel *> PyMem_Malloc(sizeof(Kernel))
    kernel.coord = coord
    kernel.nu = nu
    kernel.length_scale = length_scale
    data.A = hpro_d_matrix_build_coeff(
        bct, kernel_function, kernel, HPRO_LRAPX_ACAPLUS, acc, 1, &info
    )
    check_info(info)

    PyMem_Free(kernel)
    hpro_bct_free(bct, &info)
    check_info(info)
    hpro_admcond_free(adm, &info)
    check_info(info)
    free_coord(coord, x)

    if inverse:
        data.linop = safe_ll_inv(data.A, acc, 1)
        check_info(info)
    else:
        data.linop = <hpro_d_linearoperator_s *> data.A

    data.x = hpro_d_linearoperator_domain_vector(data.linop, &info)
    check_info(info)
    data.y = hpro_d_linearoperator_range_vector(data.linop, &info)
    check_info(info)
    return data


cdef void __matvec(Data *data, double[::1] x):
    """Perform a matrix-vector product."""
    cdef:
        int info
        hpro_d_linearoperator_s *linop
        hpro_permutation_s *e2i
        hpro_permutation_s *i2e

    linop = data.linop
    e2i = hpro_clt_perm_e2i(data.ct, &info)
    check_info(info)
    i2e = hpro_clt_perm_i2e(data.ct, &info)
    check_info(info)
    hpro_d_vector_import(data.x, &x[0], &info)
    check_info(info)
    hpro_d_vector_permute(data.x, e2i, &info)
    check_info(info)
    hpro_d_linearoperator_apply(linop, data.x, data.y, HPRO_MATOP_NORM, &info)
    check_info(info)
    hpro_d_vector_permute(data.y, i2e, &info)
    check_info(info)
    hpro_d_vector_export(data.y, &x[0], &info)
    check_info(info)


cdef void __clean_matvec(Data *data):
    """Clean up after the matrix-vector product."""
    cdef int info

    hpro_d_vector_free(data.y, &info)
    check_info(info)
    hpro_d_vector_free(data.x, &info)
    check_info(info)
    if data.inverse:
        hpro_d_linearoperator_free(data.linop, &info)
        check_info(info)
    hpro_d_matrix_free(data.A, &info)
    check_info(info)
    hpro_clt_free(data.ct, &info)
    check_info(info)
    PyMem_Free(data)


def gram(
    kernel: kernels.Kernel, x: np.ndarray, eps: float=1e-4, inverse: bool=False
) -> np.ndarray:
    """Gram matrix-vector product with hierarchical matrices."""
    cdef:
        int info
        double nu, length_scale
        Data *data

    assert isinstance(kernel, kernels.Matern), f"{kernel} not supported."
    params = kernel.get_params()
    nu = params["nu"]
    length_scale = params["length_scale"]
    data = __init_matvec(nu, length_scale, x, eps, int(inverse))

    def matvec(y: np.ndarray) -> np.ndarray:
        """Matrix-vector product."""
        z = np.copy(y).flatten()
        assert z.size == x.shape[0], "wrong dimension"
        __matvec(data, z)
        return z

    def done() -> None:
        """Finish with the matvec."""
        __clean_matvec(data)

    nnz = hpro_d_matrix_bytesize(data.A, &info) / sizeof(double)
    check_info(info)
    return matvec, done, nnz


cdef __initialize():
    """Initialize the library."""
    cdef int info

    if hpro_is_init() == 0:
        hpro_init(&info)
        check_info(info)


def initialize():
    """Initialize the library."""
    __initialize()


def __finish():
    """Finalize the library."""
    cdef int info

    hpro_done(&info)
    check_info(info)


def finish():
    """Finalize the library."""
    __finish()
