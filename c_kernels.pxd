cdef struct Kernel:
    void *params
    void (*kernel_function)(void *params, double[:, ::1] points,
                            double[::1] point, double *vector)
    void (*diag)(void *params, double[:, ::1] points, double *vector)
    bint cleanup

cdef (Kernel *) get_kernel(kernel)

cdef void covariance_vector(Kernel *kernel, double[:, ::1] points,
                            double[::1] point, double *vector)

cdef void variance_vector(Kernel *kernel, double[:, ::1] points,
                          double *vector)

cdef void kernel_cleanup(Kernel *kernel)

