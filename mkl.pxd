ctypedef int MKL_INT

cdef extern from "mkl.h":
    # CBLAS routines
    void cblas_daxpby(const MKL_INT N,
                      const double alpha, const double *X, const MKL_INT incX,
                      const double beta, double *Y, const MKL_INT incY)

    # vector math library (VML) routines
    void vdSqrt(const MKL_INT n, const double a[], double r[])
    void vdExp(const MKL_INT n, const double a[], double r[])

