cdef extern from "hpro-c.h":
    struct hpro_d_vector_s: pass
    struct hpro_d_matrix_s: pass
    struct hpro_d_linearoperator_s: pass
    struct hpro_coord_s: pass
    struct hpro_clustertree_s: pass
    struct hpro_permutation_s: pass
    struct hpro_admcond_s: pass
    struct hpro_blockclustertree_s: pass

    ctypedef struct hpro_acc_fixed_eps_t:
        hpro_acc_type_t type
        double eps

    ctypedef struct hpro_acc_fixed_rank_t:
        hpro_acc_type_t type
        unsigned int rank

    ctypedef struct hpro_acc_blocked_t:
        hpro_acc_type_t type
        const hpro_acc_u *subacc

    cdef union hpro_acc_u:
        hpro_acc_type_t type
        hpro_acc_fixed_eps_t fixed_eps
        hpro_acc_fixed_rank_t fixed_rank
        hpro_acc_blocked_t blocked

    ctypedef hpro_acc_u hpro_acc_t

    ctypedef void (*hpro_d_coeff_t)(
        const size_t n,
        const int *rowidx,
        const size_t m,
        const int *colidx,
        double *matrix,
        void *arg
    )

    void hpro_init(int *info)
    void hpro_done(int *info)
    int hpro_is_init()
    void hpro_error_desc(char *desc, const size_t size)

    hpro_coord_s *hpro_coord_import(
        const size_t n,
        const unsigned int dim,
        double **coord,
        const double *period,
        int *info
    )
    size_t hpro_coord_ncoord(const hpro_coord_s *coord, int *info)
    unsigned int hpro_coord_dim(const hpro_coord_s *coord, int *info)
    double *hpro_coord_get(
        const hpro_coord_s *coord,
        const size_t i,
        int *info
    )
    void hpro_coord_free(hpro_coord_s *coord, int *info)

    hpro_clustertree_s *hpro_clt_build_bsp(
        const hpro_coord_s *coord,
        const hpro_bsp_t bsptype,
        const unsigned int nmin,
        int *info
    )
    hpro_permutation_s *hpro_clt_perm_i2e(hpro_clustertree_s *ct, int *info)
    hpro_permutation_s *hpro_clt_perm_e2i(hpro_clustertree_s *ct, int *info)
    void hpro_clt_free(hpro_clustertree_s *ct, int *info)

    hpro_admcond_s *hpro_admcond_geom(
        const hpro_adm_t crit,
        const double eta,
        int *info
    )
    void hpro_admcond_free(const hpro_admcond_s *ac, int *info)

    hpro_blockclustertree_s *hpro_bct_build(
        const hpro_clustertree_s *rowct,
        const hpro_clustertree_s *colct,
        const hpro_admcond_s *ac,
        int *info
    )
    void hpro_bct_free(hpro_blockclustertree_s *bct, int *info)

    hpro_d_matrix_s *hpro_d_matrix_build_coeff(
        const hpro_blockclustertree_s *bct,
        const hpro_d_coeff_t f,
        void *arg,
        const hpro_lrapx_t lrapx,
        const hpro_acc_t acc,
        const int sym,
        int *info
    )
    hpro_d_matrix_s *hpro_d_matrix_copy(const hpro_d_matrix_s *A, int *info)
    size_t hpro_d_matrix_bytesize(const hpro_d_matrix_s *A, int *info)
    void hpro_d_matrix_free(hpro_d_matrix_s *A, int *info)
    hpro_d_linearoperator_s *hpro_d_matrix_ll_inv(
        hpro_d_matrix_s *A,
        const hpro_acc_t acc,
        int *info
    )
    hpro_d_linearoperator_s *hpro_d_matrix_ldl_inv(
        hpro_d_matrix_s *A,
        const hpro_acc_t acc,
        int *info
    )
    void hpro_d_matrix_add_identity(
        hpro_d_matrix_s *A,
        const double lambda_,
        int *info
    )

    hpro_acc_t hpro_acc_fixed_eps(const double eps)

    void hpro_d_vector_free(hpro_d_vector_s *v, int *info)
    hpro_d_vector_s *hpro_d_vector_build(const size_t size, int *info)
    void hpro_d_vector_import(hpro_d_vector_s *v, const double *arr, int *info)
    void hpro_d_vector_export(const hpro_d_vector_s *v, double *arr, int *info)
    void hpro_d_vector_permute(
        hpro_d_vector_s *v,
        const hpro_permutation_s *perm,
        int *info
    )

    void hpro_d_linearoperator_free(hpro_d_linearoperator_s *A, int *info)
    hpro_d_vector_s *hpro_d_linearoperator_range_vector(
        const hpro_d_linearoperator_s *A,
        int *info
    )
    hpro_d_vector_s *hpro_d_linearoperator_domain_vector(
        const hpro_d_linearoperator_s *A,
        int *info
    )
    hpro_d_linearoperator_s *hpro_d_perm_linearoperator(
        const hpro_permutation_s *P,
        const hpro_d_linearoperator_s *A,
        const hpro_permutation_s *R,
        int *info
    )
    void hpro_d_linearoperator_apply(
        const hpro_d_linearoperator_s *A,
        const hpro_d_vector_s *x,
        hpro_d_vector_s *y,
        const hpro_matop_t matop,
        int *info
    )

cdef enum:
    HPRO_NO_ERROR = 0

ctypedef enum hpro_bsp_t:
    HPRO_BSP_AUTO = 0
    HPRO_BSP_GEOM_MAX = 1
    HPRO_BSP_GEOM_REG = 2
    HPRO_BSP_CARD_MAX = 3
    HPRO_BSP_CARD_REG = 4
    HPRO_BSP_PCA = 5

ctypedef enum hpro_alg_t:
    HPRO_ALG_AUTO = 0
    HPRO_ALG_BFS = 1
    HPRO_ALG_ML = 2
    HPRO_ALG_METIS = 3
    HPRO_ALG_SCOTCH = 4

ctypedef enum hpro_adm_t:
    HPRO_ADM_AUTO = 0
    HPRO_ADM_STD_MIN = 1
    HPRO_ADM_STD_MAX = 2
    HPRO_ADM_WEAK = 3

ctypedef enum hpro_acc_type_t:
    HPRO_ACC_FIXED_EPS = 0
    HPRO_ACC_FIXED_RANK = 1
    HPRO_ACC_BLOCKED = 2
    HPRO_ACC_STRING = 3

ctypedef enum hpro_lrapx_t:
    HPRO_LRAPX_SVD = 0
    HPRO_LRAPX_ACA = 1
    HPRO_LRAPX_ACAPLUS = 2
    HPRO_LRAPX_ACAFULL = 3
    HPRO_LRAPX_HCA = 4
    HPRO_LRAPX_ZERO = 5
    HPRO_LRAPX_RANDSVD = 6
    HPRO_LRAPX_RRQR = 7

ctypedef enum hpro_matop_t:
    HPRO_MATOP_NORM = 78  # 'N'
    HPRO_MATOP_TRANS = 84  # 'T'
    HPRO_MATOP_ADJ = 67  # 'C'
