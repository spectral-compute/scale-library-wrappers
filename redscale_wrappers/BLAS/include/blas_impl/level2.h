/**
 * gbmv()
 */
#define gbmv_ARGS trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy
BLAS_API(S, s, gbmv, Gbmv,
    cublasOperation_t trans,
    int m, int n, int kl, int ku,
    const float* alpha,
    const float* A, int lda,
    const float* x, int incx,
    const float* beta,
    float* y, int incy
)
BLAS_API(D, d, gbmv, Gbmv,
    cublasOperation_t trans,
    int m, int n, int kl, int ku,
    const double* alpha,
    const double* A, int lda,
    const double* x, int incx,
    const double* beta,
    double* y, int incy
)
BLAS_API(C, c, gbmv, Gbmv,
    cublasOperation_t trans,
    int m, int n, int kl, int ku,
    const cuComplex* alpha,
    const cuComplex* A, int lda,
    const cuComplex* x, int incx,
    const cuComplex* beta,
    cuComplex* y, int incy
)
BLAS_API(Z, z, gbmv, Gbmv,
    cublasOperation_t trans,
    int m, int n, int kl, int ku,
    const cuDoubleComplex* alpha,
    const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* x, int incx,
    const cuDoubleComplex* beta,
    cuDoubleComplex* y, int incy
)
#undef gbmv_ARGS

/**
 * gemv()
 */
#define gemv_ARGS trans, m, n, alpha, A, lda, x, incx, beta, y, incy
BLAS_API(S, s, gemv, Gemv,
    cublasOperation_t trans,
    int m, int n,
    const float *alpha,
    const float *A, int lda,
    const float *x, int incx,
    const float *beta,
    float *y, int incy
)
BLAS_API(D, d, gemv, Gemv,
    cublasOperation_t trans,
    int m, int n,
    const double* alpha,
    const double* A, int lda,
    const double* x, int incx,
    const double* beta,
    double* y, int incy
)
BLAS_API(C, c, gemv, Gemv,
    cublasOperation_t trans,
    int m, int n,
    const cuComplex *alpha,
    const cuComplex *A, int lda,
    const cuComplex *x, int incx,
    const cuComplex *beta,
    cuComplex *y, int incy
)
BLAS_API(Z, z, gemv, Gemv,
    cublasOperation_t trans,
    int m, int n,
    const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *x, int incx,
    const cuDoubleComplex *beta,
    cuDoubleComplex *y, int incy
)
#undef gemv_ARGS

/**
 * ger()
 */
#define ger_ARGS m, n, alpha, x, incx, y, incy, A, lda
#define gerc_ARGS ger_ARGS
#define geru_ARGS ger_ARGS
BLAS_API(S, s, ger, Ger,
    int m, int n,
    const float* alpha,
    const float* x, int incx,
    const float* y, int incy,
    float* A, int lda
)
BLAS_API(D, d, ger, Ger,
    int m, int n,
    const double* alpha,
    const double* x, int incx,
    const double* y, int incy,
    double* A, int lda
)
BLAS_API(C, c, geru, Geru,
    int m, int n,
    const cuComplex* alpha,
    const cuComplex* x, int incx,
    const cuComplex* y, int incy,
    cuComplex* A, int lda
)
BLAS_API(C, c, gerc, Gerc,
    int m, int n,
    const cuComplex* alpha,
    const cuComplex* x, int incx,
    const cuComplex* y, int incy,
    cuComplex* A, int lda
)

BLAS_API(Z, z, geru, Geru,
    int m, int n,
    const cuDoubleComplex* alpha,
    const cuDoubleComplex* x, int incx,
    const cuDoubleComplex* y, int incy,
    cuDoubleComplex* A, int lda
)
BLAS_API(Z, z, gerc, Gerc,
    int m, int n,
    const cuDoubleComplex* alpha,
    const cuDoubleComplex* x, int incx,
    const cuDoubleComplex* y, int incy,
    cuDoubleComplex* A, int lda
)

#undef ger_ARGS
#undef gerc_ARGS
#undef geru_ARGS

/**
 * sbmv()
 */
#define sbmv_ARGS fillMode, n, k, alpha, A, lda, x, incx, beta, y, incy
BLAS_API(S, s, sbmv, Sbmv,
    cublasFillMode_t fillMode,
    int n, int k, const float *alpha,
    const float *A, int lda,
    const float *x, int incx,
    const float *beta, float *y, int incy
)
BLAS_API(D, d, sbmv, Sbmv,
    cublasFillMode_t fillMode,
    int n, int k, const double *alpha,
    const double *A, int lda,
    const double *x, int incx,
    const double *beta, double *y, int incy
)
#undef sbmv_ARGS

/**
 * spmv()
 */
#define spmv_ARGS fillMode, n, alpha, A, x, incx, beta, y, incy
BLAS_API(S, s, spmv, Spmv,
    cublasFillMode_t fillMode,
    int n, const float *alpha,
    const float *A,
    const float *x, int incx,
    const float *beta, float *y, int incy
)
BLAS_API(D, d, spmv, Spmv,
    cublasFillMode_t fillMode,
    int n, const double *alpha,
    const double *A,
    const double *x, int incx,
    const double *beta, double *y, int incy
)
#undef spmv_ARGS

/**
 * spr()
 */
#define spr_ARGS fillMode, n, alpha, x, incx, A
BLAS_API(S, s, spr, Spr,
    cublasFillMode_t fillMode,
    int n, const float *alpha,
    const float *x, int incx,
    float* A
)
BLAS_API(D, d, spr, Spr,
    cublasFillMode_t fillMode,
    int n, const double* alpha,
    const double* x, int incx,
    double* A
)
#undef spr_ARGS

/**
 * spr2()
 */
#define spr2_ARGS fillMode, n, alpha, x, incx, y, incy, A
BLAS_API(S, s, spr2, Spr2,
    cublasFillMode_t fillMode,
    int n, const float *alpha,
    const float *x, int incx,
    const float *y, int incy,
    float* A
)
BLAS_API(D, d, spr2, Spr2,
    cublasFillMode_t fillMode,
    int n, const double* alpha,
    const double* x, int incx,
    const double* y, int incy,
    double* A
)
#undef spr2_ARGS

/**
 * symv()
 */
#define symv_ARGS fillMode, n, alpha, A, lda, x, incx, beta, y, incy
BLAS_API(S, s, symv, Symv,
    cublasFillMode_t fillMode,
    int n,
    const float *alpha,
    const float *A, int lda,
    const float *x, int incx,
    const float *beta,
    float *y, int incy
)
BLAS_API(D, d, symv, Symv,
    cublasFillMode_t fillMode,
    int n,
    const double *alpha,
    const double *A, int lda,
    const double *x, int incx,
    const double *beta,
    double *y, int incy
)
BLAS_API(C, c, symv, Symv,
    cublasFillMode_t fillMode,
    int n,
    const cuComplex *alpha,
    const cuComplex *A, int lda,
    const cuComplex *x, int incx,
    const cuComplex *beta,
    cuComplex *y, int incy
)
BLAS_API(Z, z, symv, Symv,
    cublasFillMode_t fillMode,
    int n,
    const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *x, int incx,
    const cuDoubleComplex *beta,
    cuDoubleComplex *y, int incy
)
#undef symv_ARGS

/**
 * syr()
 */
#define syr_ARGS fillMode, n, alpha, x ,incx, A, lda
BLAS_API(S, s, syr, Syr,
    cublasFillMode_t fillMode, int n,
    const float* alpha,
    const float* x, int incx,
    float* A, int lda
)
BLAS_API(D, d, syr, Syr,
    cublasFillMode_t fillMode,
    int n, const double* alpha,
    const double* x, int incx,
    double* A, int lda
)
BLAS_API(C, c, syr, Syr,
    cublasFillMode_t fillMode,
    int n, const cuComplex* alpha,
    const cuComplex* x, int incx,
    cuComplex* A, int lda
)
BLAS_API(Z, z, syr, Syr,
    cublasFillMode_t fillMode,
    int n, const cuDoubleComplex* alpha,
    const cuDoubleComplex* x, int incx,
    cuDoubleComplex* A, int lda
)
#undef syr_ARGS

/**
 * syr2()
 */
#define syr2_ARGS fillMode, n, alpha, x, incx, y, incy, A, lda
BLAS_API(S, s, syr2, Syr2,
    cublasFillMode_t fillMode, int n,
    const float* alpha,
    const float* x, int incx,
    const float* y, int incy,
    float* A, int lda
)
BLAS_API(D, d, syr2, Syr2,
    cublasFillMode_t fillMode,
    int n, const double* alpha,
    const double* x, int incx,
    const double* y, int incy,
    double* A, int lda
)
BLAS_API(C, c, syr2, Syr2,
    cublasFillMode_t fillMode,
    int n, const cuComplex* alpha,
    const cuComplex* x, int incx,
    const cuComplex* y, int incy,
    cuComplex* A, int lda
)
BLAS_API(Z, z, syr2, Syr2,
    cublasFillMode_t fillMode, int n,
    const cuDoubleComplex* alpha,
    const cuDoubleComplex* x, int incx,
    const cuDoubleComplex* y, int incy,
    cuDoubleComplex* A, int lda
)
#undef syr2_ARGS

/**
 * tbmv()
 */
#define tbmv_ARGS fillMode, trans, diag, n, k, A, lda, x, incx
BLAS_API(S, s, tbmv, Tbmv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, int k, const float* A, int lda,
    float* x, int incx
)
BLAS_API(D, d, tbmv, Tbmv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, int k, const double* A, int lda,
    double* x, int incx
)
BLAS_API(C, c, tbmv, Tbmv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, int k, const cuComplex* A, int lda,
    cuComplex* x, int incx
)
BLAS_API(Z, z, tbmv, Tbmv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, int k, const cuDoubleComplex* A, int lda,
    cuDoubleComplex* x, int incx
)
#undef tbmv_ARGS

/**
 * tbsv()
 */
#define tbsv_ARGS fillMode, trans, diag, n, k, A, lda, x, incx
BLAS_API(S, s, tbsv, Tbsv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, int k, const float* A, int lda,
    float* x, int incx
)
BLAS_API(D, d, tbsv, Tbsv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, int k, const double* A, int lda,
    double* x, int incx
)
BLAS_API(C, c, tbsv, Tbsv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, int k, const cuComplex* A, int lda,
    cuComplex* x, int incx
)
BLAS_API(Z, z, tbsv, Tbsv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, int k, const cuDoubleComplex* A, int lda,
    cuDoubleComplex* x, int incx
)
#undef tbsv_ARGS

/**
 * tpmv()
 */
#define tpmv_ARGS fillMode, trans, diag, n, A, x, incx
BLAS_API(S, s, tpmv, Tpmv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, const float* A,
    float* x, int incx
)
BLAS_API(D, d, tpmv, Tpmv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, const double* A,
    double* x, int incx
)
BLAS_API(C, c, tpmv, Tpmv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, const cuComplex* A,
    cuComplex* x, int incx
)
BLAS_API(Z, z, tpmv, Tpmv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, const cuDoubleComplex* A,
    cuDoubleComplex* x, int incx
)
#undef tpmv_ARGS

/**
 * tpsv()
 */
#define tpsv_ARGS fillMode, trans, diag, n, A, x, incx
BLAS_API(S, s, tpsv, Tpsv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, const float* A,
    float* x, int incx
)
BLAS_API(D, d, tpsv, Tpsv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, const double* A,
    double* x, int incx
)
BLAS_API(C, c, tpsv, Tpsv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, const cuComplex* A,
    cuComplex* x, int incx
)
BLAS_API(Z, z, tpsv, Tpsv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, const cuDoubleComplex* A,
    cuDoubleComplex* x, int incx
)
#undef tpsv_ARGS

/**
 * trmv()
 */
#define trmv_ARGS fillMode, trans, diag, n, A, lda, x, incx
BLAS_API(S, s, trmv, Trmv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, const float* A, int lda,
    float* x, int incx
)
BLAS_API(D, d, trmv, Trmv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, const double* A, int lda,
    double* x, int incx
)
BLAS_API(C, c, trmv, Trmv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, const cuComplex* A, int lda,
    cuComplex* x, int incx
)
BLAS_API(Z, z, trmv, Trmv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, const cuDoubleComplex* A, int lda,
    cuDoubleComplex* x, int incx
)
#undef trmv_ARGS

/**
 * trsv()
 */
#define trsv_ARGS fillMode, trans, diag, n, A, lda, x, incx
BLAS_API(S, s, trsv, Trsv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, const float* A, int lda,
    float* x, int incx
)
BLAS_API(D, d, trsv, Trsv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, const double* A, int lda,
    double* x, int incx
)
BLAS_API(C, c, trsv, Trsv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, const cuComplex* A, int lda,
    cuComplex* x, int incx
)
BLAS_API(Z, z, trsv, Trsv,
    cublasFillMode_t fillMode,
    cublasOperation_t trans, cublasDiagType_t diag,
    int n, const cuDoubleComplex* A, int lda,
    cuDoubleComplex* x, int incx
)
#undef trsv_ARGS

/**
 * hemv()
 */
#define hemv_ARGS fillMode, n, alpha, A, lda, x, incx, beta, y, incy
BLAS_API(C, c, hemv, Hemv,
    cublasFillMode_t fillMode,
    int n,
    const cuComplex *alpha,
    const cuComplex *A, int lda,
    const cuComplex *x, int incx,
    const cuComplex *beta,
    cuComplex *y, int incy
)
BLAS_API(Z, z, hemv, Hemv,
    cublasFillMode_t fillMode,
    int n,
    const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *x, int incx,
    const cuDoubleComplex *beta,
    cuDoubleComplex *y, int incy
)
#undef hemv_ARGS

/**
 * hbmv()
 */
#define hbmv_ARGS fillMode, n, k, alpha, A, lda, x, incx, beta, y, incy
BLAS_API(C, c, hbmv, Hbmv,
    cublasFillMode_t fillMode,
    int n, int k, const cuComplex* alpha,
    const cuComplex* A, int lda,
    const cuComplex* x, int incx,
    const cuComplex* beta,
    cuComplex* y, int incy
)
BLAS_API(Z, z, hbmv, Hbmv,
    cublasFillMode_t fillMode,
    int n, int k, const cuDoubleComplex* alpha,
    const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* x, int incx,
    const cuDoubleComplex* beta,
    cuDoubleComplex* y, int incy
)
#undef hbmv_ARGS

/**
 * hpmv()
 */
#define hpmv_ARGS fillMode, n, alpha, AP, x, incx, beta, y, incy
BLAS_API(C, c, hpmv, Hpmv,
    cublasFillMode_t fillMode,
    int n, const cuComplex* alpha,
    const cuComplex* AP,
    const cuComplex* x, int incx,
    const cuComplex* beta,
    cuComplex* y, int incy
)
BLAS_API(Z, z, hpmv, Hpmv,
    cublasFillMode_t fillMode,
    int n, const cuDoubleComplex* alpha,
    const cuDoubleComplex* AP,
    const cuDoubleComplex* x, int incx,
    const cuDoubleComplex* beta,
    cuDoubleComplex* y, int incy
)
#undef hpmv_ARGS

/**
 * her()
 */
#define her_ARGS fillMode, n, alpha, x, incx, A, lda
BLAS_API(C, c, her, Her,
    cublasFillMode_t fillMode,
    int n, const float* alpha,
    const cuComplex* x, int incx,
    cuComplex* A, int lda
)
BLAS_API(Z, z, her, Her,
    cublasFillMode_t fillMode,
    int n, const double* alpha,
    const cuDoubleComplex* x, int incx,
    cuDoubleComplex* A, int lda
)
#undef her_ARGS

/**
 * her2()
 */
#define her2_ARGS fillMode, n, alpha, x, incx, y, incy, A, lda
BLAS_API(C, c, her2, Her2,
    cublasFillMode_t fillMode,
    int n, const cuComplex* alpha,
    const cuComplex* x, int incx,
    const cuComplex* y, int incy,
    cuComplex* A, int lda
)
BLAS_API(Z, z, her2, Her2,
    cublasFillMode_t fillMode,
    int n, const cuDoubleComplex* alpha,
    const cuDoubleComplex* x, int incx,
    const cuDoubleComplex* y, int incy,
    cuDoubleComplex* A, int lda
)
#undef her2_ARGS

/**
 * hpr()
 */
#define hpr_ARGS fillMode, n, alpha, x, incx, A
BLAS_API(C, c, hpr, Hpr,
    cublasFillMode_t fillMode,
    int n, const float* alpha,
    const cuComplex* x, int incx,
    cuComplex* A
)
BLAS_API(Z, z, hpr, Hpr,
    cublasFillMode_t fillMode,
    int n, const double* alpha,
    const cuDoubleComplex* x, int incx,
    cuDoubleComplex* A
)
#undef hpr_ARGS

/**
 * hpr2()
 */
#define hpr2_ARGS fillMode, n, alpha, x, incx, y, incy, A
BLAS_API(C, c, hpr2, Hpr2,
    cublasFillMode_t fillMode,
    int n, const cuComplex* alpha,
    const cuComplex* x, int incx,
    const cuComplex* y, int incy,
    cuComplex* A
)
BLAS_API(Z, z, hpr2, Hpr2,
    cublasFillMode_t fillMode,
    int n, const cuDoubleComplex* alpha,
    const cuDoubleComplex* x, int incx,
    const cuDoubleComplex* y, int incy,
    cuDoubleComplex* A
)
#undef hpr2_ARGS
