/**
 * amax()
 */
#define amax_ARGS n, x, incx
LEGACY_BLAS_API_RET(int, Is, amax, int n, const float *x, int incx)
LEGACY_BLAS_API_RET(int, Id, amax, int n, const double *x, int incx)
LEGACY_BLAS_API_RET(int, Ic, amax, int n, const cuComplex *x, int incx)
LEGACY_BLAS_API_RET(int, Iz, amax, int n, const cuDoubleComplex *x, int incx)
#undef amax_ARGS

/**
 * amin()
 */
#define amin_ARGS n, x, incx
LEGACY_BLAS_API_RET(int, Is, amin, int n, const float *x, int incx)
LEGACY_BLAS_API_RET(int, Id, amin, int n, const double *x, int incx)
LEGACY_BLAS_API_RET(int, Ic, amin, int n, const cuComplex *x, int incx)
LEGACY_BLAS_API_RET(int, Iz, amin, int n, const cuDoubleComplex *x, int incx)
#undef amin_ARGS

/**
 * asum()
 */
#define asum_ARGS n, x, incx
LEGACY_BLAS_API_RET(float, S, asum, int n, const float* x, int incx)
LEGACY_BLAS_API_RET(double, D, asum, int n, const double* x, int incx)
LEGACY_BLAS_API_RET(float, Sc, asum, int n, const cuComplex* x, int incx)
LEGACY_BLAS_API_RET(double, Dz, asum, int n, const cuDoubleComplex* x, int incx)
#undef asum_ARGS

/**
 * axpy()
 */
#define axpy_ARGS n, &alpha, x, incx, y, incy
/* LEGACY_BLAS_API(Haxpy, int n, const __half *alpha, const __half *x, int incx, __half *y,  int incy) */
LEGACY_BLAS_API(S, axpy, int n, float alpha, const float *x, int incx, float *y,  int incy)
LEGACY_BLAS_API(D, axpy, int n, double alpha, const double *x, int incx, double *y,  int incy)
LEGACY_BLAS_API(C, axpy, int n, cuComplex alpha, const cuComplex *x, int incx, cuComplex *y,  int incy)
LEGACY_BLAS_API(Z, axpy, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy)
#undef axpy_ARGS

/**
 * copy()
 */
#define copy_ARGS n, x, incx, y, incy
LEGACY_BLAS_API(S, copy, int n, const float *x, int incx, float* y, int incy)
LEGACY_BLAS_API(D, copy, int n, const double *x, int incx, double* y, int incy)
LEGACY_BLAS_API(C, copy, int n, const cuComplex *x, int incx, cuComplex* y, int incy)
LEGACY_BLAS_API(Z, copy, int n, const cuDoubleComplex *x, int incx, cuDoubleComplex* y, int incy)
#undef copy_ARGS

/**
 * dot()
 */
#define dot_ARGS n, x, incx, y, incy
#define dotu_ARGS dot_ARGS
#define dotc_ARGS dot_ARGS
LEGACY_BLAS_API_RET(float, S, dot,  int n, const float *x, int incx, const float *y, int incy)
LEGACY_BLAS_API_RET(double, D, dot,  int n, const double *x, int incx, const double *y, int incy)
LEGACY_BLAS_API_RET(cuComplex, C, dotu, int n, const cuComplex *x, int incx, const cuComplex *y, int incy)
LEGACY_BLAS_API_RET(cuDoubleComplex, Z, dotu, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy)
LEGACY_BLAS_API_RET(cuComplex, C, dotc, int n, const cuComplex *x, int incx, const cuComplex *y, int incy)
LEGACY_BLAS_API_RET(cuDoubleComplex, Z, dotc, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy)
#undef dot_ARGS
#undef dotu_ARGS
#undef dotc_ARGS

/**
 * nrm2()
 */
#define nrm2_ARGS n, x, incx
LEGACY_BLAS_API_RET(float, S, nrm2,   int n, const float *x, int incx)
LEGACY_BLAS_API_RET(double, D, nrm2,  int n, const double *x, int incx)
LEGACY_BLAS_API_RET(float, Sc, nrm2,  int n, const cuComplex *x, int incx)
LEGACY_BLAS_API_RET(double, Dz, nrm2, int n, const cuDoubleComplex *x, int incx)
#undef nrm2_ARGS

/**
 * rot()
 */
#define rot_ARGS n, x, incx, y, incy, c, s
LEGACY_BLAS_API(S, rot,  int n, float* x, int incx, float* y, int incy, const float* c, const float* s)
LEGACY_BLAS_API(D, rot,  int n, double* x, int incx, double* y, int incy, const double* c, const double* s)
LEGACY_BLAS_API(C, rot,  int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const cuComplex* s)
LEGACY_BLAS_API(Cs, rot, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const float* s)
LEGACY_BLAS_API(Z, rot,  int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const cuDoubleComplex* s)
LEGACY_BLAS_API(Zd, rot, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const double* s)
#undef rot_ARGS

/**
 * rotg()
 */
#define rotg_ARGS a, b, c, s
LEGACY_BLAS_API(S, rotg, float* a, float* b, float* c, float* s)
LEGACY_BLAS_API(D, rotg, double* a, double* b, double* c, double* s)
LEGACY_BLAS_API(C, rotg, cuComplex* a, cuComplex* b, float* c, cuComplex* s)
LEGACY_BLAS_API(Z, rotg, cuDoubleComplex* a, cuDoubleComplex* b, double* c, cuDoubleComplex* s)
#undef rotg_ARGS

/**
 * rotm()
 */
#define rotm_ARGS n, x, incx, y, incy, param
LEGACY_BLAS_API(S, rotm, int n, float* x, int incx, float* y, int incy, const float* param)
LEGACY_BLAS_API(D, rotm, int n, double* x, int incx, double* y, int incy, const double* param)
#undef rotm_ARGS

/**
 * rotmg()
 */
#define rotmg_ARGS d1, d2, x1, y1, param
LEGACY_BLAS_API(S, rotmg, float* d1, float* d2, float* x1, const float* y1, float* param)
LEGACY_BLAS_API(D, rotmg, double* d1, double* d2, double* x1, const double* y1, double* param)
#undef rotmg_ARGS

/**
 * scal()
 */
#define scal_ARGS n, &alpha, x, incx
LEGACY_BLAS_API(S, scal, int n, float alpha, float *x, int incx)
LEGACY_BLAS_API(D, scal, int n, double alpha, double *x, int incx)
LEGACY_BLAS_API(C, scal, int n, cuComplex alpha, cuComplex *x, int incx)
LEGACY_BLAS_API(Z, scal, int n, cuDoubleComplex alpha, cuDoubleComplex *x, int incx)
LEGACY_BLAS_API(Cs, scal, int n, float alpha, cuComplex *x, int incx)
LEGACY_BLAS_API(Zd, scal, int n, double alpha, cuDoubleComplex *x, int incx)
#undef scal_ARGS

/**
 * swap()
 */
#define swap_ARGS n, x, incx, y, incy
LEGACY_BLAS_API(S, swap, int n, float *x, int incx, float* y, int incy)
LEGACY_BLAS_API(D, swap, int n, double *x, int incx, double* y, int incy)
LEGACY_BLAS_API(C, swap, int n, cuComplex *x, int incx, cuComplex* y, int incy)
LEGACY_BLAS_API(Z, swap, int n, cuDoubleComplex *x, int incx, cuDoubleComplex* y, int incy)
#undef swap_ARGS


/**
 * gbmv()
 */
#define gbmv_ARGS mapTransChar(trans), m, n, kl, ku, &alpha, A, lda, x, incx, &beta, y, incy
LEGACY_BLAS_API(S, gbmv,
    char trans,
    int m, int n, int kl, int ku,
    float alpha,
    const float* A, int lda,
    const float* x, int incx,
    float beta,
    float* y, int incy
)
LEGACY_BLAS_API(D, gbmv,
    char trans,
    int m, int n, int kl, int ku,
    double alpha,
    const double* A, int lda,
    const double* x, int incx,
    double beta,
    double* y, int incy
)
LEGACY_BLAS_API(C, gbmv,
    char trans,
    int m, int n, int kl, int ku,
    cuComplex alpha,
    const cuComplex* A, int lda,
    const cuComplex* x, int incx,
    cuComplex beta,
    cuComplex* y, int incy
)
LEGACY_BLAS_API(Z, gbmv,
    char trans,
    int m, int n, int kl, int ku,
    cuDoubleComplex alpha,
    const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* x, int incx,
    cuDoubleComplex beta,
    cuDoubleComplex* y, int incy
)
#undef gbmv_ARGS

/**
 * gemv()
 */
#define gemv_ARGS mapTransChar(trans), m, n, &alpha, A, lda, x, incx, &beta, y, incy
LEGACY_BLAS_API(S, gemv,
    char trans,
    int m, int n,
    float alpha,
    const float *A, int lda,
    const float *x, int incx,
    float beta,
    float *y, int incy
)
LEGACY_BLAS_API(D, gemv,
    char trans,
    int m, int n,
    double alpha,
    const double* A, int lda,
    const double* x, int incx,
    double beta,
    double* y, int incy
)
LEGACY_BLAS_API(C, gemv,
    char trans,
    int m, int n,
    cuComplex alpha,
    const cuComplex *A, int lda,
    const cuComplex *x, int incx,
    cuComplex beta,
    cuComplex *y, int incy
)
LEGACY_BLAS_API(Z, gemv,
    char trans,
    int m, int n,
    cuDoubleComplex alpha,
    const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *x, int incx,
    cuDoubleComplex beta,
    cuDoubleComplex *y, int incy
)
#undef gemv_ARGS

/**
 * ger()
 */
#define ger_ARGS m, n, &alpha, x, incx, y, incy, A, lda
#define gerc_ARGS ger_ARGS
#define geru_ARGS ger_ARGS
LEGACY_BLAS_API(S, ger,
    int m, int n,
    float alpha,
    const float* x, int incx,
    const float* y, int incy,
    float* A, int lda
)
LEGACY_BLAS_API(D, ger,
    int m, int n,
    double alpha,
    const double* x, int incx,
    const double* y, int incy,
    double* A, int lda
)
LEGACY_BLAS_API(C, geru,
    int m, int n,
    cuComplex alpha,
    const cuComplex* x, int incx,
    const cuComplex* y, int incy,
    cuComplex* A, int lda
)
LEGACY_BLAS_API(C, gerc,
    int m, int n,
    cuComplex alpha,
    const cuComplex* x, int incx,
    const cuComplex* y, int incy,
    cuComplex* A, int lda
)

LEGACY_BLAS_API(Z, geru,
    int m, int n,
    cuDoubleComplex alpha,
    const cuDoubleComplex* x, int incx,
    const cuDoubleComplex* y, int incy,
    cuDoubleComplex* A, int lda
)
LEGACY_BLAS_API(Z, gerc,
    int m, int n,
    cuDoubleComplex alpha,
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
#define sbmv_ARGS mapFillModeChar(fillMode), n, k, &alpha, A, lda, x, incx, &beta, y, incy
LEGACY_BLAS_API(S, sbmv,
    char fillMode,
    int n, int k, float alpha,
    const float *A, int lda,
    const float *x, int incx,
    float beta, float *y, int incy
)
LEGACY_BLAS_API(D, sbmv,
    char fillMode,
    int n, int k, double alpha,
    const double *A, int lda,
    const double *x, int incx,
    double beta, double *y, int incy
)
#undef sbmv_ARGS

/**
 * spmv()
 */
#define spmv_ARGS mapFillModeChar(fillMode), n, &alpha, A, x, incx, &beta, y, incy
LEGACY_BLAS_API(S, spmv,
    char fillMode,
    int n, float alpha,
    const float *A,
    const float *x, int incx,
    float beta, float *y, int incy
)
LEGACY_BLAS_API(D, spmv,
    char fillMode,
    int n, double alpha,
    const double *A,
    const double *x, int incx,
    double beta, double *y, int incy
)
#undef spmv_ARGS

/**
 * spr()
 */
#define spr_ARGS mapFillModeChar(fillMode), n, &alpha, x, incx, A
LEGACY_BLAS_API(S, spr,
    char fillMode,
    int n, float alpha,
    const float *x, int incx,
    float* A
)
LEGACY_BLAS_API(D, spr,
    char fillMode,
    int n, double alpha,
    const double* x, int incx,
    double* A
)
#undef spr_ARGS

/**
 * spr2()
 */
#define spr2_ARGS mapFillModeChar(fillMode), n, &alpha, x, incx, y, incy, A
LEGACY_BLAS_API(S, spr2,
    char fillMode,
    int n, float alpha,
    const float *x, int incx,
    const float *y, int incy,
    float* A
)
LEGACY_BLAS_API(D, spr2,
    char fillMode,
    int n, double alpha,
    const double* x, int incx,
    const double* y, int incy,
    double* A
)
#undef spr2_ARGS

/**
 * symv()
 */
#define symv_ARGS mapFillModeChar(fillMode), n, &alpha, A, lda, x, incx, &beta, y, incy
LEGACY_BLAS_API(S, symv,
    char fillMode,
    int n,
    float alpha,
    const float *A, int lda,
    const float *x, int incx,
    float beta,
    float *y, int incy
)
LEGACY_BLAS_API(D, symv,
    char fillMode,
    int n,
    double alpha,
    const double *A, int lda,
    const double *x, int incx,
    double beta,
    double *y, int incy
)
LEGACY_BLAS_API(C, symv,
    char fillMode,
    int n,
    cuComplex alpha,
    const cuComplex *A, int lda,
    const cuComplex *x, int incx,
    cuComplex beta,
    cuComplex *y, int incy
)
LEGACY_BLAS_API(Z, symv,
    char fillMode,
    int n,
    cuDoubleComplex alpha,
    const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *x, int incx,
    cuDoubleComplex beta,
    cuDoubleComplex *y, int incy
)
#undef symv_ARGS

/**
 * syr()
 */
#define syr_ARGS mapFillModeChar(fillMode), n, &alpha, x, incx, A, lda
LEGACY_BLAS_API(S, syr,
    char fillMode, int n,
    float alpha,
    const float* x, int incx,
    float* A, int lda
)
LEGACY_BLAS_API(D, syr,
    char fillMode,
    int n, double alpha,
    const double* x, int incx,
    double* A, int lda
)
LEGACY_BLAS_API(C, syr,
    char fillMode,
    int n, cuComplex alpha,
    const cuComplex* x, int incx,
    cuComplex* A, int lda
)
LEGACY_BLAS_API(Z, syr,
    char fillMode,
    int n, cuDoubleComplex alpha,
    const cuDoubleComplex* x, int incx,
    cuDoubleComplex* A, int lda
)
#undef syr_ARGS

/**
 * syr2()
 */
#define syr2_ARGS mapFillModeChar(fillMode), n, &alpha, x, incx, y, incy, A, lda
LEGACY_BLAS_API(S, syr2,
    char fillMode, int n,
    float alpha,
    const float* x, int incx,
    const float* y, int incy,
    float* A, int lda
)
LEGACY_BLAS_API(D, syr2,
    char fillMode,
    int n, double alpha,
    const double* x, int incx,
    const double* y, int incy,
    double* A, int lda
)
LEGACY_BLAS_API(C, syr2,
    char fillMode,
    int n, cuComplex alpha,
    const cuComplex* x, int incx,
    const cuComplex* y, int incy,
    cuComplex* A, int lda
)
LEGACY_BLAS_API(Z, syr2,
    char fillMode, int n,
    cuDoubleComplex alpha,
    const cuDoubleComplex* x, int incx,
    const cuDoubleComplex* y, int incy,
    cuDoubleComplex* A, int lda
)
#undef syr2_ARGS

/**
 * tbmv()
 */
#define tbmv_ARGS mapFillModeChar(fillMode), mapTransChar(trans), mapDiagChar(diag), n, k, A, lda, x, incx
LEGACY_BLAS_API(S, tbmv,
    char fillMode,
    char trans, char diag,
    int n, int k, const float* A, int lda,
    float* x, int incx
)
LEGACY_BLAS_API(D, tbmv,
    char fillMode,
    char trans, char diag,
    int n, int k, const double* A, int lda,
    double* x, int incx
)
LEGACY_BLAS_API(C, tbmv,
    char fillMode,
    char trans, char diag,
    int n, int k, const cuComplex* A, int lda,
    cuComplex* x, int incx
)
LEGACY_BLAS_API(Z, tbmv,
    char fillMode,
    char trans, char diag,
    int n, int k, const cuDoubleComplex* A, int lda,
    cuDoubleComplex* x, int incx
)
#undef tbmv_ARGS

/**
 * tbsv()
 */
#define tbsv_ARGS mapFillModeChar(fillMode), mapTransChar(trans), mapDiagChar(diag), n, k, A, lda, x, incx
LEGACY_BLAS_API(S, tbsv,
    char fillMode,
    char trans, char diag,
    int n, int k, const float* A, int lda,
    float* x, int incx
)
LEGACY_BLAS_API(D, tbsv,
    char fillMode,
    char trans, char diag,
    int n, int k, const double* A, int lda,
    double* x, int incx
)
LEGACY_BLAS_API(C, tbsv,
    char fillMode,
    char trans, char diag,
    int n, int k, const cuComplex* A, int lda,
    cuComplex* x, int incx
)
LEGACY_BLAS_API(Z, tbsv,
    char fillMode,
    char trans, char diag,
    int n, int k, const cuDoubleComplex* A, int lda,
    cuDoubleComplex* x, int incx
)
#undef tbsv_ARGS

/**
 * tpmv()
 */
#define tpmv_ARGS mapFillModeChar(fillMode), mapTransChar(trans), mapDiagChar(diag), n, A, x, incx
LEGACY_BLAS_API(S, tpmv,
    char fillMode,
    char trans, char diag,
    int n, const float* A,
    float* x, int incx
)
LEGACY_BLAS_API(D, tpmv,
    char fillMode,
    char trans, char diag,
    int n, const double* A,
    double* x, int incx
)
LEGACY_BLAS_API(C, tpmv,
    char fillMode,
    char trans, char diag,
    int n, const cuComplex* A,
    cuComplex* x, int incx
)
LEGACY_BLAS_API(Z, tpmv,
    char fillMode,
    char trans, char diag,
    int n, const cuDoubleComplex* A,
    cuDoubleComplex* x, int incx
)
#undef tpmv_ARGS

/**
 * tpsv()
 */
#define tpsv_ARGS mapFillModeChar(fillMode), mapTransChar(trans), mapDiagChar(diag), n, A, x, incx
LEGACY_BLAS_API(S, tpsv,
    char fillMode,
    char trans, char diag,
    int n, const float* A,
    float* x, int incx
)
LEGACY_BLAS_API(D, tpsv,
    char fillMode,
    char trans, char diag,
    int n, const double* A,
    double* x, int incx
)
LEGACY_BLAS_API(C, tpsv,
    char fillMode,
    char trans, char diag,
    int n, const cuComplex* A,
    cuComplex* x, int incx
)
LEGACY_BLAS_API(Z, tpsv,
    char fillMode,
    char trans, char diag,
    int n, const cuDoubleComplex* A,
    cuDoubleComplex* x, int incx
)
#undef tpsv_ARGS

/**
 * trmv()
 */
#define trmv_ARGS mapFillModeChar(fillMode), mapTransChar(trans), mapDiagChar(diag), n, A, lda, x, incx
LEGACY_BLAS_API(S, trmv,
    char fillMode,
    char trans, char diag,
    int n, const float* A, int lda,
    float* x, int incx
)
LEGACY_BLAS_API(D, trmv,
    char fillMode,
    char trans, char diag,
    int n, const double* A, int lda,
    double* x, int incx
)
LEGACY_BLAS_API(C, trmv,
    char fillMode,
    char trans, char diag,
    int n, const cuComplex* A, int lda,
    cuComplex* x, int incx
)
LEGACY_BLAS_API(Z, trmv,
    char fillMode,
    char trans, char diag,
    int n, const cuDoubleComplex* A, int lda,
    cuDoubleComplex* x, int incx
)
#undef trmv_ARGS

/**
 * trsv()
 */
#define trsv_ARGS mapFillModeChar(fillMode), mapTransChar(trans), mapDiagChar(diag), n, A, lda, x, incx
LEGACY_BLAS_API(S, trsv,
    char fillMode,
    char trans, char diag,
    int n, const float* A, int lda,
    float* x, int incx
)
LEGACY_BLAS_API(D, trsv,
    char fillMode,
    char trans, char diag,
    int n, const double* A, int lda,
    double* x, int incx
)
LEGACY_BLAS_API(C, trsv,
    char fillMode,
    char trans, char diag,
    int n, const cuComplex* A, int lda,
    cuComplex* x, int incx
)
LEGACY_BLAS_API(Z, trsv,
    char fillMode,
    char trans, char diag,
    int n, const cuDoubleComplex* A, int lda,
    cuDoubleComplex* x, int incx
)
#undef trsv_ARGS

/**
 * hemv()
 */
#define hemv_ARGS mapFillModeChar(fillMode), n, &alpha, A, lda, x, incx, &beta, y, incy
LEGACY_BLAS_API(C, hemv,
    char fillMode,
    int n,
    cuComplex alpha,
    const cuComplex *A, int lda,
    const cuComplex *x, int incx,
    cuComplex beta,
    cuComplex *y, int incy
)
LEGACY_BLAS_API(Z, hemv,
    char fillMode,
    int n,
    cuDoubleComplex alpha,
    const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *x, int incx,
    cuDoubleComplex beta,
    cuDoubleComplex *y, int incy
)
#undef hemv_ARGS

/**
 * hbmv()
 */
#define hbmv_ARGS mapFillModeChar(fillMode), n, k, &alpha, A, lda, x, incx, &beta, y, incy
LEGACY_BLAS_API(C, hbmv,
    char fillMode,
    int n, int k, cuComplex alpha,
    const cuComplex* A, int lda,
    const cuComplex* x, int incx,
    cuComplex beta,
    cuComplex* y, int incy
)
LEGACY_BLAS_API(Z, hbmv,
    char fillMode,
    int n, int k, cuDoubleComplex alpha,
    const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* x, int incx,
    cuDoubleComplex beta,
    cuDoubleComplex* y, int incy
)
#undef hbmv_ARGS

/**
 * hpmv()
 */
#define hpmv_ARGS mapFillModeChar(fillMode), n, &alpha, AP, x, incx, &beta, y, incy
LEGACY_BLAS_API(C, hpmv,
    char fillMode,
    int n, cuComplex alpha,
    const cuComplex* AP,
    const cuComplex* x, int incx,
    cuComplex beta,
    cuComplex* y, int incy
)
LEGACY_BLAS_API(Z, hpmv,
    char fillMode,
    int n, cuDoubleComplex alpha,
    const cuDoubleComplex* AP,
    const cuDoubleComplex* x, int incx,
    cuDoubleComplex beta,
    cuDoubleComplex* y, int incy
)
#undef hpmv_ARGS

/**
 * her()
 */
#define her_ARGS mapFillModeChar(fillMode), n, &alpha, x, incx, A, lda
LEGACY_BLAS_API(C, her,
    char fillMode,
    int n, float alpha,
    const cuComplex* x, int incx,
    cuComplex* A, int lda
)
LEGACY_BLAS_API(Z, her,
    char fillMode,
    int n, double alpha,
    const cuDoubleComplex* x, int incx,
    cuDoubleComplex* A, int lda
)
#undef her_ARGS

/**
 * her2()
 */
#define her2_ARGS mapFillModeChar(fillMode), n, &alpha, x, incx, y, incy, A, lda
LEGACY_BLAS_API(C, her2,
    char fillMode,
    int n, cuComplex alpha,
    const cuComplex* x, int incx,
    const cuComplex* y, int incy,
    cuComplex* A, int lda
)
LEGACY_BLAS_API(Z, her2,
    char fillMode,
    int n, cuDoubleComplex alpha,
    const cuDoubleComplex* x, int incx,
    const cuDoubleComplex* y, int incy,
    cuDoubleComplex* A, int lda
)
#undef her2_ARGS

/**
 * hpr()
 */
#define hpr_ARGS mapFillModeChar(fillMode), n, &alpha, x, incx, A
LEGACY_BLAS_API(C, hpr,
    char fillMode,
    int n, float alpha,
    const cuComplex* x, int incx,
    cuComplex* A
)
LEGACY_BLAS_API(Z, hpr,
    char fillMode,
    int n, double alpha,
    const cuDoubleComplex* x, int incx,
    cuDoubleComplex* A
)
#undef hpr_ARGS

/**
 * hpr2()
 */
#define hpr2_ARGS mapFillModeChar(fillMode), n, &alpha, x, incx, y, incy, A
LEGACY_BLAS_API(C, hpr2,
    char fillMode,
    int n, cuComplex alpha,
    const cuComplex* x, int incx,
    const cuComplex* y, int incy,
    cuComplex* A
)
LEGACY_BLAS_API(Z, hpr2,
    char fillMode,
    int n, cuDoubleComplex alpha,
    const cuDoubleComplex* x, int incx,
    const cuDoubleComplex* y, int incy,
    cuDoubleComplex* A
)
#undef hpr2_ARGS


/**
 * gemm()
 */
#define gemm_ARGS mapTransChar(transa), mapTransChar(transb), m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc
LEGACY_BLAS_API(S, gemm,
    char transa, char transb,
    int m, int n, int k,
    float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float beta,
    float* C, int ldc
)
LEGACY_BLAS_API(D, gemm,
    char transa, char transb,
    int m, int n, int k,
    double alpha,
    const double* A, int lda,
    const double* B, int ldb,
    double beta,
    double* C, int ldc
)
LEGACY_BLAS_API(C, gemm,
    char transa, char transb,
    int m, int n, int k,
    cuComplex alpha,
    const cuComplex *A, int lda,
    const cuComplex *B, int ldb,
    cuComplex beta,
    cuComplex *C, int ldc
)
LEGACY_BLAS_API(Z, gemm,
    char transa, char transb,
    int m, int n, int k,
    cuDoubleComplex alpha,
    const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *B, int ldb,
    cuDoubleComplex beta,
    cuDoubleComplex *C, int ldc
)
LEGACY_BLAS_API(H, gemm,
    char transa, char transb,
    int m, int n, int k,
    __half alpha,
    const __half *A, int lda,
    const __half *B, int ldb,
    __half beta,
    __half *C, int ldc
)
#undef gemm_ARGS

/**
 * symm()
 */

#define symm_ARGS mapSideMode(sideMode), mapFillModeChar(fillMode), m, n, &alpha, A, lda, B, ldb, &beta, C, ldc
LEGACY_BLAS_API(S, symm,
    char sideMode, char fillMode,
    int m, int n,
    float alpha,
    const float *A, int lda,
    const float *B, int ldb,
    float beta,
         float *C, int ldc
)
LEGACY_BLAS_API(D, symm,
    char sideMode, char fillMode,
    int m, int n,
    double alpha,
    const double *A, int lda,
    const double *B, int ldb,
    double beta,
         double *C, int ldc
)
LEGACY_BLAS_API(C, symm,
    char sideMode, char fillMode,
    int m, int n,
    cuComplex alpha,
    const cuComplex *A, int lda,
    const cuComplex *B, int ldb,
    cuComplex beta,
    cuComplex *C, int ldc
)
LEGACY_BLAS_API(Z, symm,
    char sideMode, char fillMode,
    int m, int n,
    cuDoubleComplex alpha,
    const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *B, int ldb,
    cuDoubleComplex beta,
    cuDoubleComplex *C, int ldc
)
#undef symm_ARGS

/**
 * syrk()
 */
#define syrk_ARGS mapFillModeChar(fillMode), mapTransChar(trans), n, k, &alpha, A, lda, &beta, C, ldc
LEGACY_BLAS_API(S, syrk,
    char fillMode, char trans,
    int n, int k,
    float alpha,
    const float *A, int lda,
    float beta,
    float *C, int ldc
)
LEGACY_BLAS_API(D, syrk,
    char fillMode, char trans,
    int n, int k,
    double alpha,
    const double *A, int lda,
    double beta,
    double *C, int ldc
)
LEGACY_BLAS_API(C, syrk,
    char fillMode, char trans,
    int n, int k,
    cuComplex alpha,
    const cuComplex *A, int lda,
    cuComplex beta,
    cuComplex *C, int ldc
)
LEGACY_BLAS_API(Z, syrk,
    char fillMode, char trans,
    int n, int k,
    cuDoubleComplex alpha,
    const cuDoubleComplex *A, int lda,
    cuDoubleComplex beta,
    cuDoubleComplex *C, int ldc
)
#undef syrk_ARGS

/**
 * syr2k()
 */
#define syr2k_ARGS mapFillModeChar(fillMode), mapTransChar(trans), n, k, &alpha, A, lda, B, ldb, &beta, C, ldc
LEGACY_BLAS_API(S, syr2k,
    char fillMode, char trans,
    int n, int k,
    float alpha,
    const float *A, int lda,
    const float *B, int ldb,
    float beta,
    float *C, int ldc
)
LEGACY_BLAS_API(D, syr2k,
    char fillMode, char trans,
    int n, int k,
    double alpha,
    const double *A, int lda,
    const double*B, int ldb,
    double beta,
    double *C, int ldc
)
LEGACY_BLAS_API(C, syr2k,
    char fillMode, char trans,
    int n, int k,
    cuComplex alpha,
    const cuComplex *A, int lda,
    const cuComplex *B, int ldb,
    cuComplex beta,
    cuComplex *C, int ldc
)
LEGACY_BLAS_API(Z, syr2k,
    char fillMode, char trans,
    int n, int k,
    cuDoubleComplex alpha,
    const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *B, int ldb,
    cuDoubleComplex beta,
    cuDoubleComplex *C, int ldc
)
#undef syr2k_ARGS

#if 0
/**
 * syrkx()
 */
#define syrkx_ARGS mapFillModeChar(fillMode), mapTransChar(trans), n, k, &alpha, A, lda, &beta, C, ldc
LEGACY_BLAS_API(S, syrkx,
    char fillMode, char trans,
    int n, int k,
    float alpha,
    const float *A, int lda,
    float beta,
    float *C, int ldc
)
LEGACY_BLAS_API(D, syrkx,
    char fillMode, char trans,
    int n, int k,
    double alpha,
    const double *A, int lda,
    double beta,
    double *C, int ldc
)
LEGACY_BLAS_API(C, syrkx,
    char fillMode, char trans,
    int n, int k,
    cuComplex alpha,
    const cuComplex *A, int lda,
    cuComplex beta,
    cuComplex *C, int ldc
)
LEGACY_BLAS_API(Z, syrkx,
    char fillMode, char trans,
    int n, int k,
    cuDoubleComplex alpha,
    const cuDoubleComplex *A, int lda,
    cuDoubleComplex beta,
    cuDoubleComplex *C, int ldc
)
#undef syrkx_ARGS
#endif

#if 0
/**
 * trmm()
 */
#define trmm_ARGS mapSideMode(side), mapFillModeChar(fillMode), mapTransChar(transA), mapDiagChar(diag), m, n, &alpha, A, lda, B, ldb, C, ldc
LEGACY_BLAS_API(S, trmm,
    char side, char fillMode,
    char transA, char diag,
    int m, int n,
    float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc
)
LEGACY_BLAS_API(D, trmm,
    char side, char fillMode,
    char transA, char diag,
    int m, int n,
    double alpha,
    const double* A, int lda,
    const double* B, int ldb,
    double* C, int ldc
)
LEGACY_BLAS_API(C, trmm,
    char side, char fillMode,
    char transA, char diag,
    int m, int n,
    cuComplex alpha,
    const cuComplex* A, int lda,
    const cuComplex* B, int ldb,
    cuComplex* C, int ldc
)
LEGACY_BLAS_API(Z, trmm,
    char side, char fillMode,
    char transA, char diag,
    int m, int n,
    cuDoubleComplex alpha,
    const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* B, int ldb,
    cuDoubleComplex* C, int ldc
)
#undef trmm_ARGS
#endif

/**
 * trsm()
 */
#define trsm_ARGS mapSideMode(side), mapFillModeChar(fillMode), mapTransChar(transA), mapDiagChar(diag), m, n, &alpha, A, lda, B, ldb
LEGACY_BLAS_API(S, trsm,
    char side, char fillMode,
    char transA, char diag,
    int m, int n,
    float alpha,
    float* A, int lda,
    float* B, int ldb
)

LEGACY_BLAS_API(D, trsm,
    char side, char fillMode,
    char transA, char diag,
    int m, int n,
    double alpha,
    double* A, int lda,
    double* B, int ldb
)
LEGACY_BLAS_API(C, trsm,
    char side, char fillMode,
    char transA, char diag,
    int m, int n,
    cuComplex alpha,
    cuComplex* A, int lda,
    cuComplex* B, int ldb
)
LEGACY_BLAS_API(Z, trsm,
    char side, char fillMode,
    char transA, char diag,
    int m, int n,
    cuDoubleComplex alpha,
    cuDoubleComplex* A, int lda,
    cuDoubleComplex* B, int ldb
)
#undef trsm_ARGS

/**
 * hemm()
 */
#define hemm_ARGS mapSideMode(side), mapFillModeChar(fillMode), m, n, &alpha, A, lda, B, ldb, &beta, C, ldc
LEGACY_BLAS_API(C, hemm,
    char side, char fillMode,
    int m, int n,
    cuComplex alpha,
    const cuComplex* A, int lda,
    const cuComplex* B, int ldb,
    cuComplex beta,
    cuComplex* C, int ldc
)
LEGACY_BLAS_API(Z, hemm,
    char side, char fillMode,
    int m, int n,
    cuDoubleComplex alpha,
    const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* B, int ldb,
    cuDoubleComplex beta,
    cuDoubleComplex* C, int ldc
)
#undef hemm_ARGS

/**
 * herk()
 */
#define herk_ARGS mapFillModeChar(fillMode), mapTransChar(trans), n, k, &alpha, A, lda, &beta, C, ldc
LEGACY_BLAS_API(C, herk,
    char fillMode, char trans,
    int n, int k,
    float alpha,
    const cuComplex* A, int lda,
    float beta,
    cuComplex* C, int ldc
)
LEGACY_BLAS_API(Z, herk,
    char fillMode, char trans,
    int n, int k,
    double alpha,
    const cuDoubleComplex* A, int lda,
    double beta,
    cuDoubleComplex* C, int ldc
)
#undef herk_ARGS

/**
 * her2k()
 */
#define her2k_ARGS mapFillModeChar(fillMode), mapTransChar(trans), n, k, &alpha, A, lda, B, ldb, &beta, C, ldc
LEGACY_BLAS_API(C, her2k,
    char fillMode, char trans,
    int n, int k,
    cuComplex alpha,
    const cuComplex* A, int lda,
    const cuComplex* B, int ldb,
    float beta,
    cuComplex* C, int ldc
)
LEGACY_BLAS_API(Z, her2k,
    char fillMode, char trans,
    int n, int k,
    cuDoubleComplex alpha,
    const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* B, int ldb,
    double beta,
    cuDoubleComplex* C, int ldc
)
#undef her2k_ARGS
