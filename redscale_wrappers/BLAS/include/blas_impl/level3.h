/*///////////////*/
/*/// Level 3 ///*/
/*///////////////*/

/**
 * gemm()
 */
#define gemm_ARGS transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
BLAS_API(S, s, gemm, Gemm,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta,
    float* C, int ldc
)
BLAS_API(D, d, gemm, Gemm,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double* beta,
    double* C, int ldc
)
BLAS_API(C, c, gemm, Gemm,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const cuComplex *alpha,
    const cuComplex *A, int lda,
    const cuComplex *B, int ldb,
    const cuComplex *beta,
    cuComplex *C, int ldc
)
BLAS_API(Z, z, gemm, Gemm,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *B, int ldb,
    const cuDoubleComplex *beta,
    cuDoubleComplex *C, int ldc
)
BLAS_API(H, h, gemm, Gemm,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const __half *alpha,
    const __half *A, int lda,
    const __half *B, int ldb,
    const __half *beta,
    __half *C, int ldc
)
#undef gemm_ARGS

/**
 * gemmBatched()
 */
#define SgemmBatched_ARGS transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount
#define DgemmBatched_ARGS SgemmBatched_ARGS
#define CgemmBatched_ARGS SgemmBatched_ARGS
#define ZgemmBatched_ARGS SgemmBatched_ARGS
DIRECT_BLAS_API_N(SgemmBatched, _sgemm_batched,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* const Aarray[], int lda,
    const float* const Barray[], int ldb,
    const float* beta,
    float* const Carray[], int ldc,
    int batchCount
)
DIRECT_BLAS_API_N(DgemmBatched, _dgemm_batched,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* const Aarray[], int lda,
    const double* const Barray[], int ldb,
    const double* beta,
    double* const Carray[], int ldc,
    int batchCount
)
//DIRECT_BLAS_API_N(CgemmBatched, _cgemm_batched,
//    cublasOperation_t transa, cublasOperation_t transb,
//    int m, int n, int k,
//    const cuComplex *alpha,
//    const cuComplex* const Aarray[], int lda,
//    const cuComplex* const Barray[], int ldb,
//    const cuComplex *beta,
//    cuComplex* const Carray[], int ldc,
//    int batchCount
//)
//DIRECT_BLAS_API_N(ZgemmBatched, _zgemm_batched,
//    cublasOperation_t transa, cublasOperation_t transb,
//    int m, int n, int k,
//    const cuDoubleComplex *alpha,
//    const cuDoubleComplex* const Aarray[], int lda,
//    const cuDoubleComplex* const Barray[], int ldb,
//    const cuDoubleComplex *beta,
//    cuDoubleComplex* const Carray[], int ldc,
//    int batchCount
//)
#undef cublasSgemmBatched_ARGS
#undef cublasDgemmBatched_ARGS
#undef cublasCgemmBatched_ARGS
#undef cublasZgemmBatched_ARGS

#if 0
/**
 * gemm3m()
 */
#define gemm3m_ARGS transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
BLAS_API(C, c, gemm3m, Gemm3m,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const cuComplex *alpha,
    const cuComplex *A, int lda,
    const cuComplex *B, int ldb,
    const cuComplex *beta,
    cuComplex *C, int ldc
)
BLAS_API(Z, z, gemm3m, Gemm3m,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *B, int ldb,
    const cuDoubleComplex *beta,
    cuDoubleComplex *C, int ldc
)
#undef gemm3m_ARGS
#endif

/**
 * symm()
 */

#define symm_ARGS sideMode, fillMode, m, n, alpha, A, lda, B, ldb, beta, C, ldc
BLAS_API(S, s, symm, Symm,
    cublasSideMode_t sideMode, cublasFillMode_t fillMode,
    int m, int n,
    const float *alpha,
    const float *A, int lda,
    const float *B, int ldb,
    const float *beta,
         float *C, int ldc
)
BLAS_API(D, d, symm, Symm,
    cublasSideMode_t sideMode, cublasFillMode_t fillMode,
    int m, int n,
    const double *alpha,
    const double *A, int lda,
    const double *B, int ldb,
    const double *beta,
         double *C, int ldc
)
BLAS_API(C, c, symm, Symm,
    cublasSideMode_t sideMode, cublasFillMode_t fillMode,
    int m, int n,
    const cuComplex *alpha,
    const cuComplex *A, int lda,
    const cuComplex *B, int ldb,
    const cuComplex *beta,
    cuComplex *C, int ldc
)
BLAS_API(Z, z, symm, Symm,
    cublasSideMode_t sideMode, cublasFillMode_t fillMode,
    int m, int n,
    const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *B, int ldb,
    const cuDoubleComplex *beta,
    cuDoubleComplex *C, int ldc
)
#undef symm_ARGS

/**
 * syrk()
 */
#define syrk_ARGS fillMode, trans, n, k, alpha, A, lda, beta, C, ldc
BLAS_API(S, s, syrk, Syrk,
    cublasFillMode_t fillMode, cublasOperation_t trans,
    int n, int k,
    const float *alpha,
    const float *A, int lda,
    const float *beta,
    float *C, int ldc
)
BLAS_API(D, d, syrk, Syrk,
    cublasFillMode_t fillMode, cublasOperation_t trans,
    int n, int k,
    const double *alpha,
    const double *A, int lda,
    const double *beta,
    double *C, int ldc
)
BLAS_API(C, c, syrk, Syrk,
    cublasFillMode_t fillMode, cublasOperation_t trans,
    int n, int k,
    const cuComplex *alpha,
    const cuComplex *A, int lda,
    const cuComplex *beta,
    cuComplex *C, int ldc
)
BLAS_API(Z, z, syrk, Syrk,
    cublasFillMode_t fillMode, cublasOperation_t trans,
    int n, int k,
    const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *beta,
    cuDoubleComplex *C, int ldc
)
#undef sryk_ARGS

/**
 * syr2k()
 */
#define syr2k_ARGS fillMode, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc
BLAS_API(S, s, syr2k, Syr2k,
    cublasFillMode_t fillMode, cublasOperation_t trans,
    int n, int k,
    const float *alpha,
    const float *A, int lda,
    const float *B, int ldb,
    const float *beta,
    float *C, int ldc
)
BLAS_API(D, d, syr2k, Syr2k,
    cublasFillMode_t fillMode, cublasOperation_t trans,
    int n, int k,
    const double *alpha,
    const double *A, int lda,
    const double*B, int ldb,
    const double *beta,
    double *C, int ldc
)
BLAS_API(C, c, syr2k, Syr2k,
    cublasFillMode_t fillMode, cublasOperation_t trans,
    int n, int k,
    const cuComplex *alpha,
    const cuComplex *A, int lda,
    const cuComplex *B, int ldb,
    const cuComplex *beta,
    cuComplex *C, int ldc
)
BLAS_API(Z, z, syr2k, Syr2k,
    cublasFillMode_t fillMode, cublasOperation_t trans,
    int n, int k,
    const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *B, int ldb,
    const cuDoubleComplex *beta,
    cuDoubleComplex *C, int ldc
)
#undef sry2k_ARGS

#if 0
/**
 * syrkx()
 */
#define syrkx_ARGS fillMode, trans, n, k, alpha, A, lda, beta, C, ldc
BLAS_API(S, s, syrkx, Syrkx,
    cublasFillMode_t fillMode, cublasOperation_t trans,
    int n, int k,
    const float *alpha,
    const float *A, int lda,
    const float *beta,
    float *C, int ldc
)
BLAS_API(D, d, syrkx, Syrkx,
    cublasFillMode_t fillMode, cublasOperation_t trans,
    int n, int k,
    const double *alpha,
    const double *A, int lda,
    const double *beta,
    double *C, int ldc
)
BLAS_API(C, c, syrkx, Syrkx,
    cublasFillMode_t fillMode, cublasOperation_t trans,
    int n, int k,
    const cuComplex *alpha,
    const cuComplex *A, int lda,
    const cuComplex *beta,
    cuComplex *C, int ldc
)
BLAS_API(Z, z, syrkx, Syrkx,
    cublasFillMode_t fillMode, cublasOperation_t trans,
    int n, int k,
    const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *beta,
    cuDoubleComplex *C, int ldc
)
#undef syrkx_ARGS
#endif

#if 0
/**
 * trmm()
 */
#define trmm_ARGS side, fillMode, transA, diag, m, n, alpha, A, lda, B, ldb, C, ldc
BLAS_API(S, s, trmm, Trmm,
    cublasSideMode_t side, cublasFillMode_t fillMode,
    cublasOperation_t transA, cublasDiagType_t diag,
    int m, int n,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc
)
BLAS_API(D, d, trmm, Trmm,
    cublasSideMode_t side, cublasFillMode_t fillMode,
    cublasOperation_t transA, cublasDiagType_t diag,
    int m, int n,
    const double* alpha,
    const double* A, int lda,
    const double* B, int ldb,
    double* C, int ldc
)
BLAS_API(C, c, trmm, Trmm,
    cublasSideMode_t side, cublasFillMode_t fillMode,
    cublasOperation_t transA, cublasDiagType_t diag,
    int m, int n,
    const cuComplex* alpha,
    const cuComplex* A, int lda,
    const cuComplex* B, int ldb,
    cuComplex* C, int ldc
)
BLAS_API(Z, z, trmm, Trmm,
    cublasSideMode_t side, cublasFillMode_t fillMode,
    cublasOperation_t transA, cublasDiagType_t diag,
    int m, int n,
    const cuDoubleComplex* alpha,
    const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* B, int ldb,
    cuDoubleComplex* C, int ldc
)
#undef trmm_ARGS
#endif

/**
 * trsm()
 */
#define trsm_ARGS side, fillMode, transA, diag, m, n, alpha, A, lda, B, ldb
BLAS_API(S, s, trsm, Trsm,
    cublasSideMode_t side, cublasFillMode_t fillMode,
    cublasOperation_t transA, cublasDiagType_t diag,
    int m, int n,
    const float* alpha,
    float* A, int lda,
    float* B, int ldb
)

BLAS_API(D, d, trsm, Trsm,
    cublasSideMode_t side, cublasFillMode_t fillMode,
    cublasOperation_t transA, cublasDiagType_t diag,
    int m, int n,
    const double* alpha,
    double* A, int lda,
    double* B, int ldb
)
BLAS_API(C, c, trsm, Trsm,
    cublasSideMode_t side, cublasFillMode_t fillMode,
    cublasOperation_t transA, cublasDiagType_t diag,
    int m, int n,
    const cuComplex* alpha,
    cuComplex* A, int lda,
    cuComplex* B, int ldb
)
BLAS_API(Z, z, trsm, Trsm,
    cublasSideMode_t side, cublasFillMode_t fillMode,
    cublasOperation_t transA, cublasDiagType_t diag,
    int m, int n,
    const cuDoubleComplex* alpha,
    cuDoubleComplex* A, int lda,
    cuDoubleComplex* B, int ldb
)
#undef trsm_ARGS

/**
 * hemm()
 */
#define hemm_ARGS side, fillMode, m, n, alpha, A, lda, B, ldb, beta, C, ldc
BLAS_API(C, c, hemm, Hemm,
    cublasSideMode_t side, cublasFillMode_t fillMode,
    int m, int n,
    const cuComplex* alpha,
    const cuComplex* A, int lda,
    const cuComplex* B, int ldb,
    const cuComplex* beta,
    cuComplex* C, int ldc
)
BLAS_API(Z, z, hemm, Hemm,
    cublasSideMode_t side, cublasFillMode_t fillMode,
    int m, int n,
    const cuDoubleComplex* alpha,
    const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* B, int ldb,
    const cuDoubleComplex* beta,
    cuDoubleComplex* C, int ldc
)
#undef hemm_ARGS

/**
 * herk()
 */
#define herk_ARGS fillMode, trans, n, k, alpha, A, lda, beta, C, ldc
BLAS_API(C, c, herk, Herk,
    cublasFillMode_t fillMode, cublasOperation_t trans,
    int n, int k,
    const float* alpha,
    const cuComplex* A, int lda,
    const float* beta,
    cuComplex* C, int ldc
)
BLAS_API(Z, z, herk, Herk,
    cublasFillMode_t fillMode, cublasOperation_t trans,
    int n, int k,
    const double* alpha,
    const cuDoubleComplex* A, int lda,
    const double* beta,
    cuDoubleComplex* C, int ldc
)
#undef herk_ARGS

/**
 * her2k()
 */
#define her2k_ARGS fillMode, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc
BLAS_API(C, c, her2k, Her2k,
    cublasFillMode_t fillMode, cublasOperation_t trans,
    int n, int k,
    const cuComplex* alpha,
    const cuComplex* A, int lda,
    const cuComplex* B, int ldb,
    const float* beta,
    cuComplex* C, int ldc
)
BLAS_API(Z, z, her2k, Her2k,
    cublasFillMode_t fillMode, cublasOperation_t trans,
    int n, int k,
    const cuDoubleComplex* alpha,
    const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* B, int ldb,
    const double* beta,
    cuDoubleComplex* C, int ldc
)
#undef her2k_ARGS

/**
 * herkx()
 */
#define herkx_ARGS fillMode, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc
BLAS_API(C, c, herkx, Herkx,
    cublasFillMode_t fillMode, cublasOperation_t trans,
    int n, int k,
    const cuComplex* alpha,
    const cuComplex* A, int lda,
    const cuComplex* B, int ldb,
    const float* beta,
    cuComplex* C, int ldc
)
BLAS_API(Z, z, herkx, Herkx,
    cublasFillMode_t fillMode, cublasOperation_t trans,
    int n, int k,
    const cuDoubleComplex* alpha,
    const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* B, int ldb,
    const double* beta,
    cuDoubleComplex* C, int ldc
)
#undef herkx_ARGS

/*//////////////////////////*/
/*/// Lapack Extensions  ///*/
/*//////////////////////////*/

/**
 * trtri()
 */
#define trtri_ARGS fillMode, diag, n, A, lda, invA, ldinvA
BLAS_API(S, s, trtri, Trtri,
    cublasFillMode_t fillMode, cublasDiagType_t diag,
    int n,
    float* A, int lda,
    float* invA, int ldinvA
)
BLAS_API(D, d, trtri, Trtri,
    cublasFillMode_t fillMode, cublasDiagType_t diag,
    int n,
    double* A, int lda,
    double* invA, int ldinvA
)
#undef trtri_ARGS

#if 0
/**
 * trtriBatched()
 */
#define trtriBatched_ARGS fillMode, diag, n, A, lda, bsa, invA, ldinvA, bsinvA, batch_count
BLAS_API(S, s, trtriBatched, TrtriBatched,
    cublasFillMode_t fillMode,
    cublasDiagType_t diag,
    int n,
    float* A, int lda, int bsa,
    float* invA, int ldinvA, int bsinvA,
    int batch_count
)
BLAS_API(D, d, trtriBatched, TrtriBatched,
    cublasFillMode_t fillMode,
    cublasDiagType_t diag,
    int n,
    double* A, int lda, int bsa,
    double* invA, int ldinvA, int bsinvA,
    int batch_count
)
#undef trtriBatched_ARGS
#endif


/*/////////////////////////*/
/*/// Other Extensions  ///*/
/*/////////////////////////*/
/* These are mostly things found in cuBLAS or Intel's Math Kernel Library. */

#if 0
/**
 * tpttr()
 */
#define tpttr_ARGS fillMode, n, AP, A, lda
BLAS_API(S, s, tpttr, Tpttr,
    cublasFillMode_t fillMode,
    int n,
    const float *AP,
    float *A, int lda
)
BLAS_API(D, d, tpttr, Tpttr,
    cublasFillMode_t fillMode,
    int n,
    const double *AP,
    double *A, int lda
)
BLAS_API(C, c, tpttr, Tpttr,
    cublasFillMode_t fillMode,
    int n,
    const cuComplex *AP,
    cuComplex *A, int lda
)
BLAS_API(Z, z, tpttr, Tpttr,
    cublasFillMode_t fillMode,
    int n,
    const cuDoubleComplex *AP,
    cuDoubleComplex *A, int lda
)
#undef tpttr_ARGS
#endif

#if 0
/**
 * trttp()
 */
#define trttp_ARGS fillMode, n, A, lda, AP
BLAS_API(S, s, trttp, Trttp,
    cublasFillMode_t fillMode,
    int n,
    const float *A, int lda,
    float *AP
)
BLAS_API(D, d, trttp, Trttp,
    cublasFillMode_t fillMode,
    int n,
    const double *A, int lda,
    double *AP
)
BLAS_API(C, c, trttp, Trttp,
    cublasFillMode_t fillMode,
    int n,
    const cuComplex *A, int lda,
    cuComplex *AP
)
BLAS_API(Z, z, trttp, Trttp,
    cublasFillMode_t fillMode,
    int n,
    const cuDoubleComplex *A, int lda,
    cuDoubleComplex *AP
)
#undef trttp_ARGS
#endif

/**
 * geam()
 */
#define geam_ARGS transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc
BLAS_API(S, s, geam, Geam,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n,
    const float* alpha,
    const float* A, int lda,
    const float* beta,
    const float* B, int ldb,
    float* C, int ldc
)
BLAS_API(D, d, geam, Geam,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n,
    const double* alpha,
    const double* A, int lda,
    const double* beta,
    const double* B, int ldb,
    double* C, int ldc
)
BLAS_API(C, c, geam, Geam,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n,
    const cuComplex* alpha,
    const cuComplex* A, int lda,
    const cuComplex* beta,
    const cuComplex* B, int ldb,
    cuComplex* C, int ldc
)
BLAS_API(Z, z, geam, Geam,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n,
    const cuDoubleComplex* alpha,
    const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* beta,
    const cuDoubleComplex* B, int ldb,
    cuDoubleComplex* C, int ldc
)
#undef geam_ARGS

/**
 * dgmm()
 */
#define dgmm_ARGS sideMode, m, n, A, lda, x, incx, C, ldc
BLAS_API(S, s, dgmm, Dgmm,
    cublasSideMode_t sideMode,
    int m, int n,
    const float* A, int lda,
    const float* x, int incx,
    float* C, int ldc
)
BLAS_API(D, d, dgmm, Dgmm,
    cublasSideMode_t sideMode,
    int m, int n,
    const double* A, int lda,
    const double* x, int incx,
    double* C, int ldc
)
BLAS_API(C, c, dgmm, Dgmm,
    cublasSideMode_t sideMode,
    int m, int n,
    const cuComplex* A, int lda,
    const cuComplex* x, int incx,
    cuComplex* C, int ldc
)
BLAS_API(Z, z, dgmm, Dgmm,
    cublasSideMode_t sideMode,
    int m, int n,
    const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* x, int incx,
    cuDoubleComplex* C, int ldc
)
#undef dgmm_ARGS

#if 0
/**
 * Nrm2Ex()
 */
#define Nrm2Ex_ARGS n, x, xType, incx, result, resultType, executionType
DIRECT_BLAS_API_N(Nrm2Ex, nrm2_ex,
    int n,
    const void* x,
    cudaDataType xType,
    int incx,
    void* result,
    cudaDataType resultType,
    cudaDataType executionType
)
#undef Nrm2Ex_ARGS
#endif

#if 0
/**
 * AxpyEx()
 */
#define AxpyEx_ARGS n, alpha, alphaType, x, xType, incx, y, yType, incy, executionType
DIRECT_BLAS_API_N(AxpyEx, axpy_ex,
    int n,
    const void* alpha,
    cudaDataType alphaType,
    const void* x,
    cudaDataType xType,
    int incx,
    void* y,
    cudaDataType yType,
    int incy,
    cudaDataType executiontype
)
#undef AxpyEx_ARGS
#endif

#if 0
/**
 * DotEx()
 */
#define DotEx_ARGS n, x, xType, incx, y, yType, incy, result, resultType, executionType
#define DotcEx_ARGS DotEx_ARGS
DIRECT_BLAS_API_N(DotEx, dot_ex,
    int n,
    const void* x,
    cudaDataType xType,
    int incx,
    const void* y,
    cudaDataType yType,
    int incy,
    void* result,
    cudaDataType resultType,
    cudaDataType executionType
)
DIRECT_BLAS_API_N(DotcEx, dotc_ex,
    int n,
    const void* x,
    cudaDataType xType,
    int incx,
    const void* y,
    cudaDataType yType,
    int incy,
    void* result,
    cudaDataType resultType,
    cudaDataType executionType
)
#undef DotEx_ARGS
#undef DotcEx_ARGS
#endif

