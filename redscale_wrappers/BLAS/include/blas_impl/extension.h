#include <cuda_fp16.h>

#define GemmEx_ARGS cuToRoc(transa), cuToRoc(transb), m, n, k, alpha, A, cuToRoc(Atype), lda, B, cuToRoc(Btype), ldb, beta, C, cuToRoc(Ctype), ldc, C, cuToRoc(Ctype), ldc, to_rocblas_datatype(computeType), to_rocblas_gemm_algo(algo), -1, 0
DIRECT_BLAS_API_N(GemmEx, _gemm_ex,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, cudaDataType Atype, int lda,
    const void* B, cudaDataType Btype, int ldb,
    const void* beta, 
    void* C, cudaDataType Ctype, int ldc,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo
);
#undef GemmEx_ARGS

#define GemmBatchedEx_ARGS cuToRoc(transa), cuToRoc(transb), m, n, k, alpha, (const void*) Aarray, cuToRoc(Atype), lda, (const void*) Barray, cuToRoc(Btype), ldb, beta, (void*) Carray, cuToRoc(Ctype), ldc, (void*) Carray, cuToRoc(Ctype), ldc, batchCount, to_rocblas_datatype(computeType), to_rocblas_gemm_algo(algo), -1, 0
DIRECT_BLAS_API_N(GemmBatchedEx, _gemm_batched_ex,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* const Aarray[], cudaDataType Atype, int lda,
    const void* const Barray[], cudaDataType Btype, int ldb,
    const void* beta,
    void* const Carray[], cudaDataType Ctype, int ldc,
    int batchCount,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo
);
#undef GemmStridedBatchedEx_ARGS

#define GemmStridedBatchedEx_ARGS cuToRoc(transa), cuToRoc(transb), m, n, k, alpha, A, cuToRoc(Atype), lda, strideA, B, cuToRoc(Btype), ldb, strideB, beta, C, cuToRoc(Ctype), ldc, strideC, C, cuToRoc(Ctype), ldc, strideC, batchCount, to_rocblas_datatype(computeType), to_rocblas_gemm_algo(algo), -1, 0
DIRECT_BLAS_API_N(GemmStridedBatchedEx, _gemm_strided_batched_ex,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, cudaDataType Atype, int lda, long long int strideA,
    const void* B, cudaDataType Btype, int ldb, long long int strideB,
    const void* beta,
    void* C, cudaDataType Ctype, int ldc, long long int strideC,
    int batchCount,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo
);
#undef GemmStridedBatchedEx_ARGS

#ifdef __cplusplus
GPUBLAS_EXPORT cublasStatus_t cublasGemmEx(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A,
    cudaDataType Atype,
    int lda,
    const void *B,
    cudaDataType Btype,
    int ldb,
    const void *beta,
    void *C,
    cudaDataType Ctype,
    int ldc,
    cudaDataType computeType, /* This is the only part that's different for the C++ version */
    cublasGemmAlgo_t algo
);

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const void* alpha,
    const void* const Aarray[], cudaDataType Atype, int lda,
    const void* const Barray[], cudaDataType Btype, int ldb,
    const void* beta,
    void* const Carray[], cudaDataType Ctype, int ldc,
    int batchCount,
    cudaDataType computeType,
    cublasGemmAlgo_t algo
);

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, cudaDataType Atype, int lda, long long int strideA,
    const void *B, cudaDataType Btype, int ldb, long long int strideB,
    const void *beta,
    void *C, cudaDataType Ctype, int ldc, long long int strideC,
    int batchCount,
    cudaDataType computeType, /* This is the only part that's different for the C++ version */
    cublasGemmAlgo_t algo
);
#endif /* __cplusplus */


extern "C" GPUBLAS_EXPORT cublasStatus_t cublasSgemmEx(cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const void *A, cudaDataType_t Atype, int lda,
    const void *B, cudaDataType_t Btype, int ldb,
    const float *beta,
    void *C, cudaDataType_t Ctype, int ldc
);
extern "C" GPUBLAS_EXPORT cublasStatus_t cublasCgemmEx(cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const cuComplex *alpha,
    const void *A, cudaDataType_t Atype, int lda,
    const void *B, cudaDataType_t Btype, int ldb,
    const cuComplex *beta,
    void *C, cudaDataType_t Ctype, int ldc
);

extern "C" GPUBLAS_EXPORT cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const __half *alpha,
    const __half *A, int lda,
    long long int strideA,
    const __half *B, int ldb,
    long long int strideB,
    const __half *beta,
    __half *C, int ldc,
    long long int strideC,
    int batchCount
);

extern "C" GPUBLAS_EXPORT cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda,
    long long int strideA,
    const float *B, int ldb,
    long long int strideB,
    const float *beta,
    float *C, int ldc,
    long long int strideC,
    int batchCount
);
extern "C" GPUBLAS_EXPORT cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const double *alpha,
    const double *A, int lda,
    long long int strideA,
    const double *B, int ldb,
    long long int strideB,
    const double *beta,
    double *C, int ldc,
    long long int strideC,
    int batchCount
);
extern "C" GPUBLAS_EXPORT cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const cuComplex *alpha,
    const cuComplex *A, int lda,
    long long int strideA,
    const cuComplex *B, int ldb,
    long long int strideB,
    const cuComplex *beta,
    cuComplex *C, int ldc,
    long long int strideC,
    int batchCount
);

extern "C" GPUBLAS_EXPORT cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int lda,
    long long int strideA,
    const cuDoubleComplex *B, int ldb,
    long long int strideB,
    const cuDoubleComplex *beta,
    cuDoubleComplex *C, int ldc,
    long long int strideC,
    int batchCount
);


/**
 * ScalEx()
 */
#define ScalEx_ARGS n, alpha, alphaType, x, xType, incx, executionType
DIRECT_BLAS_API_N(ScalEx, _scal_ex,
    int n,
    const void *alpha,
    cudaDataType alphaType,
    void *x,
    cudaDataType xType,
    int incx,
    cudaDataType executionType
)
#undef ScalEx_ARGS


/**
 * DotEx()
 */
#define DotEx_ARGS n, x, xType, incx, y, yType, incy, result, resultType, executionType
DIRECT_BLAS_API_N(DotEx, _dot_ex,
    int n,
    const void *x,
    cudaDataType xType,
    int incx,
    const void *y,
    cudaDataType yType,
    int incy,
    void *result,
    cudaDataType resultType,
    cudaDataType executionType
)
#undef DotEx_ARGS


/**
 * DotcEx()
 * 
 * (conjugated)
 */
#define DotcEx_ARGS n, x, xType, incx, y, yType, incy, result, resultType, executionType
DIRECT_BLAS_API_N(DotcEx, _dotc_ex,
    int n,
    const void *x,
    cudaDataType xType,
    int incx,
    const void *y,
    cudaDataType yType,
    int incy,
    void *result,
    cudaDataType resultType,
    cudaDataType executionType
)
#undef DotcEx_ARGS


// TODO: Gauss complexity reduction? :D
//cublasStatus_t cublasCgemm3mStridedBatched(cublasHandle_t handle,
//    cublasOperation_t transa,
//    cublasOperation_t transb,
//    int m, int n, int k,
//    const cuComplex *alpha,
//    const cuComplex *A, int lda,
//    long long int strideA,
//    const cuComplex *B, int ldb,
//    long long int strideB,
//    const cuComplex *beta,
//    cuComplex *C, int ldc,
//    long long int strideC,
//    int batchCount
//);
