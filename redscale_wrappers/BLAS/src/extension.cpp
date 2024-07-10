#include "blas_impl/shared.hpp"
#include "cuda.h"
#include "cosplay_impl/enum_mapper.hpp"

// Ironically, AMD's compatibility macros actually break the functions by preventing normal C++ arg-conversion.
#undef rocblas_gemm_ex
#undef rocblas_gemm_strided_batched_ex

#include "blas_impl/extension.h"

extern "C" GPUBLAS_EXPORT cublasStatus_t cublasSgemmEx(cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const void *A, cudaDataType_t Atype, int lda,
    const void *B, cudaDataType_t Btype, int ldb,
    const float *beta,
    void *C, cudaDataType_t Ctype, int ldc
) {
    return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
extern "C" GPUBLAS_EXPORT cublasStatus_t cublasCgemmEx(cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const cuComplex *alpha,
    const void *A, cudaDataType_t Atype, int lda,
    const void *B, cudaDataType_t Btype, int ldb,
    const cuComplex *beta,
    void *C, cudaDataType_t Ctype, int ldc
) {
    return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

namespace {

cublasComputeType_t computeTypeFromDataType(cudaDataType computeType) {
    // It's very unclear how this mapping should work. The manual says the C version will _just compile_,
    // but since the respective enums have no overlap that definitely isn't true. Here we implement a common-sense
    // mapping that hopefully has the desired effect.
    switch (computeType) {
        case CUDA_R_16F:
            return CUBLAS_COMPUTE_16F;
        case CUDA_R_32F:
            return CUBLAS_COMPUTE_32F;
        case CUDA_R_64F:
            return CUBLAS_COMPUTE_64F;
        case CUDA_R_32I:
        case CUDA_R_32U:
            return CUBLAS_COMPUTE_32I;
        default:
            return CUBLAS_COMPUTE_INVALID;
    }
}

} // Anonymous namespace

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
    cudaDataType computeType, /* This is the only part that's different */
    cublasGemmAlgo_t algo
) {
    cublasComputeType_t realCompTy = computeTypeFromDataType(computeType);
    if (realCompTy == CUBLAS_COMPUTE_INVALID) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, realCompTy, algo);
}

GPUBLAS_EXPORT cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle,
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
) {
    cublasComputeType_t realCompTy = computeTypeFromDataType(computeType);
    if (realCompTy == CUBLAS_COMPUTE_INVALID) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    return cublasGemmBatchedEx(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, realCompTy, algo);
}

GPUBLAS_EXPORT cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, cudaDataType Atype, int lda, long long int strideA,
    const void *B, cudaDataType Btype, int ldb, long long int strideB,
    const void *beta,
    void *C, cudaDataType Ctype, int ldc, long long int strideC,
    int batchCount,
    cudaDataType computeType, /* This is the only part that's different */
    cublasGemmAlgo_t algo
) {
    cublasComputeType_t realCompTy = computeTypeFromDataType(computeType);
    if (realCompTy == CUBLAS_COMPUTE_INVALID) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    return cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, realCompTy, algo);
}


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
) {
    return cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_16F, lda, strideA, B, CUDA_R_16F, ldb, strideB, beta, C, CUDA_R_16F, ldc, strideC, batchCount, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
}

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
) {
    return cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_32F, lda, strideA, B, CUDA_R_32F, ldb, strideB, beta, C, CUDA_R_32F, ldc, strideC, batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
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
) {
    return cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_64F, lda, strideA, B, CUDA_R_64F, ldb, strideB, beta, C, CUDA_R_64F, ldc, strideC, batchCount, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
};
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
) {
    return cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, CUDA_C_32F, lda, strideA, B, CUDA_C_32F, ldb, strideB, beta, C, CUDA_C_32F, ldc, strideC, batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

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
) {
    return cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, CUDA_C_64F, lda, strideA, B, CUDA_C_64F, ldb, strideB, beta, C, CUDA_C_64F, ldc, strideC, batchCount, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
}
