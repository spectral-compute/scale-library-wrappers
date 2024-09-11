#ifndef MATHLIBS_BLAS_SHARED_H
#define MATHLIBS_BLAS_SHARED_H

#include "blas_types.h"
#include "cuda_fp16.h"
#include "cosplay_impl/cu_to_roc.hpp"
#include "cosplay_impl/cu_to_roc_cmplx.hpp"
#include "cosplay_impl/HIPSynchronisedStream.hpp"
#include <rocblas/rocblas.h>

#define COMMA() ,
#define CU_TO_ROC(X) cuToRoc(X)

#ifndef NO_BLAS_COMPILE

/* Handy macros for declaring BLAS functions, and their handy C++ polymorphic overload wrappers. */
#define _BLAS_C_FN_NAME(LETTER, NAME) cublas ## LETTER ## NAME ## _v2
#define BLAS_C_FN_NAME(LETTER, NAME) _BLAS_C_FN_NAME(LETTER, NAME)

#define _ROC_C_FN_NAME(LETTER, NAME) rocblas ## _ ## LETTER ## NAME
#define ROC_C_FN_NAME(LETTER, NAME) _ROC_C_FN_NAME(LETTER, NAME)

#define _LEGACY_BLAS_C_FN_NAME(LIBNAME, LETTER, NAME) LIBNAME ## LETTER ## NAME
#define LEGACY_BLAS_C_FN_NAME(LIBNAME, LETTER, NAME) _LEGACY_BLAS_C_FN_NAME(LIBNAME, LETTER, NAME)

/* A BLAS API function that doesn't take a "letter" representing type. */
#define DIRECT_BLAS_API_N(NAME, ROCNAME, ...) \
    extern "C" GPUBLAS_EXPORT cublasStatus_t \
    cublas ## NAME(cublasHandle_t handle, __VA_ARGS__) { \
        CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*handle->stream}; \
        return mapReturnCode(rocblas ## ROCNAME (MAP(CU_TO_ROC, COMMA, handle->handle, NAME ## _ARGS))); \
    }

#define DIRECT_BLAS_API(NAME, ...) \
    DIRECT_BLAS_API_N(NAME, NAME, __VA_ARGS__);

#define BLAS_API(CULETTER, ROCLETTER, NAME, CXXNAME, ...) \
    extern "C" GPUBLAS_EXPORT cublasStatus_t \
    BLAS_C_FN_NAME(CULETTER, NAME)(cublasHandle_t handle, __VA_ARGS__) { \
            CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*handle->stream}; \
            return mapReturnCode(ROC_C_FN_NAME(ROCLETTER, NAME) \
                                 (MAP(CU_TO_ROC, COMMA, handle->handle, NAME ## _ARGS))); \
    }


#endif

namespace {

__attribute__((always_inline))
inline cublasStatus_t mapReturnCode(rocblas_status rocStatus) {
    switch (rocStatus) {
        case rocblas_status_success:
        case rocblas_status_continue:
            return CUBLAS_STATUS_SUCCESS;
        case rocblas_status_invalid_handle:
            return CUBLAS_STATUS_NOT_INITIALIZED;
        case rocblas_status_not_implemented:
            return CUBLAS_STATUS_NOT_SUPPORTED;
        case rocblas_status_invalid_pointer:
        case rocblas_status_invalid_size:
            return CUBLAS_STATUS_INVALID_VALUE;
        case rocblas_status_memory_error:
            return CUBLAS_STATUS_ALLOC_FAILED;
        default:
            // "Something went wrong"
            return CUBLAS_STATUS_INTERNAL_ERROR;
    }
}

MAP_ENUM_EXHAUSTIVE(rocblas_operation, cublasOperation_t,
     (CUBLAS_OP_N, rocblas_operation_none),
     (CUBLAS_OP_T, rocblas_operation_transpose),
     (CUBLAS_OP_C, rocblas_operation_conjugate_transpose)
)

MAP_ENUM_EXHAUSTIVE(rocblas_fill, cublasFillMode_t,
     (CUBLAS_FILL_MODE_UPPER, rocblas_fill_upper),
     (CUBLAS_FILL_MODE_LOWER, rocblas_fill_lower),
     (CUBLAS_FILL_MODE_FULL, rocblas_fill_full)
)

MAP_ENUM_EXHAUSTIVE(rocblas_diagonal, cublasDiagType_t,
     (CUBLAS_DIAG_NON_UNIT, rocblas_diagonal_non_unit),
     (CUBLAS_DIAG_UNIT, rocblas_diagonal_unit)
)

MAP_ENUM_EXHAUSTIVE(rocblas_side, cublasSideMode_t,
     (CUBLAS_SIDE_LEFT, rocblas_side_left),
     (CUBLAS_SIDE_RIGHT, rocblas_side_right),
     (CUBLAS_SIDE_BOTH, rocblas_side_both)
)

MAP_ENUM_EXHAUSTIVE(rocblas_atomics_mode, cublasAtomicsMode_t,
     (CUBLAS_ATOMICS_NOT_ALLOWED, rocblas_atomics_not_allowed),
     (CUBLAS_ATOMICS_ALLOWED, rocblas_atomics_allowed)
)

MAP_ENUM_EXHAUSTIVE(rocblas_pointer_mode, cublasPointerMode_t,
     (CUBLAS_POINTER_MODE_HOST, rocblas_pointer_mode_host),
     (CUBLAS_POINTER_MODE_DEVICE, rocblas_pointer_mode_device)
)

MAP_ENUM_PARTIAL(rocblas_datatype, cudaDataType,
     (CUDA_R_16F, rocblas_datatype_f16_r),
     (CUDA_R_32F, rocblas_datatype_f32_r),
     (CUDA_R_64F, rocblas_datatype_f64_r),
     (CUDA_C_16F, rocblas_datatype_f16_c),
     (CUDA_C_32F, rocblas_datatype_f32_c),
     (CUDA_C_64F, rocblas_datatype_f64_c),
     (CUDA_R_8I, rocblas_datatype_i8_r),
     (CUDA_R_8U, rocblas_datatype_u8_r),
     (CUDA_R_32I, rocblas_datatype_i32_r),
     (CUDA_R_32U, rocblas_datatype_u32_r),
     (CUDA_C_8I, rocblas_datatype_i8_c),
     (CUDA_C_8U, rocblas_datatype_u8_c),
     (CUDA_C_32I, rocblas_datatype_i32_c),
     (CUDA_C_32U, rocblas_datatype_u32_c),
     (CUDA_R_16BF, rocblas_datatype_bf16_r),
     (CUDA_C_16BF, rocblas_datatype_bf16_c)
)

ENUM_CU_TO_ROC_EXHAUSTIVE(rocblas_datatype, cublasComputeType_t,
     (CUBLAS_COMPUTE_16F, rocblas_datatype_f16_r),
     (CUBLAS_COMPUTE_16F_PEDANTIC, rocblas_datatype_f16_r),
     (CUBLAS_COMPUTE_32F, rocblas_datatype_f32_r),
     (CUBLAS_COMPUTE_32F_PEDANTIC, rocblas_datatype_f32_r),
     (CUBLAS_COMPUTE_32F_FAST_16F, rocblas_datatype_f32_r),
     (CUBLAS_COMPUTE_32F_FAST_16BF, rocblas_datatype_f32_r),
     (CUBLAS_COMPUTE_32F_FAST_TF32, rocblas_datatype_f32_r),
     (CUBLAS_COMPUTE_64F, rocblas_datatype_f64_r),
     (CUBLAS_COMPUTE_64F_PEDANTIC, rocblas_datatype_f64_r),
     (CUBLAS_COMPUTE_32I, rocblas_datatype_i32_r),
     (CUBLAS_COMPUTE_32I_PEDANTIC, rocblas_datatype_i32_r)
)

ENUM_CU_TO_ROC_PARTIAL(rocblas_gemm_algo, cublasGemmAlgo_t,
     (CUBLAS_GEMM_DEFAULT, rocblas_gemm_algo_standard),
     (CUBLAS_GEMM_ALGO0, rocblas_gemm_algo_standard),
     (CUBLAS_GEMM_ALGO1, rocblas_gemm_algo_standard),
     (CUBLAS_GEMM_ALGO2, rocblas_gemm_algo_standard),
     (CUBLAS_GEMM_ALGO3, rocblas_gemm_algo_standard),
     (CUBLAS_GEMM_ALGO4, rocblas_gemm_algo_standard),
     (CUBLAS_GEMM_ALGO5, rocblas_gemm_algo_standard),
     (CUBLAS_GEMM_ALGO6, rocblas_gemm_algo_standard),
     (CUBLAS_GEMM_ALGO7, rocblas_gemm_algo_standard),
     
     (CUBLAS_GEMM_DEFAULT_TENSOR_OP, rocblas_gemm_algo_standard)
)

} // Anonymous namespace

struct cublasHandle {
    // TODO: First-class support for cuda streams in rocm libs, pls :D
    std::shared_ptr<CudaRocmWrapper::HIPSynchronisedStream> stream =
        CudaRocmWrapper::HIPSynchronisedStream::getForCudaStream(nullptr);
    rocblas_handle handle;

    operator rocblas_handle() {
        return handle;
    }
};

template <>
struct CuToRoc<__half *> final
{
    [[maybe_unused]] __attribute__((always_inline))
    static rocblas_half *operator()(__half *value)
    {
        return reinterpret_cast<rocblas_half *>(value); /*-fno-strict-aliasing is enabled*/
    }
};

#undef MAP_ENUM_ONE_WAY_SUBSET
#include "cosplay_impl/pop_enum_mapper.hpp"

#endif // MATHLIBS_BLAS_SHARED_H
