#ifndef CUBLAS_TYPES_H
#define CUBLAS_TYPES_H

#include "cublas/export.h"
#include <rocblas/rocblas.h>
#include <driver_types.h>
#include <library_types.h>

#define CUBLAS_VER_MAJOR 12
#define CUBLAS_VER_MINOR 5
#define CUBLAS_VER_PATCH 2
#define CUBLAS_VER_BUILD 999
#define CUBLAS_VERSION ((CUBLAS_VER_MAJOR * 10000) + (CUBLAS_VER_MINOR * 100) + CUBLAS_VER_PATCH)

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================ */

/** Select whether a matrix is transposed or not. **/
typedef enum {
    CUBLAS_OP_N = 0, /*!< Not transposed. */
    CUBLAS_OP_T = 1, /*!< Transposed. */
    CUBLAS_OP_C = 2  /*!< Conjugate-transposed. */
} cublasOperation_t;

/** Indicates which of the upper or lower part of a dense triangular/hermitian matrix should be updated. **/
typedef enum {
    CUBLAS_FILL_MODE_LOWER = 0,
    CUBLAS_FILL_MODE_UPPER = 1,
    CUBLAS_FILL_MODE_FULL = 2
} cublasFillMode_t;

/** Indicates whether a matrix's diagonal consists entirely of 1s or not. **/
typedef enum {
    CUBLAS_DIAG_NON_UNIT = 0,
    CUBLAS_DIAG_UNIT = 1
} cublasDiagType_t;

/** Indicates the side matrix A is located relative to matrix B during multiplication. **/
typedef enum {
    CUBLAS_SIDE_LEFT = 141,
    CUBLAS_SIDE_RIGHT = 142,
    CUBLAS_SIDE_BOTH = 143
} cublasSideMode_t;

typedef enum {
    CUBLAS_POINTER_MODE_HOST = 0,
    CUBLAS_POINTER_MODE_DEVICE = 1
} cublasPointerMode_t;

/**
 * This enum does nothing, and is included only to make existing cuBLAS programs work without changes.
 *
 * The cublas versions of the routines in cuBLAS that change behaviour when atomics are enabled don't need to be
 * nondeterministic to be faster than cuBLAS ;)
 */
typedef enum {
    CUBLAS_ATOMICS_NOT_ALLOWED = 0,
    CUBLAS_ATOMICS_ALLOWED = 1
} cublasAtomicsMode_t;

/// Enum to determine if argument validation is enabled. It's basically a bool.
typedef enum {
    CUBLAS_ARGUMENT_VALIDATION_DISABLED = 0,
    CUBLAS_ARGUMENT_VALIDATION_ENABLED = 1
} cublasValidationMode_t;

/**
 * This enum does nothing, and is included only to make existing cuBLAS programs work without changes.
 *
 * A mysterious and barely documented tuning flag is not a useful thing to have...
 */
typedef enum {
    CUBLAS_GEMM_DEFAULT = -1,
    CUBLAS_GEMM_DFALT = -1,
    CUBLAS_GEMM_ALGO0 = 0,
    CUBLAS_GEMM_ALGO1 = 1,
    CUBLAS_GEMM_ALGO2 = 2,
    CUBLAS_GEMM_ALGO3 = 3,
    CUBLAS_GEMM_ALGO4 = 4,
    CUBLAS_GEMM_ALGO5 = 5,
    CUBLAS_GEMM_ALGO6 = 6,
    CUBLAS_GEMM_ALGO7 = 7,

    CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
    CUBLAS_GEMM_DFALT_TENSOR_OP = 99,
} cublasGemmAlgo_t;

typedef enum {
  CUBLAS_DEFAULT_MATH = 0,
  CUBLAS_TENSOR_OP_MATH = 1,
  CUBLAS_PEDANTIC_MATH = 2,
  CUBLAS_TF32_TENSOR_OP_MATH = 3,
  CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16,
} cublasMath_t;

/* Discarding status codes is bad. */
#if defined(__cplusplus) && __cplusplus >= 201703L
#define __GPUBLAS_NODISCARD [[nodiscard]]
#else
#define __GPUBLAS_NODISCARD
#endif

/** C-Style status codes.  **/
typedef enum __GPUBLAS_NODISCARD {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16,  /* <- Included only so people's switch statements compile. Never returned. */
} cublasStatus_t;

struct cublasHandle;

/** Make the types interchangeable... **/
typedef cublasHandle* cublasHandle_t;
typedef cudaDataType cublasDataType_t;

#ifdef __cplusplus
}
#endif


typedef enum {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_16F_PEDANTIC = 65,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_PEDANTIC = 69,
    CUBLAS_COMPUTE_32F_FAST_16F = 74,
    CUBLAS_COMPUTE_32F_FAST_16BF = 75,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
    CUBLAS_COMPUTE_64F = 70,
    CUBLAS_COMPUTE_64F_PEDANTIC = 71,
    CUBLAS_COMPUTE_32I = 72,
    CUBLAS_COMPUTE_32I_PEDANTIC = 73,
    CUBLAS_COMPUTE_INVALID = 999999
} cublasComputeType_t;

struct __half;

#endif /* CUBLAS_TYPES_H */
