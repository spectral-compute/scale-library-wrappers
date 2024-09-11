#ifndef CUBLAS_H_
#define CUBLAS_H_

#include "blas_impl/blas_types.h"
#include "cublas/export.h"
#include <cuComplex.h>

typedef cublasStatus_t cublasStatus;

#ifdef __cplusplus
extern "C" {
#endif

GPUBLAS_EXPORT cublasStatus cublasInit(void);
GPUBLAS_EXPORT cublasStatus cublasShutdown(void);
GPUBLAS_EXPORT cublasStatus cublasGetVersion(int* version);

GPUBLAS_EXPORT cublasStatus cublasSetKernelStream(cudaStream_t stream);

GPUBLAS_EXPORT cublasStatus cublasGetError(void);

GPUBLAS_EXPORT cublasStatus cublasAlloc(int n, int elemSize, void** devicePtr);
GPUBLAS_EXPORT cublasStatus cublasFree(void* devicePtr);

#include "blas_impl/get_set_fns.h"

#define _LEGACY_BLAS_C_FN_NAME(LIBNAME, LETTER, NAME) LIBNAME ## LETTER ## NAME
#define LEGACY_BLAS_C_FN_NAME(LIBNAME, LETTER, NAME) _LEGACY_BLAS_C_FN_NAME(LIBNAME, LETTER, NAME)

#define LEGACY_BLAS_API(LETTER, NAME, ...) \
    GPUBLAS_EXPORT void LEGACY_BLAS_C_FN_NAME(cublas, LETTER, NAME)(__VA_ARGS__);

#define LEGACY_BLAS_API_RET(RTYPE, LETTER, NAME, ...) \
    GPUBLAS_EXPORT RTYPE LEGACY_BLAS_C_FN_NAME(cublas, LETTER, NAME)(__VA_ARGS__);

#include "blas_impl/legacy.h"

#undef LEGACY_BLAS_API
#undef LEGACY_BLAS_API_RET
#undef _LEGACY_BLAS_C_FN_NAME
#undef LEGACY_BLAS_C_FN_NAME

#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif
