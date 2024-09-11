#ifndef GPUBLAS_APIS_H
#define GPUBLAS_APIS_H

// Alien's helper_cuda.h looks for this.
#define CUBLAS_API_H_

#include <cuComplex.h>
#include "cublas/export.h"
#include "blas_types.h"

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else /* __cplusplus */
#define EXTERN_C
#endif /* __cplusplus */

/* Handy macros for declaring BLAS functions, and their handy C++ polymorphic overload wrappers. */
#define _BLAS_C_FN_NAME(LETTER, NAME) cublas ## LETTER ## NAME ## _v2
#define BLAS_C_FN_NAME(LETTER, NAME) _BLAS_C_FN_NAME(LETTER, NAME)

#define BLAS_C_API(LETTER, NAME, ...) \
    EXTERN_C GPUBLAS_EXPORT cublasStatus_t \
    BLAS_C_FN_NAME(LETTER, NAME)(cublasHandle_t handle, __VA_ARGS__);

/* A BLAS API function that doesn't take a "letter" representing type. */
#define DIRECT_BLAS_API(NAME, ...) \
    EXTERN_C GPUBLAS_EXPORT cublasStatus_t \
    cublas ## NAME(cublasHandle_t handle, __VA_ARGS__);

/* A BLAS API function that doesn't take a "letter" representing type. */
#define DIRECT_BLAS_API_N(NAME, ROCNAME, ...) \
    EXTERN_C GPUBLAS_EXPORT cublasStatus_t \
    cublas ## NAME(cublasHandle_t handle, __VA_ARGS__);


#ifdef __cplusplus

    /*
     * In C++, also provide polymorphic overrides. If you're working with templates, it's incredibly annoying to have to
     * call sscal/dscal depending on type. By providing overrides, users can just call scal() and it'll resolve properly.
     * To avoid binary size silliness (like the compiler inlining the entire implementation into both the C and C++ entry
     * points), this API wrapper is defined in the header and inlines into user code (eventually compiling to just a call to
     * the appropriate C89 BLAS function).
     */
    #define CXX_BLAS_WRAPPER(LETTER, NAME, CXX_NAME, ...) \
        inline cublasStatus_t \
        cublas ## CXX_NAME(cublasHandle_t handle, __VA_ARGS__) { \
            return BLAS_C_FN_NAME(LETTER, NAME)(handle, NAME ## _ARGS); \
        }

#else
    #define CXX_BLAS_WRAPPER(LETTER, NAME, CXX_NAME, ...)
#endif

#define BLAS_API(CULETTER, ROCLETTER, NAME, CXXNAME, ...) \
    EXTERN_C GPUBLAS_EXPORT cublasStatus_t \
    BLAS_C_FN_NAME(CULETTER, NAME)(cublasHandle_t handle, __VA_ARGS__); \
    CXX_BLAS_WRAPPER(CULETTER, NAME, CXXNAME, __VA_ARGS__);

#include "blas_impl/level1.h"
#include "blas_impl/level2.h"
#include "blas_impl/level3.h"
#include "blas_impl/extension.h"

/* Don't leak macros. */
#undef BLAS_C_API
#undef BLAS_API
#undef DIRECT_BLAS_API
#undef BLAS_C_FN_NAME
#undef _BLAS_C_FN_NAME
#undef CXX_BLAS_WRAPPER
#undef EXTERN_C
#undef DIRECT_BLAS_API_N

#endif /* GPUBLAS_APIS_H */
