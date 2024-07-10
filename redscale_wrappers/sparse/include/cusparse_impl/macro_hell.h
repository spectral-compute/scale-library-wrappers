#ifndef MACRO_HELL_H
#define MACRO_HELL_H

#include "common.h"


#include "cusparse/export.h"
#include "rocsparse/rocsparse-types.h"
#include "redscale_impl/cppmagic/PushMagic.hpp"
#include "redscale_impl/impl_defines.h"

#define CU_TO_ROC(X) cuToRoc(X)


/* Handy macros for declaring BLAS functions, and their handy C++ polymorphic overload wrappers. */
#define SPARSE_C_FN_NAME(LETTER, NAME) cusparse ## LETTER ## NAME

#define SPARSE_C_API(LETTER, NAME, ...) \
    GPUSPARSE_EXPORT_C cusparseStatus_t \
    SPARSE_C_FN_NAME(LETTER, NAME)(cusparseHandle_t handle, __VA_ARGS__)

#ifdef __cplusplus
    /*
     * In C++, also provide polymorphic overrides. If you're working with templates, it's incredibly annoying to have to
     * call sscal/dscal depending on type. By providing overrides, users can just call scal() and it'll resolve properly.
     * To avoid binary size silliness (like the compiler inlining the entire implementation into both the C and C++ entry
     * points), this API wrapper is defined in the header and inlines into user code (eventually compiling to just a call to
     * the appropriate C89 SPARSE function).
     */
    #define SPARSE_CXX_WRAPPER(LETTER, NAME, CXX_NAME, ...) \
        inline cusparseStatus_t \
        cusparse ## CXX_NAME(cusparseHandle_t handle, MAP_KV(SPARSE_CAT_ARG, COMMA, __VA_ARGS__)) { \
            return SPARSE_C_FN_NAME(LETTER, NAME)(handle, MAP_KV(SECOND, COMMA, __VA_ARGS__)); \
        }

#else
    #define SPARSE_CXX_WRAPPER(LETTER, NAME, CXX_NAME, ...)
#endif


#define SPARSE_CAT_ARG(TYPE, NAME) TYPE NAME
#define SPARSE_USE_ARG(TYPE, NAME) CU_TO_ROC(NAME)

#define SPARSE_API_DIRECT_NOHANDLE(NAME, ROCNAME, ...) \
    GPUSPARSE_EXPORT_C cusparseStatus_t \
    cusparse ## NAME(MAP_KV(SPARSE_CAT_ARG, COMMA, __VA_ARGS__)) \
        INLINE_BODY_STATUS(return rocsparse_ ## ROCNAME \
            (MAP_KV(SPARSE_USE_ARG, COMMA, __VA_ARGS__));)

#define SPARSE_API_DIRECT_CANT_INLINE(NAME, ROCNAME, ...) \
    GPUSPARSE_EXPORT_C cusparseStatus_t \
    cusparse ## NAME(cusparseHandle_t handle, MAP_KV(SPARSE_CAT_ARG, COMMA, __VA_ARGS__)) \
        BODY(return __redscale_cusparseStatus_t(rocsparse_ ## ROCNAME \
            (handle->handle, MAP_KV(SPARSE_USE_ARG, COMMA, __VA_ARGS__)));)

#define SPARSE_API_DIRECT(NAME, ROCNAME, ...) \
    GPUSPARSE_EXPORT_C cusparseStatus_t \
    cusparse ## NAME(cusparseHandle_t handle, MAP_KV(SPARSE_CAT_ARG, COMMA, __VA_ARGS__)) \
        INLINE_BODY_STATUS(return rocsparse_ ## ROCNAME \
            (handle->handle, MAP_KV(SPARSE_USE_ARG, COMMA, __VA_ARGS__));)

#define SPARSE_API(CULETTER, ROCLETTER, NAME, CXXNAME, ...) \
    GPUSPARSE_EXPORT_C cusparseStatus_t \
    SPARSE_C_FN_NAME(CULETTER, NAME)(cusparseHandle_t handle, MAP_KV(SPARSE_CAT_ARG, COMMA, __VA_ARGS__)) \
        INLINE_BODY_STATUS(return rocsparse_ ## ROCLETTER ## NAME \
                        (handle->handle, MAP_KV(SPARSE_USE_ARG, COMMA, __VA_ARGS__));) \
    SPARSE_CXX_WRAPPER(CULETTER, NAME, CXXNAME, __VA_ARGS__)


#define MAYBE_ERROR(X)     \
    {                             \
        rocsparse_status ret = X; \
        if (ret != 0)             \
            return ret;           \
    }

#endif // MACRO_HELL_H
