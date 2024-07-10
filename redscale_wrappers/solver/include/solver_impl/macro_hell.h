#ifndef SOLVER_MACRO_HELL_H
#define SOLVER_MACRO_HELL_H

#include "common.h"

#include "export_flags.h"

#include "redscale_impl/cppmagic/PushMagic.hpp"
#include "redscale_impl/impl_defines.h"


/* Handy macros for declaring BLAS functions, and their handy C++ polymorphic overload wrappers. */
#define SOLVER_C_FN_NAME(LETTER, NAME) cusolver ## LETTER ## NAME

#define SOLVER_C_API(LETTER, NAME, ...) \
    GPUSOLVER_EXPORT_C cusolverStatus_t \
    SOLVER_C_FN_NAME(LETTER, NAME)(cusolverHandle_t handle, __VA_ARGS__)

#ifdef __cplusplus
    /*
     * In C++, also provide polymorphic overrides. If you're working with templates, it's incredibly annoying to have to
     * call sscal/dscal depending on type. By providing overrides, users can just call scal() and it'll resolve properly.
     * To avoid binary size silliness (like the compiler inlining the entire implementation into both the C and C++ entry
     * points), this API wrapper is defined in the header and inlines into user code (eventually compiling to just a call to
     * the appropriate C89 SPARSE function).
     */
    #define SOLVER_CXX_WRAPPER(LETTER, NAME, CXX_NAME, ...) \
        inline cusolverStatus_t \
        cusolver ## CXX_NAME(cusolverHandle_t handle, MAP_KV(SOLVER_CAT_ARG, COMMA, __VA_ARGS__)) { \
            return SOLVER_C_FN_NAME(LETTER, NAME)(handle, MAP_KV(SECOND, COMMA, __VA_ARGS__)); \
        }

#else
    #define SOLVER_CXX_WRAPPER(LETTER, NAME, CXX_NAME, ...)
#endif


#define SOLVER_CAT_ARG(TYPE, NAME) TYPE NAME
#define SOLVER_USE_ARG(TYPE, NAME) CU_TO_ROC(NAME)

#define SOLVER_API_DIRECT_NOHANDLE(NAME, ROCNAME, ...) \
    GPUSOLVER_EXPORT_C cusolverStatus_t \
    cusolver ## NAME(MAP_KV(SOLVER_CAT_ARG, COMMA, __VA_ARGS__)) \
        INLINE_BODY_STATUS(return rocsolver_ ## ROCNAME \
            (MAP_KV(SOLVER_USE_ARG, COMMA, __VA_ARGS__));)

#define SOLVER_API_DIRECT_CANT_INLINE(NAME, ROCNAME, ...) \
    GPUSOLVER_EXPORT_C cusolverStatus_t \
    cusolver ## NAME(cusolverHandle_t handle, MAP_KV(SOLVER_CAT_ARG, COMMA, __VA_ARGS__)) \
        BODY ( \
            CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*handle->stream}; \
            return __redscale_cusolverStatus_t(rocSOLVER_ ## ROCNAME \
                                              (handle->handle, MAP_KV(SOLVER_USE_ARG, COMMA, __VA_ARGS__))); \
        )

#define SOLVER_API_DIRECT(NAME, ROCNAME, ...) \
    GPUSOLVER_EXPORT_C cusolverStatus_t \
    cusolver ## NAME(cusolverHandle_t handle, MAP_KV(SOLVER_CAT_ARG, COMMA, __VA_ARGS__)) \
        INLINE_BODY_STATUS ( \
            CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*handle->stream}; \
            return rocsolver_ ## ROCNAME \
                   (handle->handle, MAP_KV(SOLVER_USE_ARG, COMMA, __VA_ARGS__)); \
            )

#define SOLVER_API_LETTER(CULETTER, ROCLETTER, NAME, CXXNAME, ...) \
    GPUSOLVER_EXPORT_C cusolverStatus_t \
    SOLVER_C_FN_NAME(CULETTER, NAME)(cusolverHandle_t handle, MAP_KV(SOLVER_CAT_ARG, COMMA, __VA_ARGS__)) \
        INLINE_BODY_STATUS( \
            CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*handle->stream}; \
            return rocsolver_ ## ROCLETTER ## NAME \
                   (handle->handle, MAP_KV(SOLVER_USE_ARG, COMMA, __VA_ARGS__)); \
            ) \
    SOLVER_CXX_WRAPPER(CULETTER, NAME, CXXNAME, __VA_ARGS__)

#define SOLVER_API(NAME, CXXNAME, ...) \
    SOLVER_API_LETTER(S, s, NAME, CXXNAME, __VA_ARGS__) \
    SOLVER_API_LETTER(D, d, NAME, CXXNAME, __VA_ARGS__) \
    SOLVER_API_LETTER(C, c, NAME, CXXNAME, __VA_ARGS__) \
    SOLVER_API_LETTER(Z, z, NAME, CXXNAME, __VA_ARGS__) \

#define MAYBE_ERROR(X)     \
    {                             \
        rocblas_status ret = X; \
        if (ret != 0)             \
            return ret;           \
    }



#define T_DTYPE T
#define FLOAT_DTYPE float
#define DOUBLE_DTYPE double
#define COMPLEX_FLOAT_DTYPE rocblas_float_complex
#define COMPLEX_DOUBLE_DTYPE rocblas_double_complex

#define CU_FLOAT_DTYPE float
#define CU_DOUBLE_DTYPE double
#define CU_COMPLEX_FLOAT_DTYPE cuComplex
#define CU_COMPLEX_DOUBLE_DTYPE cuDoubleComplex

#define CHECK_DTYPE_DTYPE 0, 

#define EXPAND_DTYPE(DTYPE, ARG) \
    IF_ELSE(FIRST(CHECK_DTYPE_ ## ARG))( ARG, DTYPE ## _ ## ARG)


#define C_ROC_CXX_PACK(NAME, ...) \
    _C_ROC_CXX_PACK_IMPL(NAME, \
        MAP_KV(SOLVER_CAT_ARG, COMMA, __VA_ARGS__), \
        MAP_KV(SECOND, COMMA, __VA_ARGS__))

#define _C_ROC_CXX_PACK_IMPL(NAME, TYPE_ARG_LIST, ARG_LIST) \
    template<typename T> \
    inline rocblas_status rocsolver_x ## NAME(CURRIED_MAP(EXPAND_DTYPE, T, COMMA, TYPE_ARG_LIST)); \
    template<> \
    inline rocblas_status rocsolver_x ## NAME(CURRIED_MAP(EXPAND_DTYPE, FLOAT, COMMA, TYPE_ARG_LIST)) { \
        return rocsolver_s ## NAME(ARG_LIST); \
    } \
    template<> \
    inline rocblas_status rocsolver_x ## NAME(CURRIED_MAP(EXPAND_DTYPE, DOUBLE, COMMA, TYPE_ARG_LIST)) { \
        return rocsolver_d ## NAME(ARG_LIST); \
    } \
    template<> \
    inline rocblas_status rocsolver_x ## NAME(CURRIED_MAP(EXPAND_DTYPE, COMPLEX_FLOAT, COMMA, TYPE_ARG_LIST)) { \
        return rocsolver_c ## NAME(ARG_LIST); \
    } \
    template<> \
    inline rocblas_status rocsolver_x ## NAME(CURRIED_MAP(EXPAND_DTYPE, COMPLEX_DOUBLE, COMMA, TYPE_ARG_LIST)) { \
        return rocsolver_z ## NAME(ARG_LIST); \
    }

// Macro that creates the expected 4 C specializations of a function that map to rocsolver_x##NAME
#define C_CU_CXX_UNPACK(LIB, NAME, ...) \
    _C_CU_CXX_UNPACK_IMPL(LIB, NAME, \
        MAP_KV(SOLVER_CAT_ARG, COMMA, __VA_ARGS__), \
        MAP(cuToRoc, COMMA, MAP_KV(SECOND, COMMA, __VA_ARGS__)))

#define _C_CU_CXX_UNPACK_IMPL(LIB, NAME, TYPE_ARG_LIST, ARG_LIST) \
    GPUSOLVER_EXPORT_C cusolverStatus_t cusolver## LIB ##S ## NAME(CURRIED_MAP(EXPAND_DTYPE, CU_FLOAT, COMMA, TYPE_ARG_LIST)) \
        INLINE_BODY_STATUS( \
        return solver_x ## NAME(ARG_LIST); \
    ) \
    GPUSOLVER_EXPORT_C cusolverStatus_t cusolver## LIB ##D ## NAME(CURRIED_MAP(EXPAND_DTYPE, CU_DOUBLE, COMMA, TYPE_ARG_LIST)) \
        INLINE_BODY_STATUS( \
        return solver_x ## NAME(ARG_LIST); \
    ) \
    GPUSOLVER_EXPORT_C cusolverStatus_t cusolver## LIB ##C ## NAME(CURRIED_MAP(EXPAND_DTYPE, CU_COMPLEX_FLOAT, COMMA, TYPE_ARG_LIST)) \
        INLINE_BODY_STATUS( \
        return solver_x ## NAME(ARG_LIST); \
    ) \
    GPUSOLVER_EXPORT_C cusolverStatus_t cusolver## LIB ##Z ## NAME(CURRIED_MAP(EXPAND_DTYPE, CU_COMPLEX_DOUBLE, COMMA, TYPE_ARG_LIST)) \
        INLINE_BODY_STATUS( \
        return solver_x ## NAME(ARG_LIST); \
    )


#endif // SOLVER_MACRO_HELL_H
