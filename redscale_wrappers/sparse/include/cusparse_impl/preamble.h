#ifndef MATHLIB_SPARSE_INCLUDED
#error This cannot be included directly
#endif

#ifndef MATHLIBS_SPARSE_PREAMBLE_H
#define MATHLIBS_SPARSE_PREAMBLE_H

#ifdef __cplusplus
// If we were to use rocsparse_float_complex, rocsparse_double_complex, we would get build problems 
// with any library including both cusolver and cusparse, due to cuToRoc() being a direct conversion.
// We do a slightly evil thing here.

#include <rocblas/rocblas.h>
#define _ROCSPARSE_COMPLEX_TYPES_H_
#define rocsparse_float_complex rocblas_float_complex
#define rocsparse_double_complex rocblas_double_complex

#include <rocsparse/rocsparse-types.h> 

#define rocsparse_datatype rocblas_datatype
#define rocsparse_datatype rocblas_datatype

#include <rocsparse/rocsparse.h> 
#endif // __cplusplus


#include <cuComplex.h>        // cuComplex
#include <cuda_runtime_api.h> // cudaStream_t
#include <library_types.h>    // CUDA_R_32F
#include <stdint.h>           // int64_t


#ifdef __cplusplus
#   include <cuda_fp16.h>     // __half
#endif // __cplusplus


// Version

#define CUSPARSE_VER_MAJOR 12
#define CUSPARSE_VER_MINOR 1
#define CUSPARSE_VER_PATCH 1
#define CUSPARSE_VER_BUILD 53
#define CUSPARSE_VERSION (CUSPARSE_VER_MAJOR * 1000 + \
                          CUSPARSE_VER_MINOR *  100 + \
                          CUSPARSE_VER_PATCH)


#if defined(__cplusplus)
    /* Under C++ we will do a funny thing...
       We will just make all calls to cusparse inlinable to calls to rocsparse,
       by providing all the glue definitions in this file.
    */
    #ifdef SPARSE_INLINE_EVERYTHING
        #include <rocsparse/rocsparse.h>

        #include "mapped_types.hpp"

        // ifndef is used here to allow impl.cpp to override this
        #ifndef GPUSPARSE_EXPORT_C
            #define GPUSPARSE_EXPORT_C extern "C" inline GPUSPARSE_EXPORT __attribute__((gnu_inline)) 
        #endif

        #define INLINE_BODY(X) { X }
        #define INLINE_BODY_STATUS(X) { \
            try { \
                return __redscale_cusparseStatus_t([=]{ \
                        X \
                }()); \
            } catch (CudaRocmWrapper::CosplayEnumError& e) { \
                printf("Exception occured: %s", e.what()); \
                return CUSPARSE_STATUS_INVALID_VALUE; \
            } \
        }
    #else // SPARSE_INLINE_EVERYTHING
        #define GPUSPARSE_EXPORT_C extern "C" GPUSPARSE_EXPORT
        #define INLINE_BODY(X) ;
        #define INLINE_BODY_STATUS(X) ;
    #endif // !SPARSE_INLINE_EVERYTHING
#else
    #define GPUSPARSE_EXPORT_C GPUSPARSE_EXPORT
    #define INLINE_BODY(X) ;
    #define INLINE_BODY_STATUS(X) ;
#endif // defined(__cplusplus)

// In the case where we can't inline, defer it to impl.cpp
#ifndef BODY
#define BODY(X) ;
#endif
       
#include "types.h"

#include "macro_hell.h"


// Initialization and life cycle

#ifdef SPARSE_INLINE_EVERYTHING
    #include <memory>
    #include "cosplay_impl/HIPSynchronisedStream.hpp"

    struct cusparseContext {
        // TODO: First-class support for cuda streams in rocm libs, pls :D
        std::shared_ptr<CudaRocmWrapper::HIPSynchronisedStream> stream =
            CudaRocmWrapper::HIPSynchronisedStream::getForCudaStream(nullptr);
        rocsparse_handle handle;

        cusparseContext(rocsparse_handle handle): handle(handle) {}
    };

    template <>
    struct CuToRoc<cusparseHandle_t> final
    { 
        [[maybe_unused]] __attribute__((always_inline)) 
        static rocsparse_handle& operator()(cusparseHandle_t value)
        { 
            return value->handle;
        }
    };
#endif

GPUSPARSE_EXPORT_C cusparseStatus_t
cusparseCreate(cusparseHandle_t* handle)
    INLINE_BODY_STATUS(
    rocsparse_handle h;
    MAYBE_ERROR(rocsparse_create_handle(&h));

    *handle = new cusparseContext(h);

    return rocsparse_status_success;
)

GPUSPARSE_EXPORT_C cusparseStatus_t
cusparseDestroy(cusparseHandle_t handle)
    INLINE_BODY_STATUS(
    return rocsparse_destroy_handle(handle->handle);
)

SPARSE_API_DIRECT(GetVersion, get_version,
    int*,             version)

// Let's hope nobody misses this
// GPUSPARSE_EXPORT_C cusparseStatus_t
// cusparseGetProperty(libraryPropertyType type,
//                     int*                value);

GPUSPARSE_EXPORT_C const char* 
cusparseGetErrorName(cusparseStatus_t status);

GPUSPARSE_EXPORT_C const char* 
cusparseGetErrorString(cusparseStatus_t status);

GPUSPARSE_EXPORT_C cusparseStatus_t
cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId)
INLINE_BODY_STATUS(
    handle->stream = CudaRocmWrapper::HIPSynchronisedStream::getForCudaStream(streamId);
    return rocsparse_status_success;
)

GPUSPARSE_EXPORT_C cusparseStatus_t
cusparseGetStream(cusparseHandle_t handle, cudaStream_t* streamId)
INLINE_BODY_STATUS(
    *streamId = *handle->stream;
    return rocsparse_status_success;
)

SPARSE_API_DIRECT(GetPointerMode, get_pointer_mode, 
                       cusparsePointerMode_t*, mode)

SPARSE_API_DIRECT(SetPointerMode, set_pointer_mode, 
                       cusparsePointerMode_t, mode)

#endif // MATHLIBS_SPARSE_PREAMBLE_H
