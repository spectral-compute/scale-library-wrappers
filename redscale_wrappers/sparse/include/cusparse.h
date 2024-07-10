#ifndef MATHLIBS_SPARSE_H
#define MATHLIBS_SPARSE_H

#define MATHLIB_SPARSE_INCLUDED

#ifdef __cplusplus
// If we were to use rocsparse_float_complex, rocsparse_double_complex, we would get build problems 
// with any library including both cusolver and cusparse, due to cuToRoc() being a direct conversion.
// We do a slightly evil thing here.

#include <rocblas/rocblas.h>
// For rocsparse <  6.0.0
#define _ROCSPARSE_COMPLEX_TYPES_H_
// For rocsparse >= 6.0.0
#define ROCSPARSE_COMPLEX_TYPES_H
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

        #include "cusparse_impl/mapped_types.hpp"

        // ifndef is used here to allow impl.cpp to override this
        #ifndef GPUSPARSE_EXPORT_C
            #define GPUSPARSE_EXPORT_C extern "C" inline GPUSPARSE_EXPORT __attribute__((gnu_inline)) 
        #endif

        #define INLINE_BODY(X) { X }
        #define INLINE_BODY_STATUS(X) { return __redscale_cusparseStatus_t([=]{X}()); }
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
       
#include "cusparse_impl/types.h"

#include "cusparse_impl/macro_hell.h"

#include "cusparse_impl/life_cycle.h"

#include "cusparse_impl/logging.h"
#include "cusparse_impl/helpers.h"

#include "cusparse_impl/level_2.h"
#include "cusparse_impl/level_3.h"

#include "cusparse_impl/preconditioners.h"
#include "cusparse_impl/extra.h"

#include "cusparse_impl/reordering_and_format_conversion.h"

#include "cusparse_impl/sorting.h"

#include "cusparse_impl/vector.h"
#include "cusparse_impl/matrix.h"

#include "cusparse_impl/op_vector_vector.h"
#include "cusparse_impl/op_matrix_vector.h"
#include "cusparse_impl/op_matrix_matrix.h"

#include "cusparse_impl/macro_hell_rescind.h"

#undef MATHLIB_SPARSE_INCLUDED

#endif // MATHLIBS_SPARSE_H
