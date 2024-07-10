#ifndef MATHLIBS_SOLVER_EXPORT_FLAGS_H
#define MATHLIBS_SOLVER_EXPORT_FLAGS_H

#include "common.h"

#include "cusolver/export.h"

#define CU_TO_ROC(X) cuToRoc(X)

#if defined(__cplusplus)
    /* Under C++ we will do a funny thing...
       We will just make all calls to cusolver inlinable to calls to rocsparse,
       by providing all the glue definitions in this file.
    */
    #ifdef SOLVER_INLINE_EVERYTHING

        // ifndef is used here to allow impl.cpp to override this
        #ifndef GPUSOLVER_EXPORT_C
        #define GPUSOLVER_EXPORT_C extern "C" inline GPUSOLVER_EXPORT __attribute__((gnu_inline))
        #endif

        #define INLINE_BODY(X) { X }
        #define INLINE_BODY_STATUS(X) { return __redscale_cusolverStatus_t([=]{X}()); }
    #else
        #define GPUSOLVER_EXPORT_C extern "C" GPUSOLVER_EXPORT
        #define INLINE_BODY(X) ;
        #define INLINE_BODY_STATUS(X) ;
    #endif
#else
    #define GPUSOLVER_EXPORT_C GPUSOLVER_EXPORT
    #define INLINE_BODY(X) ;
    #define INLINE_BODY_STATUS(X) ;
#endif // defined(__cplusplus)

// In the case where we can't inline, defer it to impl.cpp
#ifndef BODY
#define BODY(X) ;
#endif

#endif // MATHLIBS_SOLVER_EXPORT_FLAGS_H
