#ifndef MATHLIBS_SOLVER_DN_H
#define MATHLIBS_SOLVER_DN_H

/// External includes

#include "cublas_v2.h"

#ifdef __cplusplus
#include "rocsolver/rocsolver.h"

/// We need some casting operators from blas lib
#ifdef SOLVER_INLINE_EVERYTHING
#define NO_BLAS_COMPILE
#include "blas_impl/shared.hpp"
#endif
#endif

/// Package-local includes

#define MATHLIBS_SOLVER_INCLUDED
#include "solver_impl/macro_hell.h"

#include "solver_impl/types.h"
#include "solver_impl/mapped_types.hpp"
#include "solver_impl/life_cycle.h"




/// Lower-Upper Factorization
C_ROC_CXX_PACK(getrf, 
    rocblas_handle, handle,
    int,                m,
    int,                n,
    DTYPE *,            A,
    int,                lda,
    int *,              devIpiv,
    int *,              devInfo)

// Complete dummy
template<typename T>
inline rocblas_status solver_xgetrf(
    rocblas_handle     handle,
    int                m,
    int                n,
    T *                A,
    int                lda,
    __CXX17UNUSED T *         Workspace,
    int *              devIpiv,
    int *              devInfo)
INLINE_BODY(
    return rocsolver_xgetrf(handle, m, n, A, lda, devIpiv, devInfo);
)

C_CU_CXX_UNPACK(Dn, getrf,
    cusolverDnHandle_t, handle,
    int,                m,
    int,                n,
    DTYPE *,            A,
    int,                lda,
    DTYPE *,            Workspace,
    int *,              devIpiv,
    int *,              devInfo)


// Complete dummy
template<typename T>
inline rocblas_status solver_xgetrf_bufferSize(
    __CXX17UNUSED rocblas_handle     handle,
    __CXX17UNUSED int                m,
    __CXX17UNUSED int                n,
    __CXX17UNUSED T *                A,
    __CXX17UNUSED int                lda,
           int *              Lwork)
INLINE_BODY(
    *Lwork = 4;
    return rocblas_status_success;
)

C_CU_CXX_UNPACK(Dn, getrf_bufferSize,
    cusolverDnHandle_t, handle,
    int,                m,
    int,                n,
    DTYPE *,            A,
    int,                lda,
    int *,              Lwork)


/// Lower-Upper solving

C_ROC_CXX_PACK(getrs,
    rocblas_handle,     handle,
    rocblas_operation,  trans,
    int,                n,
    int,                nrhs,
    DTYPE *,            A,
    int,                lda,
    const int *,        devIpiv,
    DTYPE *,            B,
    int,                ldb)

template<typename T>
inline rocblas_status solver_xgetrs(
    rocblas_handle      handle,
    rocblas_operation   trans,
    int                 n,
    int                 nrhs,
    const T*                  A,
    int                 lda,
    const int*                devIpiv,
    T*                  B,
    int                 ldb,
    __CXX17UNUSED int*         devInfo)
INLINE_BODY(
    return rocsolver_xgetrs(handle, trans, n, nrhs, const_cast<T*>(A), lda, 
        const_cast<int*>(devIpiv), B, ldb);
)

C_CU_CXX_UNPACK(Dn, getrs,
    cusolverDnHandle_t, handle,
    cublasOperation_t,  trans,
    int,                n,
    int,                nrhs,
    DTYPE const*,       A,
    int,                lda,
    const int *,        devIpiv,
    DTYPE *,            B,
    int,                ldb,
    int *,              devInfo)


#include "solver_impl/macro_hell_rescind.h"
#undef MATHLIBS_SOLVER_INCLUDED

#endif // MATHLIBS_SOLVER_DN_H
