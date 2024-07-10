#ifndef MATHLIBS_SOLVER_MAPPED_TYPES_H
#define MATHLIBS_SOLVER_MAPPED_TYPES_H

#include "common.h"

#include <cuComplex.h>        // cuComplex
#include <cuda_runtime_api.h> // cudaStream_t
#include <library_types.h>    // CUDA_R_32F
#include <stdint.h>           // int64_t
#include <stdio.h>            // FILE*


#include "cosplay_impl/cu_to_roc.hpp"
#include "cosplay_impl/cu_to_roc_cmplx.hpp"

#include "rocblas/internal/rocblas-types.h"
#include "types.h"
#include "rocsolver/rocsolver.h"

struct __redscale_cusolverStatus_t {
    cusolverStatus_t value;

    __redscale_cusolverStatus_t(rocblas_status rocStatus) {
        switch (rocStatus) {
            case rocblas_status_success:   
                value = CUSOLVER_STATUS_SUCCESS;
                break;
            case rocblas_status_invalid_handle:   
                value = CUSOLVER_STATUS_INVALID_VALUE;
                break;
            case rocblas_status_not_implemented:   
                value = CUSOLVER_STATUS_NOT_SUPPORTED;
                break;
            case rocblas_status_invalid_pointer:   
                value = CUSOLVER_STATUS_INVALID_VALUE;
                break;
            case rocblas_status_invalid_size:   
                value = CUSOLVER_STATUS_INVALID_VALUE;
                break;
            case rocblas_status_memory_error:   
                value = CUSOLVER_STATUS_ALLOC_FAILED;
                break;
            case rocblas_status_internal_error:   
                value = CUSOLVER_STATUS_INTERNAL_ERROR;
                break;
            case rocblas_status_invalid_value:   
                value = CUSOLVER_STATUS_INVALID_VALUE;
                break;
            default:
                // "Something went wrong"
                value = CUSOLVER_STATUS_INTERNAL_ERROR;
                break;
        }
    }

    operator cusolverStatus_t() {
        return this->value;
    }
};


#endif // MATHLIBS_SOLVER_MAPPED_TYPES_H
