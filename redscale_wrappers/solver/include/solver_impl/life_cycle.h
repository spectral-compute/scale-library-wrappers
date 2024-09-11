
#include "common.h"

#include <cuda_runtime_api.h> // cudaStream_t

#ifdef SOLVER_INLINE_EVERYTHING
#include <memory>

#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include "cosplay_impl/HIPSynchronisedStream.hpp"
#include "mapped_types.hpp"

struct __redscaleDnHandle {
    // TODO: First-class support for cuda streams in rocm libs, pls :D
    std::shared_ptr<CudaRocmWrapper::HIPSynchronisedStream> stream =
        CudaRocmWrapper::HIPSynchronisedStream::getForCudaStream(nullptr);
    rocblas_handle handle;

    __redscaleDnHandle(rocblas_handle handle): handle(handle) {}
};

#define cusolverDnHandle_t __redscaleDnHandle* 


template <>
struct CuToRoc<cusolverDnHandle_t> final
{ 
    [[maybe_unused]] __attribute__((always_inline)) 
    static rocblas_handle& operator()(cusolverDnHandle_t value)
    { 
        return value->handle;
    }
};
#else // SOLVER_INLINE_EVERYTHING

struct cusolverDnHandle;
typedef cusolverDnHandle* cusolverDnHandle_t;

#endif


#include "macro_hell.h"


GPUSOLVER_EXPORT_C cusolverStatus_t
cusolverDnCreate(cusolverDnHandle_t *handle)
INLINE_BODY_STATUS(
    CudaRocmWrapper::SetHipDeviceToCurrentCudaDevice raii;
    rocblas_handle roc_handle;
    MAYBE_ERROR(rocblas_create_handle(&roc_handle));
    *handle = new __redscaleDnHandle(roc_handle);
    return rocblas_status_success;
)

GPUSOLVER_EXPORT_C cusolverStatus_t
cusolverDnDestroy(cusolverDnHandle_t handle)
INLINE_BODY_STATUS(
    CudaRocmWrapper::SetHipDeviceToCurrentCudaDevice raii;
    MAYBE_ERROR(rocblas_destroy_handle(handle->handle))
    delete handle;
    return rocblas_status_success;
)

GPUSOLVER_EXPORT_C cusolverStatus_t
cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t streamId)
INLINE_BODY_STATUS(
    handle->stream = CudaRocmWrapper::HIPSynchronisedStream::getForCudaStream(streamId);
    CudaRocmWrapper::SetHipDevice setHipDevice(handle->stream->getHipDevice());
    return rocblas_set_stream(handle->handle, *handle->stream);
)

GPUSOLVER_EXPORT_C cusolverStatus_t
cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t *streamId)
INLINE_BODY_STATUS(
    *streamId = *handle->stream;
    return rocblas_status_success;
)
