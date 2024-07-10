
#include "common.h"
#include "cusparse.h"

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
