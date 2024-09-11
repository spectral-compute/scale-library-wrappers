#include "blas_impl/blas_auxiliary.h"
#include "blas_impl/shared.hpp"

cublasStatus_t cublasCreate(cublasHandle_t *handle) {
    *handle = new cublasHandle;
    CudaRocmWrapper::SetHipDeviceToCurrentCudaDevice raii;
    return mapReturnCode(rocblas_create_handle(&(*handle)->handle));
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    CudaRocmWrapper::SetHipDeviceToCurrentCudaDevice raii;
    cublasStatus_t out = mapReturnCode(rocblas_destroy_handle(handle->handle));
    delete handle;
    return out;
}


cublasStatus_t cublasGetVersion_v2(cublasHandle_t, int* version) {
    *version = CUBLAS_VERSION;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetProperty(libraryPropertyType type, int* value) {
    switch (type) {
        case MAJOR_VERSION:
            *value = CUBLAS_VER_MAJOR;
            break;
        case MINOR_VERSION:
            *value = CUBLAS_VER_MINOR;
            break;
        case PATCH_LEVEL:
            *value = CUBLAS_VER_PATCH;
            break;
    }

    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t s) {
    handle->stream = CudaRocmWrapper::HIPSynchronisedStream::getForCudaStream(s);
    CudaRocmWrapper::SetHipDevice setHipDevice(handle->stream->getHipDevice());
    return mapReturnCode(rocblas_set_stream(*handle, *handle->stream));
}

cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* stream) {
    *stream = *handle->stream;
    return CUBLAS_STATUS_SUCCESS;
}



cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode) {
    return mapReturnCode(rocblas_set_pointer_mode(*handle, cuToRoc(mode)));
}

cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t *mode) {
    return mapReturnCode(rocblas_get_pointer_mode(*handle, reinterpret_cast<rocblas_pointer_mode *>(mode)));
}

cublasStatus_t cublasSetMathMode(cublasHandle_t, cublasMath_t) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMathMode(cublasHandle_t, cublasMath_t* mode) {
    *mode = CUBLAS_DEFAULT_MATH;
    return CUBLAS_STATUS_SUCCESS;
}


cublasStatus_t cublasSetVectorAsync(int n, int elem_size, const void *x, int incx, void *y, int incy, cudaStream_t stream) {
    std::shared_ptr<CudaRocmWrapper::HIPSynchronisedStream> s =
        CudaRocmWrapper::HIPSynchronisedStream::getForCudaStream(stream);
    CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*s};
    return mapReturnCode(rocblas_set_vector_async(
        n, elem_size, x, incx, y, incy, *s
    ));
}

cublasStatus_t cublasSetVector(int n, int elem_size, const void *x, int incx, void *y, int incy) {
    return mapReturnCode(rocblas_set_vector(
        n, elem_size, x, incx, y, incy
    ));
}

cublasStatus_t cublasGetVectorAsync(int n, int elem_size, const void *x, int incx, void *y, int incy, cudaStream_t stream) {
    std::shared_ptr<CudaRocmWrapper::HIPSynchronisedStream> s =
        CudaRocmWrapper::HIPSynchronisedStream::getForCudaStream(stream);
    CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*s};
    return mapReturnCode(rocblas_get_vector_async(
        n, elem_size, x, incx, y, incy, *s
    ));
}

cublasStatus_t cublasGetVector(int n, int elem_size, const void *x, int incx, void *y, int incy) {
    return mapReturnCode(rocblas_get_vector(
        n, elem_size, x, incx, y, incy
    ));
}

cublasStatus_t cublasSetMatrixAsync(
    int rows, int cols,
    int elem_size,
    const void *a, int lda,
    void *b, int ldb, cudaStream_t stream
) {
    std::shared_ptr<CudaRocmWrapper::HIPSynchronisedStream> s =
        CudaRocmWrapper::HIPSynchronisedStream::getForCudaStream(stream);
    CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*s};
    return mapReturnCode(rocblas_set_matrix_async(
        rows, cols, elem_size, a, lda, b, ldb, *s
    ));
}

cublasStatus_t cublasSetMatrix(
    int rows, int cols,
    int elem_size,
    const void *a, int lda,
    void *b, int ldb
) {
    return mapReturnCode(rocblas_set_matrix(
        rows, cols, elem_size, a, lda, b, ldb
    ));
}

/** Copy a matrix from device to host. **/
cublasStatus_t cublasGetMatrixAsync(
    int rows, int cols,
    int elem_size,
    const void *a, int lda,
    void *b, int ldb, cudaStream_t stream
) {
    std::shared_ptr<CudaRocmWrapper::HIPSynchronisedStream> s =
        CudaRocmWrapper::HIPSynchronisedStream::getForCudaStream(stream);
    CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*s};
    return mapReturnCode(rocblas_get_matrix_async(
        rows, cols, elem_size, a, lda, b, ldb, *s
    ));
}

cublasStatus_t cublasGetMatrix(
    int rows, int cols,
    int elem_size,
    const void *a, int lda,
    void *b, int ldb
) {
    return mapReturnCode(rocblas_get_matrix(
        rows, cols, elem_size, a, lda, b, ldb
    ));
}

cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void* ptr, size_t sz) {
    return mapReturnCode(rocblas_set_workspace(*handle, ptr, sz));
}

cublasStatus_t cublasSetSmCountTarget(cublasHandle_t, int) {
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasGetSmCountTarget(cublasHandle_t, int *tgt) {
    *tgt = 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLoggerConfigure(int, int, int, const char*) {
    // TODO: Implement this in rocm. Its logging is controlled _only_ via env vars.
    return CUBLAS_STATUS_SUCCESS;
}

const char* cublasGetStatusName(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
}

const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "Success";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "Library has not been initialised";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "Operation not supported";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "Invalid input value";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "Allocation failure";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "Architecture mismatch";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "Mapping error";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "Execution failed";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "License error";
        default:
            return "Internal error";
    }
}
const char* cublasGetStatusString(cublasStatus_t status) {
    return cublasGetErrorString(status);
}
