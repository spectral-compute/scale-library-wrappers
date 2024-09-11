#include "cuda.h"
#include "cublas.h"
#include <cuComplex.h>
#include "blas_impl/blas_functions.h"
#include "blas_impl/blas_auxiliary.h"
#include "blas_impl/shared.hpp"


static cublasHandle* theHandle;
static cublasStatus theError = CUBLAS_STATUS_SUCCESS;

cublasStatus cublasInit() {
    theHandle = new cublasHandle;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus cublasShutdown() {
    delete theHandle;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus cublasGetVersion(int* version) {
    *version = CUBLAS_VERSION;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus cublasSetKernelStream(cudaStream_t stream) {
    return cublasSetStream(theHandle, stream);
}

cublasStatus cublasGetError() {
    auto tmp = theError;
    theError = CUBLAS_STATUS_SUCCESS;
    return tmp;
}

cublasStatus cublasAlloc(int n, int elemSize, void** devicePtr) {
    auto e = cudaMalloc(devicePtr, n * elemSize);
    return e == 0 ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_ALLOC_FAILED;
}
cublasStatus cublasFree(void* devicePtr) {
    return (cublasStatus) cudaFree(devicePtr);
}

static cublasOperation_t mapTransChar(char c) {
    switch (c) {
        case 't':
        case 'T':
            return CUBLAS_OP_T;
        case 'c':
        case 'C':
            return CUBLAS_OP_C;
        case 'n':
        case 'N':
            return CUBLAS_OP_N;
        default:
            return (cublasOperation_t) 42; // Will lead to an error from the v2 function.

    }
}
static cublasDiagType_t mapDiagChar(char c) {
    switch (c) {
        case 'n':
        case 'N':
            return CUBLAS_DIAG_NON_UNIT;
        case 'u':
        case 'U':
            return CUBLAS_DIAG_UNIT;
        default:
            return (cublasDiagType_t) 42; // Will lead to an error from the v2 function.

    }
}
static cublasFillMode_t mapFillModeChar(char c) {
    switch (c) {
        case 'l':
        case 'L':
            return CUBLAS_FILL_MODE_LOWER;
        case 'u':
        case 'U':
            return CUBLAS_FILL_MODE_UPPER;
        case 'f':
        case 'F':
            return CUBLAS_FILL_MODE_FULL;
        default:
            return (cublasFillMode_t) 42; // Will lead to an error from the v2 function.
    }
}
static cublasSideMode_t mapSideMode(char c) {
    switch (c) {
        case 'l':
        case 'L':
            return CUBLAS_SIDE_LEFT;
        case 'r':
        case 'R':
            return CUBLAS_SIDE_RIGHT;
        case 'b':
        case 'B':
            return CUBLAS_SIDE_BOTH;
        default:
            return (cublasSideMode_t) 42; // Will lead to an error from the v2 function.
    }
}

#define LEGACY_BLAS_API(LETTER, NAME, ...) \
    void LEGACY_BLAS_C_FN_NAME(cublas, LETTER, NAME)(__VA_ARGS__) { \
        theError = BLAS_C_FN_NAME(LETTER, NAME)(theHandle, NAME ## _ARGS); \
    }

#define LEGACY_BLAS_API_RET(RTYPE, LETTER, NAME, ...) \
    RTYPE LEGACY_BLAS_C_FN_NAME(cublas, LETTER, NAME)(__VA_ARGS__) { \
        RTYPE output; \
        theError = BLAS_C_FN_NAME(LETTER, NAME)(theHandle, NAME ## _ARGS, &output); \
        return output; \
    }

#include "blas_impl/legacy.h"
