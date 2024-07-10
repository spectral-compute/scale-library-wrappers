#ifndef GPUBLAS_AUXILIARY_H_
#define GPUBLAS_AUXILIARY_H_

#include "blas_types.h"
#include "cublas/export.h"


#ifdef __cplusplus
extern "C" {
#endif

/** Create a library handle. **/
GPUBLAS_EXPORT cublasStatus_t
cublasCreate(cublasHandle_t *handle);

/** Destroy a library handle. **/
GPUBLAS_EXPORT cublasStatus_t
cublasDestroy(cublasHandle_t handle);

/** Get the version number of the library. The handle parameter is not used and may be null. **/
GPUBLAS_EXPORT cublasStatus_t
cublasGetVersion(cublasHandle_t handle, int* version);

/**
 * Get a property from the library. This is basically just a more irritating way of querying version.
 */
GPUBLAS_EXPORT cublasStatus_t
cublasGetProperty(libraryPropertyType type, int* value);

/** Set the CUDA stream used by a given BLAS handle. By default, the default stream is used. **/
GPUBLAS_EXPORT cublasStatus_t
cublasSetStream(cublasHandle_t handle, cudaStream_t stream);

/** Get the stream a handle is using for its kernel launches. **/
GPUBLAS_EXPORT cublasStatus_t
cublasGetStream(cublasHandle_t handle, cudaStream_t *stream);


/**
 * Enable or disable argument validation.
 *
 * If enabled (the default) cublas will yield errors in response to invalid inputs such as null pointers, pointers to
 * the wrong device, negative sizes, etc.
 *
 * The checks done by cublas are somewhat more thorough than done by cublas, and they include checking that pointers
 * refer to the correct device (or the host). This make sdebugging easier, but some of these checks are relatively
 * expensive, meaning the process can take nearly a millisecond for some BLAS functions due to how obnoxiously slow
 * querying pointer information from the NVIDIA® CUDA® runtime is.
 *
 * Low-latency applications will probably want to disable argument validation for non-debug builds, as a result.
 *
 * Passing invalid arguments to the library when validation is turned off will probably crash your program.
 **/
GPUBLAS_EXPORT cublasStatus_t
cublasSetValidationEnabled(cublasHandle_t handle, cublasValidationMode_t v);

/** Determine if argument validation is enabled. */
GPUBLAS_EXPORT cublasStatus_t
cublasGetValidationEnabled(cublasHandle_t handle, cublasValidationMode_t* v);


/** Set the atomics mode. That does nothing, but we keep the handle field to avoid breaking user code. **/
GPUBLAS_EXPORT cublasStatus_t
cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode);

/** Read the atomics mode previously stored in the handle. **/
GPUBLAS_EXPORT cublasStatus_t
cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t* mode);


/** Set the pointer mode. **/
GPUBLAS_EXPORT cublasStatus_t
cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode);

/** Read the pointer mode previously stored in the handle. **/
GPUBLAS_EXPORT cublasStatus_t
cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t* mode);


/** Set the math mode. **/
GPUBLAS_EXPORT cublasStatus_t
cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);

/** Read the math mode previously stored in the handle. **/
GPUBLAS_EXPORT cublasStatus_t
cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode);


/** Copy a vector from host to device **/
GPUBLAS_EXPORT cublasStatus_t
cublasSetVectorAsync(int n, int elem_size, const void *x, int incx, void *y, int incy, cudaStream_t stream);

GPUBLAS_EXPORT cublasStatus_t
cublasSetVector(int n, int elem_size, const void *x, int incx, void *y, int incy);


/** Copy a vector from device to host. **/
GPUBLAS_EXPORT cublasStatus_t
cublasGetVectorAsync(int n, int elem_size, const void *x, int incx, void *y, int incy, cudaStream_t stream);
GPUBLAS_EXPORT cublasStatus_t
cublasGetVector(int n, int elem_size, const void *x, int incx, void *y, int incy);


/** Copy a matrix from host to device. **/
GPUBLAS_EXPORT cublasStatus_t
cublasSetMatrixAsync(int rows, int cols,
                       int elem_size,
                       const void *a, int lda,
                       void *b, int ldb, cudaStream_t stream);

GPUBLAS_EXPORT cublasStatus_t
cublasSetMatrix(int rows, int cols,
                  int elem_size,
                  const void *a, int lda,
                  void *b, int ldb);


/** Copy a matrix from device to host. **/
GPUBLAS_EXPORT cublasStatus_t
cublasGetMatrixAsync(int rows, int cols,
                       int elem_size,
                       const void *a, int lda,
                       void *b, int ldb, cudaStream_t stream);

GPUBLAS_EXPORT cublasStatus_t
cublasGetMatrix(int rows, int cols,
                  int elem_size,
                  const void *a, int lda,
                  void *b, int ldb);


GPUBLAS_EXPORT cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void* buf, size_t sz);
GPUBLAS_EXPORT cublasStatus_t cublasSetSmCountTarget(cublasHandle_t, int);
GPUBLAS_EXPORT cublasStatus_t cublasGetSmCountTarget(cublasHandle_t, int *tgt);

GPUBLAS_EXPORT const char* cublasGetStatusName(cublasStatus_t s);

GPUBLAS_EXPORT cublasStatus_t cublasLoggerConfigure(int active, int stdOut, int stdErr, const char* fileName);

/** Convert a status code to a string, usually for printing. **/
GPUBLAS_EXPORT const char* cublasGetErrorString(cublasStatus_t status);
GPUBLAS_EXPORT const char* cublasGetStatusString(cublasStatus_t status);


#ifdef __cplusplus
}
#endif

#endif  /* GPUBLAS_AUXILIARY_H_ */
