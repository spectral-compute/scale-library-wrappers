# `BLAS maths wrapper`

This directory contains RedSCALE's BLAS math wrapper which aims to provide API-compatibility with cuBLAS.

Headers in `include/blas_impl` forward supported functions to the appropriate vendor-provided BLAS
library.

Depending on the naming convention of the function we wish to forward, this is accomplished in one of
two¹ ways:

For BLAS API functions that take a letter representing type, there is the `BLAS_API` macro:

```c++
#define BLAS_API(CULETTER, ROCLETTER, NAME, CXXNAME, ...) \
    extern "C" GPUBLAS_EXPORT cublasStatus_t \
    BLAS_C_FN_NAME(CULETTER, NAME)(cublasHandle_t handle, __VA_ARGS__) { \
            CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*handle->stream}; \
            return mapReturnCode(ROC_C_FN_NAME(ROCLETTER, NAME) \
                                 (MAP(CU_TO_ROC, COMMA, handle->handle, NAME ## _ARGS))); \
    }
```

Those functions that do not take a letter representing type but are named differently in the target
library can be forwarded with the `DIRECT_BLAS_API_N` macro:

```c++
#define DIRECT_BLAS_API_N(NAME, ROCNAME, ...) \
    extern "C" GPUBLAS_EXPORT cublasStatus_t \
    cublas ## NAME(cublasHandle_t handle, __VA_ARGS__) { \
        CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*handle->stream}; \
        return mapReturnCode(rocblas ## ROCNAME (MAP(CU_TO_ROC, COMMA, handle->handle, NAME ## _ARGS))); \
    }
```

An example of both of these follows:


Here we use `BLAS_API` four times (once for each type) inside a definition which populates the `_ARGS`
required by `mapReturnCode`

```c++
/**
 * amax()
 */
#define amax_ARGS n, x, incx, result
BLAS_API(Is, is, amax, Amax, int n, const float *x, int incx, int *result)
BLAS_API(Id, id, amax, Amax, int n, const double *x, int incx, int *result)
BLAS_API(Ic, ic, amax, Amax, int n, const cuComplex *x, int incx, int *result)
BLAS_API(Iz, iz, amax, Amax, int n, const cuDoubleComplex *x, int incx, int *result)
#undef amax_ARGS
```

In the case of `DIRECT_BLAS_API_N`, only a single use of the macro is needed, with a similar definition
to populate the `_ARGS` as before

```c++
/**
 * ScalEx()
 */
#define ScalEx_ARGS n, alpha, alphaType, x, xType, incx, executionType
DIRECT_BLAS_API_N(ScalEx, _scal_ex,
    int n,
    const void *alpha,
    cudaDataType alphaType,
    void *x,
    cudaDataType xType,
    int incx,
    cudaDataType executionType
)
#undef ScalEx_ARGS
```

**NOTE:** These examples only cover the most simple cases. In the event that casting between types is
required as part of forwarding the function, the `cuToRoc` template function can be used. Additionally,
if there are differences between the C and C++ implementations of a given function, it will also be
necessary to wrap the C++ function. Examples of both can be seen for `GemmEx` in
`include/blas_impl/extension.h:3` and `src/extension.cpp:59` respectively.

---

¹ There is also a `DIRECT_BLAS_API` macro for functions which are named identically in both libraries,
  however this is currently unused.
