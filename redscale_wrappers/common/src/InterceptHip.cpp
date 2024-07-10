/**
 * @file Intercept HIP APIs.
 *
 * HIP and RedScale have their own separate memory tracking systems. This means that when the ROCm libraries call
 * hipMemcpy with a device pointer allocated by RedScale, the pointer is not recognized as a device pointer. To solve
 * this, we intercept some of the HIP APIs to call RedScale APIs where necessary.
 */

#include "cosplay_impl/cuda_decl.hpp"
#include "cosplay_impl/HIPSynchronisedStream.hpp"
#include "cosplay_impl/SignalPool.hpp"
#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <functional>

namespace
{
/**
 * Get the next implementation of a given symbol from the linked DSOs.
 *
 * @tparam Symbol The symbol to get the next symbol for. This serves both to give the type of the (function) symbol, and
 *                serves as a template key so that the static variable it contains is unique for each function even if
 *                they have the same return and argument types.
 * @param name The name of the next symbol.
 */
template <auto Symbol>
decltype(Symbol) getNextSymbol(const char *name)
{
    static decltype(Symbol) symbol = nullptr;
    if (symbol) {
        return symbol;
    }

    symbol = (decltype(Symbol))dlsym(RTLD_NEXT, name);
    if (!symbol) {
        throw nullptr;
    }
    return symbol;
}

/**
 * Use CUDA to implement a HIP API that works on a stream.
 *
 * @param fn The function to run to implement the API call. Its argument is the CUDA stream to use.
 * @param hipStream The HIP stream onto which the copy was originally enqueued.
 * @param isAsync Whether the copy is from hipMemcpyAsync or hipMemcpy.
 * @return The return value for the HIP API.
 */
hipError_t doCudaStreamApi(std::function<void (cudaStream_t)> fn, hipStream_t hipStream, bool isAsync)
{
    try {
        /* Get the CUDA stream for the HIP stream. */
        cudaStream_t cudaStream = hipStream ? CudaRocmWrapper::HIPSynchronisedStream::getCudaStream(hipStream) :
                                              nullptr;

        /* Get the synchronization signals. */
        static CudaRocmWrapper::SignalPool signals;
        auto [startSignal, endSignal] = signals.getSignalGroup<2>(-1);

        /* Add the synchronization to the HIP stream. */
        // Make the CUDA stream go.
        if (hipStreamWriteValue32(hipStream, startSignal, 0, 0) != hipSuccess) {
            return hipErrorUnknown;
        }

        // Wait for the CUDA stream to finish.
        if (hipStreamWaitValue32(hipStream, endSignal, 0, hipStreamWaitValueEq) != hipSuccess) {
            return hipErrorUnknown;
        }

        // Recycle the signals. This has to be the 64-bit version because hipStreamWriteValue32 does not sign extend.
        if (hipStreamWriteValue64(hipStream, endSignal, -1, 0) != hipSuccess) {
            return hipErrorUnknown;
        }

        /* Add the synchronization and API call to the CUDA stream. */
        // Wait for the HIP stream.
        redscale::Hsa::enqueueStreamMemoryFence(cudaStream, HSA_FENCE_SCOPE_AGENT,
                                               CudaRocmWrapper::getSignalFromValuePtr(startSignal));

        // API call.
        fn(cudaStream);

        // Make the HIP stream start up again.
        redscale::Hsa::enqueueStreamMemoryFence(cudaStream, HSA_FENCE_SCOPE_AGENT, {0},
                                               CudaRocmWrapper::getSignalFromValuePtr(endSignal));

        // Make sure the stream knows it's got new fences.
        redscale::Hsa::notifyStream(cudaStream);

        /* Wait for the CUDA stream if we're implementing hipMemcpy rather than hipMemcpyAsync. For example, if the copy
           includes the host, the host address might go out of scope before the copy happens if we don't wait. */
        if (!isAsync && cudaStreamSynchronize(cudaStream) != 0) {
            return hipErrorUnknown;
        }

        /* Done :) */
        return hipSuccess;
    }
    catch (...) {
        return hipErrorUnknown;
    }
}

} // namespace

hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t *, int)
{
    try {
        throw nullptr;
    }
    catch (...) {
        return hipErrorUnknown;
    }
}

hipError_t hipDeviceSynchronize()
{
    try {
        if (cudaDeviceSynchronize() != 0) {
            return hipErrorUnknown;
        }
        return getNextSymbol<hipDeviceSynchronize>("hipDeviceSynchronize")();
    }
    catch (...) {
        return hipErrorUnknown;
    }
}

hipError_t hipFree(void *ptr)
{
    try {
        return (cudaFree(ptr) == 0) ? hipSuccess : hipErrorUnknown;
    }
    catch (...) {
        return hipErrorUnknown;
    }
}

hipError_t hipFreeAsync(void *dev_ptr, hipStream_t stream)
{
    return doCudaStreamApi([=](cudaStream_t cudaStream) {
        if (cudaFreeAsync(dev_ptr, cudaStream) != 0) {
            throw nullptr;
        }
    }, stream, true);
}

hipError_t hipMalloc(void **ptr, size_t size)
{
    try {
        return (cudaMalloc(ptr, size) == 0) ? hipSuccess : hipErrorUnknown;
    }
    catch (...) {
        return hipErrorUnknown;
    }
}

hipError_t hipMallocAsync(void **dev_ptr, size_t size, hipStream_t stream)
{
    return doCudaStreamApi([=](cudaStream_t cudaStream) {
        if (cudaMallocAsync(dev_ptr, size, cudaStream) != 0) {
            throw nullptr;
        }
    }, stream, true);
}

hipError_t hipMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
                            hipMemcpyKind kind, hipStream_t stream)
{
    return doCudaStreamApi([=](cudaStream_t cudaStream) {
        if (cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, (cudaMemcpyKind)kind, cudaStream) != 0)
        {
            throw nullptr;
        }
    }, stream, true);
}

hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind)
{
    return doCudaStreamApi([=](cudaStream_t) {
        if (cudaMemcpy(dst, src, sizeBytes, (cudaMemcpyKind)kind) != 0) {
            throw nullptr;
        }
    }, nullptr, false);
}

hipError_t hipMemcpyAsync(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream)
{
    return doCudaStreamApi([=](cudaStream_t cudaStream) {
        if (cudaMemcpyAsync(dst, src, sizeBytes, (cudaMemcpyKind)kind, cudaStream) != 0) {
            throw nullptr;
        }
    }, stream, true);
}

hipError_t hipMemPoolTrimTo(hipMemPool_t, size_t)
{
    try {
        throw nullptr;
    }
    catch (...) {
        return hipErrorUnknown;
    }
}

hipError_t hipMemsetAsync(void *dst, int value, size_t sizeBytes, hipStream_t stream)
{
    return doCudaStreamApi([=](cudaStream_t cudaStream) {
        if (cudaMemsetAsync(dst, value, sizeBytes, cudaStream) != 0) {
            throw nullptr;
        }
    }, stream, true);
}

hipError_t hipPointerGetAttributes(hipPointerAttribute_t *hipAttributes, const void *ptr)
{
    try {
        cudaPointerAttributes cudaAttributes;
        if (cudaPointerGetAttributes(&cudaAttributes, ptr) != 0) {
            return hipErrorUnknown;
        }

        hipAttributes->type = hipMemoryTypeHost;
        switch (cudaAttributes.type) {
            case cudaMemoryTypeUnregistered: hipAttributes->type = hipMemoryTypeHost ; break;
            case cudaMemoryTypeHost: hipAttributes->type = hipMemoryTypeHost ; break;
            case cudaMemoryTypeDevice: hipAttributes->type = hipMemoryTypeDevice; break;
            case cudaMemoryTypeManaged: hipAttributes->type = hipMemoryTypeManaged; break;
        }
        hipAttributes->device = cudaAttributes.device;
        hipAttributes->devicePointer = cudaAttributes.devicePointer;
        hipAttributes->hostPointer = cudaAttributes.hostPointer;
        hipAttributes->isManaged = false;
        hipAttributes->allocationFlags = 0;
        return hipSuccess;
    }
    catch (...) {
        return hipErrorUnknown;
    }
}

namespace CudaRocmWrapper
{

/**
 * Make sure the HIP interception is linked.
 */
void linkHipInterception()
{
    // This function actually does nothing. It exists so that the .text section is referenced, and thus the interception
    // gets dynamically linked.
}

} // namespace CudaRocmWrapper
