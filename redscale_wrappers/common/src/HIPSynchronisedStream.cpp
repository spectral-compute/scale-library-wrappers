#include "cosplay_impl/HIPSynchronisedStream.hpp"
#include "cosplay_impl/cuda_decl.hpp"
#include "cosplay_impl/Exception.hpp"
#include <hip/hip_runtime_api.h>
#include <functional>
#include <map>

namespace redscale
{

void addShutdownFunction(std::function<void ()>);

} // namespace redscale

namespace CudaRocmWrapper
{

void linkHipInterception();

} // namespace CudaRocmWrapper

namespace
{

/**
 * A mapping between HIP stream and CUDA stream.
 */
class StreamMap final
{
public:
    cudaStream_t get(hipStream_t hipStream) const
    {
        std::lock_guard lock(mutex);
        return map.at(hipStream);
    }

    void add(cudaStream_t cudaStream, hipStream_t hipStream)
    {
        std::lock_guard lock(mutex);
        map[hipStream] = cudaStream;
    }

    void remove(hipStream_t cudaStream)
    {
        std::lock_guard lock(mutex);
        map.erase(cudaStream);
    }

private:
    mutable CudaRocmWrapper::Spinlock mutex;
    std::map<hipStream_t, cudaStream_t> map;
};

StreamMap &getStreamMap()
{
    static StreamMap* map = nullptr;
    if (!map) {
        map = new StreamMap;
        redscale::addShutdownFunction([&]() {
            if (map != nullptr) {
                delete map;
                map = nullptr;
            }
        });
    }
    return *map;
}

} // namespace

cudaStream_t CudaRocmWrapper::HIPSynchronisedStream::getCudaStream(hipStream_t hipStream)
{
    return getStreamMap().get(hipStream);
}

std::shared_ptr<CudaRocmWrapper::HIPSynchronisedStream>
CudaRocmWrapper::HIPSynchronisedStream::getForCudaStream(cudaStream_t cudaStream)
{
    linkHipInterception();

    static char key = 0;
    std::shared_ptr<HIPSynchronisedStream> result =
        std::static_pointer_cast<HIPSynchronisedStream>(redscale::getUserPointerFromStream(&key, cudaStream, false));
    if (!result) {
        result = std::make_shared<HIPSynchronisedStream>(cudaStream);
        redscale::addUserPointerToStream(&key, result, cudaStream);
    }
    return result;
}

CudaRocmWrapper::HIPSynchronisedStream::~HIPSynchronisedStream()
{
    getStreamMap().remove(hipStream);

    /* Destroy the stream. */
    [[maybe_unused]] hipError_t e = hipStreamSynchronize(hipStream);
    e = hipStreamDestroy(hipStream);
}

CudaRocmWrapper::HIPSynchronisedStream::HIPSynchronisedStream(cudaStream_t cudaStream) : cudaStream(cudaStream)
{
    HipException::test(hipStreamCreate(&hipStream), "Could not create HIP stream.");
    getStreamMap().add(cudaStream, hipStream);
}

hsa_signal_value_t *CudaRocmWrapper::HIPSynchronisedStream::enqueueStartOfHipItems()
{
    /* Get and prepare the signals. */
    auto [startSignal, endSignal] = signals.getSignalGroup<2>(-1);

    /* Add the signals to the CUDA stream. */
    redscale::Hsa::enqueueStreamMemoryFence(cudaStream, HSA_FENCE_SCOPE_AGENT, {0}, getSignalFromValuePtr(startSignal));
    redscale::Hsa::notifyStream(cudaStream);

    /* Make the HIP stream wait for the CUDA stream. */
    HipException::test(hipStreamWaitValue32(hipStream, startSignal, 0, hipStreamWaitValueEq),
                       "Could not enqueue signal wait onto HIP stream.");

    /* Give the end signal to the caller to give back to enqueueEndOfHipItems. */
    return endSignal;
}

void CudaRocmWrapper::HIPSynchronisedStream::enqueueEndOfHipItems(hsa_signal_value_t *signal)
{
    /* Make the HIP stream write zero to the signl to make the CUDA stream proceed. */
    HipException::test(hipStreamWriteValue32(hipStream, signal, 0, 0),
                       "Could not enqueue signal write onto HIP stream.");

    /* Make the CUDA stream wait for the write we just added. */
    // Use as completion signal too lets us use this signal to wait for -1 in the recycling thread. HIP completion sets
    // the end signal to 0, which unblocks the CUDA stream, which decrements the end signal, thus giving -1.
    redscale::Hsa::enqueueStreamMemoryFence(cudaStream, HSA_FENCE_SCOPE_AGENT, getSignalFromValuePtr(signal),
                                           getSignalFromValuePtr(signal));
    redscale::Hsa::notifyStream(cudaStream);
}
