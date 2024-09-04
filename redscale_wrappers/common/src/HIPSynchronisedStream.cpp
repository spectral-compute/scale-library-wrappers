#include "cosplay_impl/HIPSynchronisedStream.hpp"
#include "cosplay_impl/cuda_decl.hpp"
#include "cosplay_impl/Exception.hpp"
#include <hip/hip_runtime_api.h>
#include <functional>
#include <map>

using namespace redscale;

extern "C" int cudaGetDevice(int *);

namespace redscale
{

cudaStream_t desugarStream(cudaStream_t stream);
int getStreamDevice(cudaStream_t stream);
void addShutdownFunction(std::function<void ()>);

} // namespace redscale

namespace CudaRocmWrapper
{

void linkHipInterception();
void getDevicePciId(int deviceId, int &domain, int &bus, int &device);

} // namespace CudaRocmWrapper

namespace
{

/**
 * Get the HIP device corresponding to a given CUDA device.
 */
int getHipDeviceForCudaDevice(int cudaDevice)
{
    /* Figure out the PCI ID that we can use to identify the device in HIP. */
    int domain = 0;
    int bus = 0;
    int device = 0;
    CudaRocmWrapper::getDevicePciId(cudaDevice, domain, bus, device);

    /* Search the HIP devices looking for a match. */
    // Figure out how many HIP devices exist.
    int numHipDevices = 0;
    CudaRocmWrapper::HipException::test(hipGetDeviceCount(&numHipDevices), "Could not get number of HIP devices.");

    // Iterate the devices, testing each PCI address.
    for (int hipDevice = 0; hipDevice < numHipDevices; hipDevice++) {
        hipDeviceProp_t p;
        memset(&p, 0, sizeof(p));
        CudaRocmWrapper::HipException::test(hipGetDeviceProperties(&p, hipDevice),
                                            "Could not get properties for HIP device.");
        if (p.pciDeviceID == device && p.pciBusID == bus && p.pciDomainID == domain) {
            return hipDevice;
        }
    }

    /* Could not find a corresponding device. */
    throw CudaRocmWrapper::Exception("Could not find HIP device for CUDA device " + std::to_string(cudaDevice) + ".");
}

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
    cudaStream = desugarStream(cudaStream);
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

CudaRocmWrapper::HIPSynchronisedStream::HIPSynchronisedStream(cudaStream_t cudaStream) :
    cudaStream(desugarStream(cudaStream))
{
    /* Make sure we restore the HIP device selection to match the CUDA device selection when we're done. */
    // Get the current CUDA device.
    int currentCudaDevice = 0;
    if (cudaGetDevice(&currentCudaDevice) != 0) {
        throw Exception("Could not get current CUDA device.");
    }

    // Get the corresponding HIP device.
    int currentHipDevice = getHipDeviceForCudaDevice(currentCudaDevice);

    // An RAII object to restore it after we're done.
    auto restoreHipDevice = std::shared_ptr<void>(nullptr, [=](void *) {
        HipException::test(hipSetDevice(currentHipDevice), "Could not restore HIP device.");
    });

    /* Set the HIP device to the one we want to create the stream on. */
    HipException::test(hipSetDevice(getHipDeviceForCudaDevice(redscale::getStreamDevice(this->cudaStream))),
                       "Could not set HIP device.");

    /* Add a new stream to the map. */
    HipException::test(hipStreamCreate(&hipStream), "Could not create HIP stream.");
    getStreamMap().add(this->cudaStream, hipStream);
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
