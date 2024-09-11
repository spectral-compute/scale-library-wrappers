#pragma once

#include "SignalPool.hpp"

struct CUstream_st;
typedef CUstream_st *cudaStream_t;

struct ihipStream_t;
typedef struct ihipStream_t* hipStream_t;

namespace CudaRocmWrapper
{

/**
 * A "cuda stream" that creates a HIP stream and synchronises with it. Things that get enqueued onto the HIP
 * stream while the associated `EnqueueHipItems` is in scope will appear to be enqueued onto the CUDA stream.
 *
 * This allows interoperability with libraries that demand HIP streams: we can provide a HIP stream that doesn't
 * break our stream semantics, with only mildly hideous overhead.
 *
 * This class belongs as part of the wrapper libraries' handles. Its constructor should be given the CUDA stream that
 * the wrapper library is given.
 */
class HIPSynchronisedStream final
{
public:
    /**
     * State object to pass between enqueueStartOfHipItems and enqueueEndOfHipItems.
     *
     * This isn't stored in the class because that would not be thread-local.
     */
    struct EnqueueState final
    {
        hsa_signal_value_t *signal = nullptr;
        int originalHipDevice = -1;
    };

    /**
     * An RAII object that makes things added to the HIP stream while it exists appear as a single item on the CUDA
     * stream.
     *
     * This class bundles everything that is added to a HIP stream together into one item on the CUDA stream. It should
     * be in scope when adding a sequence of HIP items to the HIP stream via the ROCm library APIs.
     */
    class EnqueueHipItems final
    {
    public:
        ~EnqueueHipItems()
        {
            streams.enqueueEndOfHipItems(state);
        }

        explicit EnqueueHipItems(HIPSynchronisedStream &streams) :
            streams(streams), state(streams.enqueueStartOfHipItems())
        {
        }

    private:
        HIPSynchronisedStream &streams;
        EnqueueState state;
    };

    /**
     * Get the CUDA stream that corresponds to a HIP stream created by this object.
     *
     * @param hipStream The HIP stream to look up.
     * @return The CUDA stream for which the given HIP stream was created.
     */
    static cudaStream_t getCudaStream(hipStream_t hipStream);

    /**
     * Get (and possibly create) a HIP synchronized stream for a given CUDA stream.
     *
     * @param cudaStream The CUDA stream to get a synchronization glue object for.
     */
    static std::shared_ptr<HIPSynchronisedStream> getForCudaStream(cudaStream_t cudaStream);

    /**
     * Destroy the HIP stream and associated stuff.
     */
    ~HIPSynchronisedStream();

    /**
     * Get the HIP stream.
     */
    operator hipStream_t()
    {
        return hipStream;
    }
    operator cudaStream_t()
    {
        return cudaStream;
    }

    /**
     * Get the HIP device for this stream.
     */
    int getHipDevice() const
    {
        return hipDevice;
    }

    /**
     * Enqueue the start of a sequence of HIP items as a single CUDA item.
     *
     * @return The state to give to enqueueEndOfHipItems.
     */
    EnqueueState enqueueStartOfHipItems();

    /**
     * Enqueue the end of a sequence of HIP items started with enqueueStartOfHipItems.
     *
     * @param state The value returned by enqueueStartOfHipItems.
     */
    void enqueueEndOfHipItems(EnqueueState state);

    /**
     * Create a HIP stream and manage it as synchronized to the given CUDA stream.
     *
     * Should not be called directly: use `getForCudaStream()`.
     *
     * The HIP stream is owned by this object, but this object does not take ownership of the CUDA stream.
     *
     * @param cudaStream The CUDA stream to synchronize with. By, the default stream is used.
     */
    explicit HIPSynchronisedStream(cudaStream_t cudaStream);

private:
    SignalPool signals;
    cudaStream_t cudaStream = nullptr;
    hipStream_t hipStream = nullptr;
    int hipDevice = -1;
};

/**
 * A RAII object to set the HIP device for this stream and restore it afterwards.
 */
class SetHipDevice final
{
public:
    ~SetHipDevice();
    explicit SetHipDevice(int device);

private:
    int originalHipDevice = -1;
};

/**
 * A RAII object to set the current HIP device to match the current CUDA device and restore it afterwards.
 */
class SetHipDeviceToCurrentCudaDevice final
{
public:
    SetHipDeviceToCurrentCudaDevice();

private:
    SetHipDevice setHipDevice;
};

} // CudaRocmWrapper
