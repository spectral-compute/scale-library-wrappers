#pragma once

#include "Spinlock.hpp"
#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>
#include <array>
#include <condition_variable>
#include <deque>
#include <thread>

namespace CudaRocmWrapper
{

/**
 * A signal pool with automatic signal recycling.
 *
 * The recycling is efficient if the signals are used in approximately the same order as they're allocated.
 */
class SignalPool final
{
public:
    ~SignalPool();
    explicit SignalPool();

    /**
     * Get signals from the pool, or allocate one if none exist.
     *
     * The signals are automatically recycled once the last one reaches zero.
     *
     * @tparam N The number of signals to get.
     * @param waitFor The signals are recycled when the last signal is decremented to this value.
     * @return Pointers to the signal values. The signals are initialzied to 1.
     */
    template <size_t N>
    std::array<hsa_signal_value_t *, N> getSignalGroup(int waitFor = 0)
    {
        std::array<hsa_signal_value_t *, N> result;
        getSignalGroup(result.data(), N, waitFor);
        return result;
    }

private:
    /**
     * Implementation for getSignalGroup.
     */
    void getSignalGroup(hsa_signal_value_t **dst, size_t n, int waitFor);

    /**
     * A worker thread that recycles signals from activeSignals back into signalPool as they become available.
     */
    void recycleSignals();

    Spinlock mutex;
    std::condition_variable_any cv;
    bool terminate = false;

    /**
     * A pool of signals.
     *
     * This grows to the size needed by the application. Each element is the value returned by HIP, which is a pointer
     * to the signal's atomic value. The actual HSA signal is a pointer to a struct containing this. See
     * getSignalFromValuePtr.
     */
    std::deque<hsa_signal_value_t *> signalPool; // The signal value pointers for the signals.

    /**
     * The set of signals that are in use.
     *
     * Each element is a pair: signal, and value to wait for. If the signal is nullptr, then all the prior signals are
     * recycled once the last signal is decremented to the integer in the pair.
     *
     * This queue is in (approximately in multi-threaded applications) the same order as the CUDA queue, so it's only
     * ever necessary to wait for the front of the queue. Signal groups are added atomically.
     */
    std::deque<std::pair<hsa_signal_value_t *, int>> activeSignals; // Start signal, end signal.

    /**
     * A worker thread to recycle signals.
     */
    std::thread signalRecyclingThread;
};

/**
 * Get an HSA signal from its value pointer.
 *
 * This exists because there is simply no API provided by the AMD APIs that:
 * 1. Gives access to an hsa_signal_t.
 * 2. Produces a signal that HIP can use.
 *
 * The reason why a signal allocated by hsa_amd_signal_create cannot satisfy (2) is because the HIP API returns a
 * pointer to the signal value (from hsa_amd_signal_value_pointer) rather than the signal itself, and finds the signal
 * itself by looking it up in a map it stores internally. There's no API to go in the other direction to
 * hsa_amd_signal_value_pointer (i.e: from value pointer to signal rather than from signal to value pointer).
 *
 * The HSA signal type is actually an `amd_signal_t`. The value returned by the HIP allocator is a pointer to a field
 * within that struct. This function does a reverse struct element address calculation.
 */
inline hsa_signal_t getSignalFromValuePtr(hsa_signal_value_t *ptr)
{
    return hsa_signal_t{(uint64_t)ptr - offsetof(amd_signal_t, value)};
}

} // namespace CudaRocmWrapper
