#include "cosplay_impl/cuda_decl.hpp"
#include "cosplay_impl/Deinit.hpp"
#include "cosplay_impl/Exception.hpp"
#include "cosplay_impl/SignalPool.hpp"
#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <atomic>
#include <cassert>
#include <vector>

using namespace CudaRocmWrapper;

namespace
{

/**
 * A HIP signal allocator.
 *
 * This means we don't have to call hipExtMallocWithFlags every time we need a signal, and means we can avoid a signal
 * leak and destruction-time issues (e.g: a bug where calling hipFree deadlocked iwth hipStreamWriteValue64).
 *
 * Using a single source of signals like this works because the signals are actually on the host.
 */
class HipSignalPool final
{
public:
    static HipSignalPool &getPool()
    {
        static HipSignalPool pool;
        return pool;
    }

    // No need for a destructor, since it would only run at destructor time, when we don't know that HIP hasn't adlready
    // been destroyed.

    hsa_signal_value_t *allocate()
    {
        /* Try and return an already allocated signal. */
        {
            std::lock_guard lock(mutex);
            if (!signals.empty()) {
                hsa_signal_value_t *result = signals.back();
                signals.pop_back();
                return result;
            }
        }

        /* Create new signals if necessary. */
        // The API for doing this is not well documented, and also seems to work backwards. Internally it calls
        // hsa_amd_signal_create, and then the value HIP's malloc returns is from hsa_amd_signal_value_pointer. The library
        // then looks up that pointer to find the original signal.
        hsa_signal_value_t *result = nullptr;
        HipException::test(hipExtMallocWithFlags((void **)&result, 8, hipMallocSignalMemory),
                           "Could not create HIP signal.");
        return result;
    }

    void free(hsa_signal_value_t *signal)
    {
        std::lock_guard lock(mutex);
        signals.push_back(signal); // Rarely actually allocates.
    }

private:
    Spinlock mutex;

    /**
     * A pool of signals.
     *
     * This grows to the size needed by the application. Each element is the value returned by HIP, which is a pointer
     * to the signal's atomic value. The actual HSA signal is a pointer to a struct containing this. See
     * getSignalFromValuePtr.
     */
    std::vector<hsa_signal_value_t *> signals;
};

} // namespace

CudaRocmWrapper::SignalPool::~SignalPool()
{
    /* Wait for all the signals to be put back into the pool. */
    {
        std::lock_guard lock(mutex);
        terminate = true;
    }
    cv.notify_one();
    signalRecyclingThread.join();
    assert(activeSignals.empty());
}

CudaRocmWrapper::SignalPool::SignalPool() : signalRecyclingThread([this]() { recycleSignals(); })
{}

void CudaRocmWrapper::SignalPool::getSignalGroup(hsa_signal_value_t **dst, size_t n, int waitFor)
{
    /* Get new signals. */
    for (size_t i = 0; i < n; i++) {
        dst[i] = HipSignalPool::getPool().allocate();
    }

    /* Add the signals to the active signal set. */
    {
        std::lock_guard lock(mutex);
        for (size_t i = 0; i < n; i++) {
            __atomic_store_n(dst[i], 1, __ATOMIC_RELAXED);
            activeSignals.emplace_back(dst[i], 0);
        }
        activeSignals.emplace_back(nullptr, waitFor); // End of group.
    }

    /* Set the recycler thread waiting for the newly allocated signals. */
    cv.notify_one();
}

void CudaRocmWrapper::SignalPool::recycleSignals()
{
    std::vector<hsa_signal_value_t *> signalsInGroup(2); // A rarely reallocated dynamically sized group of signals.
    size_t numSignalsInGroup = 0;

    std::unique_lock lock(mutex);
    while (!terminate || !activeSignals.empty()) {
        /* Wait for there to be a signal to wait on. */
        if (activeSignals.empty()) {
            assert(!terminate);
            cv.wait(lock);
            continue;
        }

        /* Extract the next signal group. */
        auto [signal, waitFor] = activeSignals.front();
        activeSignals.pop_front();

        /* Append to the group if we have a signal. */
        if (signal) {
            if (signalsInGroup.size() <= numSignalsInGroup) {
                signalsInGroup.resize(signalsInGroup.size() * 2);
            }
            signalsInGroup[numSignalsInGroup++] = signal;
            continue;
        }

        /* Wait for the end signal to be completed. */
        if (numSignalsInGroup == 0) {
            continue;
        }

        lock.unlock();
        while (hsa_signal_wait_scacquire(getSignalFromValuePtr(signalsInGroup[numSignalsInGroup - 1]),
                                         HSA_SIGNAL_CONDITION_LT, waitFor + 1, UINT64_MAX,
                                         HSA_WAIT_STATE_BLOCKED) > waitFor) {}
        lock.lock();

        /* Recycle the signals. */
        for (size_t i = 0; i < numSignalsInGroup; i++) {
            HipSignalPool::getPool().free(signalsInGroup[i]);
        }
        numSignalsInGroup = 0;
    }
}
