#include "cosplay_impl/SignalPool.hpp"
#include "cosplay_impl/Exception.hpp"
#include "cosplay_impl/cuda_decl.hpp"
#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <cassert>
#include <vector>

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

    // If we're in global de-init, don't bother hipFree-ing anything.
    // HIP might have already been de-initialised (in which case this
    // would crash),
    if (!redscale::isLibraryShutdownInProgress()) {
        /* Destroy the signal. */
        static auto hipFree = (hipError_t (*)(void *)) dlsym(RTLD_NEXT, "hipFree");
        for (hsa_signal_value_t *signal: signalPool) {
            [[maybe_unused]] hipError_t e = hipFree(signal);
        }
    }
}

CudaRocmWrapper::SignalPool::SignalPool() : signalRecyclingThread([this]() { recycleSignals(); })
{
}

void CudaRocmWrapper::SignalPool::getSignalGroup(hsa_signal_value_t **dst, size_t n, int waitFor)
{
    size_t i = 0;

    /* Try and get signals that we've already created. */
    {
        std::unique_lock lock(mutex);
        for (; i < n && !signalPool.empty(); i++) {
            dst[i] = signalPool.back();
            signalPool.pop_back();
        }
    }

    /* Create new signals if necessary. */
    // The API for doing this is not well documented, and also seems to work backwards. Internally it calls
    // hsa_amd_signal_create, and then the value HIP's malloc returns is from hsa_amd_signal_value_pointer. The library
    // then looks up that pointer to find the original signal.
    for (; i < n; i++) {
        HipException::test(hipExtMallocWithFlags((void **)(dst + i), 8, hipMallocSignalMemory),
                           "Could not create HIP signal.");
    }

    /* Add the signals to the active signal set. */
    {
        std::lock_guard lock(mutex);
        for (i = 0; i < n; i++) {
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
            signalPool.emplace_back(signalsInGroup[i]);
        }
        numSignalsInGroup = 0;
    }
}
