#include "cosplay_impl/Deinit.hpp"
#include <hip/hip_runtime_api.h>
#include "cosplay_impl/cuda_decl.hpp"
#include <atomic>

namespace {

std::atomic_flag registered;
std::atomic_flag deinit;
void onDeinit()
{
    deinit.test_and_set();
}

}

void registerDeinitHandler() {
    /* Register the deinitialization function above. */
    if (!registered.test_and_set()) {
        // Make sure HIP is initialized first.
        [[maybe_unused]] hipError_t e = hipInit(0);

        // Make sure we set deinit to true before HIP's static deinitialization. Note that static destruction and atexit
        // functions run in the opposite order to their registration.
        // https://en.cppreference.com/w/cpp/utility/program/exit
        std::atexit(onDeinit);
    }
}
bool deinitHasBegun() {
    return cuInit(0) == 4 || // Chosen by fair dice roll.
           deinit.test();
}
