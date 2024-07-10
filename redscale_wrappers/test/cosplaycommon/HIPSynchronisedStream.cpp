#include "cosplay_impl/HIPSynchronisedStream.hpp"
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <atomic>
#include <thread>

namespace redscale
{

class Stream;

} // namespace redscale

typedef redscale::Stream CUstream_st;
typedef CUstream_st *cudaStream_t;

typedef void (* cudaStreamCallback_t)(cudaStream_t, int, void *);

extern "C"
{

int cudaStreamSynchronize(cudaStream_t);
int cudaStreamAddCallback(cudaStream_t, cudaStreamCallback_t, void *, unsigned int);

} // extern "C"

namespace
{

TEST(HIPSynchronisedStream, Simple)
{
    std::shared_ptr<CudaRocmWrapper::HIPSynchronisedStream> streams =
        CudaRocmWrapper::HIPSynchronisedStream::getForCudaStream(nullptr);
    {
        CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems raii(*streams);
    }
    EXPECT_EQ(cudaStreamSynchronize(nullptr), 0);
}

TEST(HIPSynchronisedStream, Many)
{
std::shared_ptr<CudaRocmWrapper::HIPSynchronisedStream> streams =
    CudaRocmWrapper::HIPSynchronisedStream::getForCudaStream(nullptr);
    for (int i = 0; i < 100; i++){
        CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems raii(*streams);
    }
    EXPECT_EQ(cudaStreamSynchronize(nullptr), 0);
}

struct OrderTestState final
{
    template <int N, int T>
    static void load(auto, auto, void *userData)
    {
        const OrderTestState &state = *(OrderTestState *)userData;
        std::this_thread::sleep_for(std::chrono::milliseconds(T));
        EXPECT_EQ(state.value.load(), N);
    }

    template <int N, int T>
    static void store(auto, auto, void *userData)
    {
        OrderTestState &state = *(OrderTestState *)userData;
        std::this_thread::sleep_for(std::chrono::milliseconds(T));
        state.value.store(N);
    }

    std::atomic<int> value = 0;
};

TEST(HIPSynchronisedStream, Order)
{
    OrderTestState state;

    EXPECT_EQ(cudaStreamAddCallback(nullptr, OrderTestState::store<1, 250>, &state, 0), 0);

    // This tests that the CUDA stream waits for the HIP stream.
    std::shared_ptr<CudaRocmWrapper::HIPSynchronisedStream> streams =
        CudaRocmWrapper::HIPSynchronisedStream::getForCudaStream(nullptr);
    {
        CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems raii(*streams);
        EXPECT_EQ(cudaStreamAddCallback(nullptr, OrderTestState::load<1, 0>, &state, 0), 0);
        EXPECT_EQ(hipStreamAddCallback(nullptr, OrderTestState::store<2, 125>, &state, 0), hipSuccess);
    }
    EXPECT_EQ(cudaStreamSynchronize(nullptr), 0);
    EXPECT_EQ(state.value.load(), 2);
}

} // namespace
