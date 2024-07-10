#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>

namespace
{

void testAsum(cudaStream_t stream, bool deviceScalars)
{
    /* Copy the input to the device. */
    float hostMem[] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f };
    float *deviceMem = nullptr;
    EXPECT_EQ(cudaMalloc((void **)&deviceMem, sizeof(hostMem)), cudaSuccess);
    EXPECT_EQ(cudaMemcpyAsync(deviceMem, hostMem, sizeof(hostMem), cudaMemcpyHostToDevice, stream), cudaSuccess);

    /* Run the BLAS function. */
    cublasHandle_t handle = nullptr;
    EXPECT_EQ(cublasCreate(&handle), CUBLAS_STATUS_SUCCESS);
    if (stream) {
        EXPECT_EQ(cublasSetStream(handle, stream), CUBLAS_STATUS_SUCCESS);
    }
    if (deviceScalars) {
        EXPECT_EQ(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE), CUBLAS_STATUS_SUCCESS);
    }
    EXPECT_EQ(cublasSasum(handle, 4, deviceMem + 1, 1, deviceScalars ? deviceMem : hostMem), CUBLAS_STATUS_SUCCESS);

    /* Copy back and check. */
    if (deviceScalars) {
        EXPECT_EQ(cudaMemcpyAsync(hostMem, deviceMem, sizeof(float), cudaMemcpyDeviceToHost, stream), cudaSuccess);
    }
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_LT(std::abs(hostMem[0] - 10.0f), 0.001f);

    /* Done :) */
    EXPECT_EQ(cublasDestroy(handle), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(cudaFree(deviceMem), cudaSuccess);
}

TEST(asum, default_stream_host_scalars)
{
    testAsum(nullptr, false);
}

TEST(asum, default_stream_device_scalars)
{
    testAsum(nullptr, true);
}

TEST(asum, explicit_stream_host_scalars)
{
    cudaStream_t stream = nullptr;
    EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    testAsum(stream, false);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(asum, explicit_stream_device_scalars)
{
    cudaStream_t stream = nullptr;
    EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    testAsum(stream, true);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

} // namespace
