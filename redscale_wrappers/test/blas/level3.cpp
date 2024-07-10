#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <vector>

namespace
{

void testGemm(cudaStream_t stream, bool deviceScalars)
{
    float a11 = 1.0f;
    float a12 = 2.0f;
    float a21 = 3.0f;
    float a22 = 4.0f;
    float b11 = 5.0f;
    float b12 = 6.0f;
    float b21 = 7.0f;
    float b22 = 8.0f;
    float c11 = 9.0f;
    float c12 = 10.0f;
    float c21 = 11.0f;
    float c22 = 12.0f;
    float a = 1.5;
    float b = 2.25;

    /* Copy the input to the device. */
    float hostMem[] = {
        // A
        a11, a21,
        a12, a22,

        // B
        b11, b21,
        b12, b22,

        // C
        c11, c21,
        c12, c22,

        // Alpha, Beta.
        a, b
    };
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
    EXPECT_EQ(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2,
                          (deviceScalars ? deviceMem : hostMem) + 12, deviceMem, 2, deviceMem + 4, 2,
                          (deviceScalars ? deviceMem : hostMem) + 13, deviceMem + 8, 2),
              CUBLAS_STATUS_SUCCESS);

    /* Copy back. */
    EXPECT_EQ(cudaMemcpyAsync(hostMem + 8, deviceMem + 8, sizeof(float) * 4, cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    /* Check. */
    float d11 = hostMem[8];
    float d21 = hostMem[9];
    float d12 = hostMem[10];
    float d22 = hostMem[11];
    EXPECT_LT(std::abs(a * (a11 * b11 + a12 * b21) + b * c11 - d11), 0.001f);
    EXPECT_LT(std::abs(a * (a11 * b12 + a12 * b22) + b * c12 - d12), 0.001f);
    EXPECT_LT(std::abs(a * (a21 * b11 + a22 * b21) + b * c21 - d21), 0.001f);
    EXPECT_LT(std::abs(a * (a21 * b12 + a22 * b22) + b * c22 - d22), 0.001f);

    /* Done :) */
    EXPECT_EQ(cublasDestroy(handle), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(cudaFree(deviceMem), cudaSuccess);
}

TEST(gemm, default_stream_host_scalars)
{
    testGemm(nullptr, false);
}

TEST(gemm, default_stream_device_scalars)
{
    testGemm(nullptr, true);
}

TEST(gemm, explicit_stream_host_scalars)
{
    cudaStream_t stream = nullptr;
    EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    testGemm(stream, false);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(gemm, explicit_stream_device_scalars)
{
    cudaStream_t stream = nullptr;
    EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    testGemm(stream, true);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

} // namespace
