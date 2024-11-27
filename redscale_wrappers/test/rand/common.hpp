#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace
{

constexpr int blockSize = 64;
constexpr int numBlocks = 2;

template <typename RNG>
void doTest(unsigned int *expected)
{
    unsigned int* gpuBuf;
    unsigned int* hostOut;

    int n = blockSize * numBlocks;
    size_t size = sizeof(unsigned int) * n;
    EXPECT_EQ(cudaMalloc(&gpuBuf, size), cudaSuccess);

    RNG{}(gpuBuf);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_EQ(cudaMallocHost((void**) &hostOut, size), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(hostOut, gpuBuf, size, cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < n; i++) {
        EXPECT_EQ(expected[i], hostOut[i]);
    }

    EXPECT_EQ(cudaFree(gpuBuf), cudaSuccess);
    EXPECT_EQ(cudaFreeHost(hostOut), cudaSuccess);
}

} // namespace
