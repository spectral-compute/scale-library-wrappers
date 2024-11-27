#include "common.hpp"
#include <curand_kernel.h>

// NVIDIA's SOBOL32 changed and is no longer compliant with the reference patterns.
// WAT.
#ifdef __REDSCALE__

namespace
{

// A very crude sanity check for the curand device APIs.
// To a significant extent this is just testing that the header magic we have works. Sobol32
// seems to be the only RNG that AMD have implemented identically to nvidia's version.
unsigned int EXPECTED_KERNEL[128] = {
    0, 0, 65565, 65565, 196647, 196647, 131130, 131130, 65645, 65645, 112, 112, 131146, 131146, 196695, 196695,
    458787, 458787, 393278, 393278, 262148, 262148, 327705, 327705, 393294, 393294, 458835, 458835, 327785, 327785, 262260,
    262260, 65765, 65765, 248, 248, 131266, 131266, 196831, 196831, 136, 136, 65685, 65685, 196783, 196783, 131250, 131250, 393414,
    393414, 458971, 458971, 327905, 327905, 262396, 262396, 458923, 458923, 393398, 393398, 262284, 262284, 327825, 327825, 196671,
    196671, 131106, 131106, 24, 24, 65541, 65541, 131154, 131154, 196687, 196687, 65653, 65653, 104, 104, 262172, 262172, 327681,
    327681, 458811, 458811, 393254, 393254, 327793, 327793, 262252, 262252, 393302, 393302, 458827, 458827, 131290, 131290, 196807,
    196807, 65789, 65789, 224, 224, 196791, 196791, 131242, 131242, 144, 144, 65677, 65677, 327929, 327929, 262372, 262372,
    393438, 393438, 458947, 458947, 262292, 262292, 327817, 327817, 458931, 458931, 393390, 393390
};

__global__ void kernel(unsigned int* out, int count) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int dvs[32];
    for (int i = 0; i < 32; i++) {
        dvs[i] = i * 65565;
    }

    curandStateSobol32_t rng;
    curand_init(dvs, threadId * count, &rng);

    // A really horrible and stupid access pattern, but it ensures the data from
    // each thread is adjacent in memory, which makes it easier to inspect failures.
    // But, yes, I know this will not coalesce at all. It's a unit test, bite me.
    int start = threadId * count;
    for (int i = 0; i < count; i++) {
        out[start + i] = curand(&rng);
    }
}

struct KernelRNG {
    void operator()(unsigned int* gpuBuf) {
        kernel<<<numBlocks, blockSize>>>(gpuBuf, 1);
    }
};

TEST(CuRand_Device, Sobol32) {
    doTest<KernelRNG>(EXPECTED_KERNEL);
}

} // namespace

#endif // __REDSCALE__
