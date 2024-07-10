#include <gtest/gtest.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>


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

unsigned int EXPECTED_HOST[128] = {
    0, 2147483648, 3221225472, 1073741824, 1610612736, 3758096384, 2684354560, 536870912, 805306368, 2952790016,
    4026531840, 1879048192, 1342177280, 3489660928, 2415919104, 268435456, 402653184, 2550136832, 3623878656,
    1476395008, 2013265920, 4160749568, 3087007744, 939524096, 671088640, 2818572288, 3892314112, 1744830464,
    1207959552, 3355443200, 2281701376, 134217728, 201326592, 2348810240, 3422552064, 1275068416, 1811939328,
    3959422976, 2885681152, 738197504, 1006632960, 3154116608, 4227858432, 2080374784, 1543503872, 3690987520,
    2617245696, 469762048, 335544320, 2483027968, 3556769792, 1409286144, 1946157056, 4093640704, 3019898880,
    872415232, 603979776, 2751463424, 3825205248, 1677721600, 1140850688, 3288334336, 2214592512, 67108864,
    100663296, 2248146944, 3321888768, 1174405120, 1711276032, 3858759680, 2785017856, 637534208, 905969664,
    3053453312, 4127195136, 1979711488, 1442840576, 3590324224, 2516582400, 369098752, 503316480, 2650800128,
    3724541952, 1577058304, 2113929216, 4261412864, 3187671040, 1040187392, 771751936, 2919235584, 3992977408,
    1845493760, 1308622848, 3456106496, 2382364672, 234881024, 167772160, 2315255808, 3388997632, 1241513984,
    1778384896, 3925868544, 2852126720, 704643072, 973078528, 3120562176, 4194304000, 2046820352, 1509949440,
    3657433088, 2583691264, 436207616, 301989888, 2449473536, 3523215360, 1375731712, 1912602624, 4060086272,
    2986344448, 838860800, 570425344, 2717908992, 3791650816, 1644167168, 1107296256, 3254779904, 2181038080,
    33554432
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

constexpr int blockSize = 64;
constexpr int numBlocks = 2;


template<typename RNG>
void doTest(unsigned int* expected) {
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
//        printf("%u,", hostOut[i]);
        EXPECT_EQ(expected[i], hostOut[i]);
    }

    EXPECT_EQ(cudaFree(gpuBuf), cudaSuccess);
    EXPECT_EQ(cudaFreeHost(hostOut), cudaSuccess);
}

// NVIDIA's SOBOL32 changed and is no longer compliant with the reference patterns.
// WAT.
#ifdef __REDSCALE__
struct KernelRNG {
    void operator()(unsigned int* gpuBuf) {
        kernel<<<numBlocks, blockSize>>>(gpuBuf, 1);
    }
};

TEST(CuRand_Device, Sobol32) {
    doTest<KernelRNG>(EXPECTED_KERNEL);
}
#endif // __REDSCALE__


struct HostRNG {
    void operator()(unsigned int* gpuBuf) {
        curandGenerator_t gen;
        EXPECT_EQ(CURAND_STATUS_SUCCESS, curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL32));
        EXPECT_EQ(CURAND_STATUS_SUCCESS, curandGenerate(gen, gpuBuf, 128));
    }
};

TEST(CuRand_Host, Sobol32) {
    doTest<HostRNG>(EXPECTED_HOST);
}
