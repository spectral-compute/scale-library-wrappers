#include <cuda_runtime.h>
#ifdef __REDSCALE__
#include <redscale.h>
#endif // __REDSCALE__
#include <gtest/gtest.h>

namespace
{

void setDeviceFromEnvironment()
{
    const char *deviceStr = getenv("REDSCALE_TEST_SET_DEVICE");
    if (!deviceStr) {
        return;
    }
    int id = atoi(deviceStr);

    fprintf(stderr, "Setting CUDA device %i\n", id);
    if (cudaSetDevice(id) != cudaSuccess) {
        throw std::runtime_error("Could not set CUDA device " + std::to_string(id) + ".");
    }
}

void printDevice()
{
    int id = 0;
    if (cudaGetDevice(&id) != cudaSuccess) {
        throw std::runtime_error("Could not get CUDA device ID.");
    }

    cudaDeviceProp p;
    memset(&p, 0, sizeof(p));
    if (cudaGetDeviceProperties(&p, id) != cudaSuccess) {
        throw std::runtime_error("Could not get CUDA device properties.");
    }
    fprintf(stderr, "Testing device %i at %04x:%02x:%02x, which is %s\n",
            id, p.pciDomainID, p.pciBusID, p.pciDeviceID, p.name);
}

} // namespace

const char *argv0 = nullptr;

int main(int argc, char **argv)
{
    /* Capture the path to the program so we can find other stuff. */
    argv0 = argv[0];

    /* Turn on exceptions. */
#ifdef __REDSCALE__
    redscale::Exception::enable();
#endif // __REDSCALE__

    /* Select the CUDA device for the tests. */
    setDeviceFromEnvironment();
    printDevice();

    /* Run the GTest tests. */
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
