/* A translation unit for things using that are inconvenient to prototype, but where we need to use them with HIP.
   This exists because HIP and SCALE headers can't be mixed. */

#include "cosplay_impl/Exception.hpp"
#include <cuda.h>

namespace CudaRocmWrapper
{

void getDevicePciId(int deviceId, int &domain, int &bus, int &device)
{
    cudaDeviceProp p;
    memset(&p, 0, sizeof(p));
    if (cudaGetDeviceProperties(&p, deviceId) != cudaSuccess) {
        throw Exception("Could not get properties for device.");
    }
    domain = p.pciDomainID;
    bus = p.pciBusID;
    device = p.pciDeviceID;
}

} // namespace CudaRocmWrapper
