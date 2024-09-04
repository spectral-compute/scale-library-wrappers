#pragma once

#include <hsa/hsa.h>
#include <cstddef>
#include <memory>

/* Declarations of the RedSCALE API we need. We can't include all of is because that would conflict with HIP. */

struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;

enum cudaMemcpyKind
{
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

enum cudaMemoryType
{
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3
};

struct cudaPointerAttributes
{
    cudaMemoryType memoryType;
    cudaMemoryType type;
    int device;
    void *devicePointer;
    void *hostPointer;
};

extern "C"
{

int cudaDeviceSynchronize();
int cudaStreamSynchronize(cudaStream_t);
int cudaFree(void *);
int cudaFreeAsync(void *, cudaStream_t);
int cudaMalloc(void **, size_t);
int cudaMallocAsync(void **, size_t, cudaStream_t);
int cudaMemcpy(void *, const void *, size_t, cudaMemcpyKind);
int cudaMemcpy2DAsync(void *, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t);
int cudaMemcpyAsync(void *, const void *, size_t, cudaMemcpyKind kind, cudaStream_t);
int cudaMemsetAsync(void *, int, size_t, cudaStream_t);
int cudaPointerGetAttributes(cudaPointerAttributes *, const void *);

} // extern "C"

namespace redscale
{

bool isLibraryShutdownInProgress();
std::shared_ptr<void> getUserPointerFromStream(const void *, cudaStream_t, bool);
void addUserPointerToStream(const void *, std::shared_ptr<void>, cudaStream_t);

namespace Hsa
{

void enqueueStreamMemoryFence(cudaStream_t, hsa_fence_scope_t = HSA_FENCE_SCOPE_NONE, hsa_signal_t = {0},
                              hsa_signal_t = {0});
void notifyStream(cudaStream_t);

} // namespace Hsa

} // namespace redscale
