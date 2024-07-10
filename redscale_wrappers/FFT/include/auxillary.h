#include <library_types.h>

cufftResult
cufftGetVersion(int *version);

cufftResult
cufftGetProperty(libraryPropertyType type, int *value);

cufftResult
cufftSetStream(cufftHandle plan, cudaStream_t stream);
