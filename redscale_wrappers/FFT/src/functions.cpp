#include "cufft.h"

extern "C" {

cufftResult cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch) {
    (void) plan;
    (void) nx;
    (void) type;
    (void) batch;
    return CUFFT_NOT_IMPLEMENTED;
}

} // extern "C"
