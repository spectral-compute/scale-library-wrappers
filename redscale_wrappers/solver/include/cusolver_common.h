
#define MATHLIBS_SOLVER_INCLUDED


#include "rocsparse/rocsparse-auxiliary.h"
#include "rocsparse/rocsparse-functions.h"
#include "rocsparse/rocsparse-types.h"
#include <rocsparse/rocsparse.h>
#include <cuComplex.h>        // cuComplex
#include <cuda_runtime_api.h> // cudaStream_t
#include <library_types.h>    // CUDA_R_32F
#include <stdint.h>           // int64_t
#include <stdio.h>            // FILE*


#ifdef __cplusplus
#   include <cuda_fp16.h>     // __half
#endif // __cplusplus


// Version

#define CUSOLVER_VER_MAJOR 11
#define CUSOLVER_VER_MINOR 5
#define CUSOLVER_VER_PATCH 0
#define CUSOLVER_VER_BUILD 53
#define CUSOLVER_VERSION                                                     \
  (CUSOLVER_VER_MAJOR * 1000 + CUSOLVER_VER_MINOR * 100 + CUSOLVER_VER_PATCH)

#include "cusparse_impl/types.h"
#include "solver_impl/types.h"

#include "cusparse_impl/macro_hell.h"

#undef MATHLIBS_SOLVER_INCLUDED
