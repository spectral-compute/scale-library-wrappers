#ifndef RAND_MATHLIBS_TYPES_H
#define RAND_MATHLIBS_TYPES_H

#include "vector_types.h"

typedef struct ihipStream_t* hipStream_t;
typedef struct rocrand_discrete_distribution_st* rocrand_discrete_distribution;
typedef rocrand_discrete_distribution curandDiscreteDistribution;
typedef rocrand_discrete_distribution curandDiscreteDistribution_t;

/* Discarding status codes is bad. */
#if defined(__cplusplus) && __cplusplus >= 201703L
#define __GPURAND_NODISCARD [[nodiscard]]
#else
#define __GPURAND_NODISCARD
#endif

/* Suppress HIP nonsense. */
#define HIP_INCLUDE_HIP_HIP_RUNTIME_H
#define HIP_INCLUDE_HIP_HIP_FP16_H
#define HIP_INCLUDE_HIP_HIP_VECTOR_TYPES_H

/* Rocrand includes support for 16bit floating point random,
 * while curand specifically does not,
 * so we just butcher the fp16 support to achieve parity. */
#define __half short
#define __float2half(X) false

/* Annoyingly, there is a `typedef __half half;` in the following header. */
#define half __silly_half

#include <rocrand/rocrand.h>

#undef half
#undef __half
#undef __float2half

/** C-Style status codes.  **/
typedef enum __GPURAND_NODISCARD {
    CURAND_STATUS_SUCCESS = 0,
    CURAND_STATUS_VERSION_MISMATCH = 100,
    CURAND_STATUS_NOT_INITIALIZED = 101,
    CURAND_STATUS_ALLOCATION_FAILED = 102,
    CURAND_STATUS_TYPE_ERROR = 103,
    CURAND_STATUS_OUT_OF_RANGE = 104,
    CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105,
    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
    CURAND_STATUS_LAUNCH_FAILURE = 201,
    CURAND_STATUS_PREEXISTING_FAILURE = 202,
    CURAND_STATUS_INITIALIZATION_FAILED = 203,
    CURAND_STATUS_ARCH_MISMATCH = 204,
    CURAND_STATUS_INTERNAL_ERROR = 999
} curandStatus_t;


__attribute__((always_inline))
inline curandStatus_t mapReturnCode(rocrand_status rocStatus) {
    switch (rocStatus) {
        case ROCRAND_STATUS_SUCCESS:
            return CURAND_STATUS_SUCCESS;
        case ROCRAND_STATUS_VERSION_MISMATCH:
            return CURAND_STATUS_VERSION_MISMATCH;
        case ROCRAND_STATUS_NOT_CREATED:
            return CURAND_STATUS_NOT_INITIALIZED;
        case ROCRAND_STATUS_ALLOCATION_FAILED:
            return CURAND_STATUS_ALLOCATION_FAILED;
        case ROCRAND_STATUS_TYPE_ERROR:
            return CURAND_STATUS_TYPE_ERROR;
        case ROCRAND_STATUS_OUT_OF_RANGE:
            return CURAND_STATUS_OUT_OF_RANGE;
        case ROCRAND_STATUS_LENGTH_NOT_MULTIPLE:
            return CURAND_STATUS_LENGTH_NOT_MULTIPLE;
        case ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return CURAND_STATUS_DOUBLE_PRECISION_REQUIRED;
        case ROCRAND_STATUS_LAUNCH_FAILURE:
            return CURAND_STATUS_LAUNCH_FAILURE;
        case ROCRAND_STATUS_INTERNAL_ERROR:
            return CURAND_STATUS_INTERNAL_ERROR;
    }
}

typedef enum curandRngType {
    CURAND_RNG_TEST = 0,
    CURAND_RNG_PSEUDO_DEFAULT = 100,
    CURAND_RNG_PSEUDO_XORWOW = 101,
    CURAND_RNG_PSEUDO_MRG32K3A = 121,
    CURAND_RNG_PSEUDO_MTGP32 = 141,
    CURAND_RNG_PSEUDO_MT19937 = 142,
    CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161,
    CURAND_RNG_QUASI_DEFAULT = 200,
    CURAND_RNG_QUASI_SOBOL32 = 201,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,
    CURAND_RNG_QUASI_SOBOL64 = 203,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204
} curandRngType_t;

typedef enum curandOrdering {
    CURAND_ORDERING_PSEUDO_BEST = 100,
    CURAND_ORDERING_PSEUDO_DEFAULT = 101,
    CURAND_ORDERING_PSEUDO_SEEDED = 102,
    CURAND_ORDERING_PSEUDO_LEGACY = 103,
    CURAND_ORDERING_PSEUDO_DYNAMIC = 104,
    CURAND_ORDERING_QUASI_DEFAULT = 201
} curandOrdering_t;


/*MAP_ENUM_CU_TO_ROC(rocrand_ordering, curandOrdering,
     (CURAND_ORDERING_PSEUDO_BEST, ROCRAND_RNG_PSEUDO_DEFAULT),
     (CURAND_ORDERING_PSEUDO_DEFAULT, ROCRAND_RNG_PSEUDO_DEFAULT),
     (CURAND_ORDERING_PSEUDO_SEEDED, ROCRAND_RNG_PSEUDO_XORWOW),
     (CURAND_ORDERING_PSEUDO_LEGACY, ROCRAND_RNG_PSEUDO_MRG32K3A),
     (CURAND_ORDERING_PSEUDO_DYNAMIC, ROCRAND_RNG_PSEUDO_MTGP32),
     (CURAND_ORDERING_QUASI_DEFAULT, ROCRAND_RNG_PSEUDO_MT19937)
)*/


typedef enum curandDirectionVectorSet {
    CURAND_DIRECTION_VECTORS_32_JOEKUO6 = 101,
    CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102,
    CURAND_DIRECTION_VECTORS_64_JOEKUO6 = 103,
    CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104
} curandDirectionVectorSet_t;


#endif /* RAND_MATHLIBS_TYPES_H */
