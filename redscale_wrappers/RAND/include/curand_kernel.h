#ifndef GPURAND_DEVICE_AUXILLARY_H
#define GPURAND_DEVICE_AUXILLARY_H

#include <redscale_impl/impl_defines.h>

// The device stuff is _fun_. Rocrand defines it in its headers, and most of the function names are the
// same, so we just wanna include it and do the appropriate preprocessor insanity to make that work.
// Fun times.

// Awesome hack:
#ifndef ROCRAND_H_

/**
 * \brief rocRAND function call status type
 */
typedef enum rocrand_status {
    ROCRAND_STATUS_SUCCESS = 0, ///< No errors
    ROCRAND_STATUS_VERSION_MISMATCH = 100, ///< Header file and linked library version do not match
    ROCRAND_STATUS_NOT_CREATED = 101, ///< Generator was not created using rocrand_create_generator
    ROCRAND_STATUS_ALLOCATION_FAILED = 102, ///< Memory allocation failed during execution
    ROCRAND_STATUS_TYPE_ERROR = 103, ///< Generator type is wrong
    ROCRAND_STATUS_OUT_OF_RANGE = 104, ///< Argument out of range
    ROCRAND_STATUS_LENGTH_NOT_MULTIPLE = 105, ///< Requested size is not a multiple of quasirandom generator's dimension,
    ///< or requested size is not even (see rocrand_generate_normal()),
    ///< or pointer is misaligned (see rocrand_generate_normal())
    ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106, ///< GPU does not have double precision
    ROCRAND_STATUS_LAUNCH_FAILURE = 107, ///< Kernel launch failure
    ROCRAND_STATUS_INTERNAL_ERROR = 108 ///< Internal library error
} rocrand_status;

#endif
#define ROCRAND_H_

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"

#define SILLY_STRUCT4(NAME, T) \
    union NAME { \
        struct { \
            T x; \
            T y; \
            T z; \
            T w; \
        } __REDSCALE_ALIGN_VECTOR_TYPE(T, 4); \
\
        unsigned int data[4] __REDSCALE_ALIGN_VECTOR_TYPE(T, 4); \
    };

#define SILLY_STRUCT2(NAME, T) \
    union NAME { \
        struct { \
            T x; \
            T y; \
        } __REDSCALE_ALIGN_VECTOR_TYPE(T, 2); \
\
        unsigned int data[2] __REDSCALE_ALIGN_VECTOR_TYPE(T, 2); \
    };

SILLY_STRUCT4(silly_uint4, unsigned int)
SILLY_STRUCT4(silly_ulonglong4, unsigned long long)
SILLY_STRUCT2(silly_uint2, unsigned int)
SILLY_STRUCT2(silly_ulonglong2, unsigned long long)

__device__ inline silly_uint4 silly_make_uint4(
        unsigned int x, unsigned int y, unsigned int z, unsigned int w) {
    silly_uint4 out;
    out.x = x;
    out.y = y;
    out.z = z;
    out.w = w;
    return out;
}

// Bitcast a silly_uint4 to a uint4.
__device__ inline uint4 unSillyify(silly_uint4 x) {
    static_assert(alignof(uint4) == alignof(silly_uint4), "Alignment mismatch");
    static_assert(sizeof(uint4) == sizeof(silly_uint4), "Size mismatch");
    uint4 out;
    memcpy(&out, &x, sizeof(silly_uint4));
    return out;
}

#pragma clang diagnostic pop

// Integer types
#define make_uint4 silly_make_uint4
#define ulonglong4 silly_ulonglong4
#define ulonglong2 silly_ulonglong2
#define uint4 silly_uint4
#define uint2 silly_uint2

// enum ...Error_t
#define hipError_t cudaError_t
#define hipSuccess cudaSuccess

// enum ...MemcpyKind
#define hipMemcpyDefault cudaMemcpyDefault
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice

// Functions
#define hipMemcpy cudaMemcpy
#define hipGetLastError cudaGetLastError

// Rocrand includes support for 16bit floating point random,
// while curand specifically does not,
// so we just butcher the fp16 support to achieve parity.
// We do this by using the boolean type, to avoid unexpected conversions
// from a numeric type to float.
#define __half bool
struct __silly_half2 { bool a1; bool a2;};
#define __half2 __silly_half2
#define __float2half(X) false

// Annoyingly, there is a `typedef __half half;` in the following header.
#define half __silly_half

#include <rocrand/rocrand.h>

#undef half

#include <rocrand/rocrand_kernel.h>

// fp16 parity
#undef __half
#undef __half2
#undef __float2half


#undef make_uint4
#undef uint4
#undef uint2
#undef ulonglong2
#undef ulonglong4
#undef hipMemcpy
#undef hipSuccess
#undef hipMemcpyHostToDevice
#undef hipGetLastError

#undef SILLY_STRUCT2
#undef SILLY_STRUCT4

typedef rocrand_state_scrambled_sobol64 curandStateScrambledSobol64_t;
typedef rocrand_state_scrambled_sobol32 curandStateScrambledSobol32_t;
typedef rocrand_state_philox4x32_10 curandStatePhilox4_32_10_t;
typedef rocrand_state_mrg32k3a curandStateMRG32k3a_t;
typedef rocrand_state_sobol64 curandStateSobol64_t;
typedef rocrand_state_sobol32 curandStateSobol32_t;
typedef rocrand_state_mtgp32 curandStateMtgp32_t;
typedef rocrand_state_xorwow curandStateXORWOW_t;
typedef curandStateXORWOW_t curandState_t;
typedef curandState_t curandState;

// The manual says so. Kinda weird that this even exists :D
typedef unsigned long long* curandDirectionVectors64_t;
typedef unsigned int* curandDirectionVectors32_t;


#define SIMPLE_RNG(T, OUT_T) \
    __device__ inline OUT_T curand(T* s) { \
        return rocrand(s); \
    }

SIMPLE_RNG(curandStateMtgp32_t, unsigned int)
SIMPLE_RNG(curandStateScrambledSobol32_t, unsigned int)
SIMPLE_RNG(curandStateSobol32_t, unsigned int)
SIMPLE_RNG(curandStateMRG32k3a_t, unsigned int)
SIMPLE_RNG(curandStatePhilox4_32_10_t, unsigned int)
SIMPLE_RNG(curandStateXORWOW_t, unsigned int)
SIMPLE_RNG(curandStateSobol64_t, unsigned long long)
SIMPLE_RNG(curandStateScrambledSobol64_t, unsigned long long)

#undef SIMPLE_RNG

//__host__ curandStatus_t curandMakeMTGP32Constants(const mtgp32_params_fast_t params[], mtgp32_kernel_params_t* p);
//__host__ curandStatus_t CURANDAPI curandMakeMTGP32KernelState(curandStateMtgp32_t* s, mtgp32_params_fast_t params[], mtgp32_kernel_params_t* k, int n, unsigned long long seed);



#define INIT_NO_VECS(T) \
    __device__ inline void curand_init(unsigned long long seed, unsigned long long subseq, unsigned long long offs, T* s) { \
        rocrand_init(seed, subseq, offs, s); \
    }

INIT_NO_VECS(curandStateMRG32k3a_t)
INIT_NO_VECS(curandStatePhilox4_32_10_t)
INIT_NO_VECS(curandStateXORWOW_t)

#undef INIT_NO_VECS


__device__ inline void curand_init(curandDirectionVectors32_t dirs, unsigned int offs, curandStateSobol32_t* s) {
    rocrand_init(dirs, offs, s);
}
__device__ inline void curand_init(curandDirectionVectors32_t dirs, unsigned int c, unsigned int offs, curandStateScrambledSobol32_t* s) {
    rocrand_init(dirs, c, offs, s);
}

// TODO: AMD library defect: `offs` is only 32-bit, but nvidia's version is wider. Could break programs!
__device__ inline void curand_init(curandDirectionVectors64_t dirs, unsigned long long c, unsigned long long offs, curandStateScrambledSobol64_t* s) {
    rocrand_init(dirs, c, (unsigned int) offs, s);
}
__device__ inline void curand_init(curandDirectionVectors64_t dirs, unsigned long long offs, curandStateSobol64_t* s) {
    rocrand_init(dirs, (unsigned int) offs, s);
}


//__device__ inline float curand_mtgp32_single(curandStateMtgp32_t* s);
//__device__ inline float curand_mtgp32_single_specific(curandStateMtgp32_t* s, unsigned char index, unsigned char n);
//__device__ inline unsigned int curand_mtgp32_specific(curandStateMtgp32_t* s, unsigned char index, unsigned char n);

#define NORMAL2_DELEGATE(T) \
    __device__ inline float2 curand_normal2(T* s) { \
        return rocrand_normal2(s); \
    } \
    __device__ inline double2 curand_normal2_double(T* s) { \
        return rocrand_normal_double2(s); \
    } \
    __device__ inline float2 curand_log_normal2(T* s, float mean, float stdDev) { \
        return rocrand_log_normal2(s, mean, stdDev); \
    } \
    __device__ inline double2 curand_log_normal2_double(T* s, double mean, double stdDev) { \
        return rocrand_log_normal_double2(s, mean, stdDev); \
    }

NORMAL2_DELEGATE(curandStateMRG32k3a_t)
NORMAL2_DELEGATE(curandStatePhilox4_32_10_t)
NORMAL2_DELEGATE(curandStateXORWOW_t)

//
//UNIFORM_DELEGATE(curandStateMtgp32_t)
//UNIFORM_DELEGATE(curandStatePhilox4_32_10_t)
//UNIFORM_DELEGATE(curandStateMRG32k3a_t)
//UNIFORM_DELEGATE(curandStateXORWOW_t)

#define UNIFORM_DELEGATE(T) \
    __device__ inline float curand_uniform(T* s) { \
        return rocrand_uniform(s); \
    } \
    __device__ inline float curand_uniform_double(T* s) { \
        return rocrand_uniform_double(s); \
    } \
    __device__ inline float curand_normal(T* s) { \
        return rocrand_normal(s); \
    } \
    __device__ inline double curand_normal_double(T* s) { \
        return rocrand_normal_double(s); \
    } \
    __device__ inline unsigned int curand_poisson(T* s, double x) { \
        return (unsigned int) rocrand_poisson(s, x); \
    } \
    __device__ inline float curand_log_normal(T* s, double mean, double stdDev) { \
        return rocrand_log_normal(s, mean, stdDev); \
    } \
    __device__ inline double curand_log_normal_double(T* s, double mean, double stdDev) { \
        return rocrand_log_normal_double(s, mean, stdDev); \
    }


// TODO: AMD library defect: scrambled sobol64 poisson returns i32 on nvidia, but i64 here.
UNIFORM_DELEGATE(curandStateScrambledSobol64_t)
UNIFORM_DELEGATE(curandStateSobol64_t)
UNIFORM_DELEGATE(curandStateScrambledSobol32_t)
UNIFORM_DELEGATE(curandStateSobol32_t)
UNIFORM_DELEGATE(curandStateMtgp32_t)
UNIFORM_DELEGATE(curandStatePhilox4_32_10_t)
UNIFORM_DELEGATE(curandStateMRG32k3a_t)
UNIFORM_DELEGATE(curandStateXORWOW_t)

#undef UNIFORM_DELEGATE



/// Special Philox stuff
/////////////////////////////////////////

// TODO: Oh noes! Rocm has `float2 rocrand_uniform2(s)` which they then use to implement the
//       nvidia curand_uniform2_double when hipifying. Aaaaa!
//__device__ inline double2 curand_uniform2_double(curandStatePhilox4_32_10_t* s) {
//    return rocrand_uniform2_double(s);
//}
__device__ inline float4 curand_uniform4(curandStatePhilox4_32_10_t* s) {
    return rocrand_uniform4(s);
}
__device__ inline uint4 curand4(curandStatePhilox4_32_10_t* s) {
    return unSillyify(rocrand4(s));
}
__device__ inline float4 curand_log_normal4(curandStatePhilox4_32_10_t* s, float mean, float stdDev) {
    return rocrand_log_normal4(s, mean, stdDev);
}
__device__ inline float4 curand_normal4(curandStatePhilox4_32_10_t* s) {
    return rocrand_normal4(s);
}
__device__ inline uint4 curand_poisson4(curandStatePhilox4_32_10_t* s, double x) {
    return unSillyify(rocrand_poisson4(s, x));
}

#include <redscale_impl/impl_undefines.h>

#endif //GPURAND_DEVICE_AUXILLARY_H
