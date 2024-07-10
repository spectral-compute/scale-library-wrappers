#include <cuComplex.h>
#include <cuda.h>
#include <rocfft/rocfft.h>

#define cuFFTFORWARD -1
#define cuFFTINVERSE 1

typedef enum cufftResult_t {
    CUFFT_SUCCESS = 0,
    CUFFT_INVALID_PLAN = 1,
    CUFFT_ALLOC_FAILED = 2,
    CUFFT_INVALID_TYPE = 3,
    CUFFT_INVALID_VALUE = 4,
    CUFFT_INTERNAL_ERROR = 5,
    CUFFT_EXEC_FAILED = 6,
    CUFFT_SETUP_FAILED = 7,
    CUFFT_INVALID_SIZE = 8,
    CUFFT_UNALIGNED_DATA = 9,
    CUFFT_INCOMPLETE_PARAMETER_LIST = 10,
    CUFFT_INVALID_DEVICE = 11,
    CUFFT_PARSE_ERROR = 12,
    CUFFT_NO_WORKSPACE = 13,
    CUFFT_NOT_IMPLEMENTED = 14,
    CUFFT_LICENSE_ERROR = 15,
    CUFFT_NOT_SUPPORTED = 16
} cufftResult;

typedef enum cufftType_t {
    CUFFT_R2C = 0x2a,  // Real to complex (interleaved)
    CUFFT_C2R = 0x2c,  // Complex (interleaved) to real
    CUFFT_C2C = 0x29,  // Complex to complex (interleaved)
    CUFFT_D2Z = 0x6a,  // Double to double-complex (interleaved)
    CUFFT_Z2D = 0x6c,  // Double-complex (interleaved) to double
    CUFFT_Z2Z = 0x69   // Double-complex to double-complex (interleaved)
} cufftType;

typedef enum cufftXtCallbackType_t {
    CUFFT_CB_LD_COMPLEX = 0x0,
    CUFFT_CB_LD_COMPLEX_DOUBLE = 0x1,
    CUFFT_CB_LD_REAL = 0x2,
    CUFFT_CB_LD_REAL_DOUBLE = 0x3,
    CUFFT_CB_ST_COMPLEX = 0x4,
    CUFFT_CB_ST_COMPLEX_DOUBLE = 0x5,
    CUFFT_CB_ST_REAL = 0x6,
    CUFFT_CB_ST_REAL_DOUBLE = 0x7,
    CUFFT_CB_UNDEFINED = 0x8
} cufftXtCallbackType;

typedef rocfft_plan_t* cufftHandle;
typedef float cufftReal;
typedef double cufftDoubleReal;
typedef cuComplex cufftComplex;
typedef cuDoubleComplex cufftDoubleComplex;


typedef cufftReal (*cufftCallbackLoadR)(void *in, size_t offset, void *caller, void *sptr);
typedef cufftComplex (*cufftCallbackLoadC)(void *in, size_t offset, void *caller, void *sptr);
typedef cufftDoubleComplex (*cufftCallbackLoadZ)(void *in, size_t offset, void *caller, void *sptr);
typedef cufftDoubleReal(*cufftCallbackLoadD)(void *in, size_t offset, void *caller, void *sptr);

typedef void (*cufftCallbackStoreR)(void *out, size_t offset, cufftReal x, void *caller, void *sptr);
typedef void (*cufftCallbackStoreD)(void *out, size_t offset, cufftDoubleReal x, void *caller, void *sptr);
typedef void (*cufftCallbackStoreC)(void *out, size_t offset, cufftComplex x, void *caller, void *sptr);
typedef void (*cufftCallbackStoreZ)(void *out, size_t offset, cufftDoubleComplex x, void *caller, void *sptr);
