#ifndef MATHLIBS_SPARSE_TYPES_H
#define MATHLIBS_SPARSE_TYPES_H

#include "common.h"

#include <stdint.h>           // int64_t
#include <cuComplex.h>        // cuComplex
#include <cuda_runtime_api.h> // cudaStream_t
#include "library_types.h"    // CUDA_R_32F

#ifdef __cplusplus

#include <rocsparse/rocsparse.h>

#define cusparseSpMatDescr _rocsparse_spmat_descr
#define cusparseDnMatDescr _rocsparse_dnmat_descr
#define cusparseMatDescr _rocsparse_mat_descr

#define cusparseDnVecDescr _rocsparse_dnvec_descr

#endif

//
// Opaque pointer types
//

struct cusparseContext;
typedef struct cusparseContext*       cusparseHandle_t;

struct cusparseMatDescr;
typedef struct cusparseMatDescr* cusparseMatDescr_t;

struct bsrsv2Info;
typedef struct bsrsv2Info* bsrsv2Info_t;

struct bsrsm2Info;
typedef struct bsrsm2Info* bsrsm2Info_t;

struct csric02Info;
typedef struct csric02Info* csric02Info_t;

struct bsric02Info;
typedef struct bsric02Info* bsric02Info_t;

struct csrilu02Info;
typedef struct csrilu02Info* csrilu02Info_t;

struct bsrilu02Info;
typedef struct bsrilu02Info* bsrilu02Info_t;

struct csru2csrInfo;
typedef struct csru2csrInfo* csru2csrInfo_t;

struct cusparseColorInfo;
typedef struct cusparseColorInfo* cusparseColorInfo_t;

struct pruneInfo;
typedef struct pruneInfo* pruneInfo_t;

//
// Enumerations
//

typedef enum {
    CUSPARSE_STATUS_SUCCESS                   = 0,
    CUSPARSE_STATUS_NOT_INITIALIZED           = 1,
    CUSPARSE_STATUS_ALLOC_FAILED              = 2,
    CUSPARSE_STATUS_INVALID_VALUE             = 3,
    CUSPARSE_STATUS_ARCH_MISMATCH             = 4,
    CUSPARSE_STATUS_MAPPING_ERROR             = 5,
    CUSPARSE_STATUS_EXECUTION_FAILED          = 6,
    CUSPARSE_STATUS_INTERNAL_ERROR            = 7,
    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    CUSPARSE_STATUS_ZERO_PIVOT                = 9,
    CUSPARSE_STATUS_NOT_SUPPORTED             = 10,
    CUSPARSE_STATUS_INSUFFICIENT_RESOURCES    = 11
} cusparseStatus_t;

typedef enum {
    CUSPARSE_POINTER_MODE_HOST   = 0,
    CUSPARSE_POINTER_MODE_DEVICE = 1
} cusparsePointerMode_t;

typedef enum {
    CUSPARSE_ACTION_SYMBOLIC = 0,
    CUSPARSE_ACTION_NUMERIC  = 1
} cusparseAction_t;

typedef enum {
    CUSPARSE_MATRIX_TYPE_GENERAL    = 0,
    CUSPARSE_MATRIX_TYPE_SYMMETRIC  = 1,
    CUSPARSE_MATRIX_TYPE_HERMITIAN  = 2,
    CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3
} cusparseMatrixType_t;

typedef enum {
    CUSPARSE_FILL_MODE_LOWER = 0,
    CUSPARSE_FILL_MODE_UPPER = 1
} cusparseFillMode_t;

typedef enum {
    CUSPARSE_DIAG_TYPE_NON_UNIT = 0,
    CUSPARSE_DIAG_TYPE_UNIT     = 1
} cusparseDiagType_t;

typedef enum {
    CUSPARSE_INDEX_BASE_ZERO = 0,
    CUSPARSE_INDEX_BASE_ONE  = 1
} cusparseIndexBase_t;

typedef enum {
    CUSPARSE_OPERATION_NON_TRANSPOSE       = 0,
    CUSPARSE_OPERATION_TRANSPOSE           = 1,
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
} cusparseOperation_t;

typedef enum {
    CUSPARSE_DIRECTION_ROW    = 0,
    CUSPARSE_DIRECTION_COLUMN = 1
} cusparseDirection_t;

typedef enum {
    CUSPARSE_SOLVE_POLICY_NO_LEVEL = 0,
    CUSPARSE_SOLVE_POLICY_USE_LEVEL = 1
} cusparseSolvePolicy_t;

typedef enum {
    CUSPARSE_COLOR_ALG0 = 0, // default
    CUSPARSE_COLOR_ALG1 = 1
} cusparseColorAlg_t;


//
// Logging
//

typedef void (*cusparseLoggerCallback_t)(int         logLevel,
                                         const char* functionName,
                                         const char* message);


//
// csr2csc
//

typedef enum {
    CUSPARSE_CSR2CSC_ALG_DEFAULT = 1,
    CUSPARSE_CSR2CSC_ALG1 = 1
} cusparseCsr2CscAlg_t;

//
// GENERIC APIs - Enumerators and Opaque Data Structures
//

typedef enum {
    CUSPARSE_FORMAT_CSR            = 1, ///< Compressed Sparse Row (CSR)
    CUSPARSE_FORMAT_CSC            = 2, ///< Compressed Sparse Column (CSC)
    CUSPARSE_FORMAT_COO            = 3, ///< Coordinate (COO) - Structure of Arrays
    CUSPARSE_FORMAT_BLOCKED_ELL    = 5, ///< Blocked ELL
    CUSPARSE_FORMAT_BSR            = 6, ///< Blocked Compressed Sparse Row (BSR)
    CUSPARSE_FORMAT_SLICED_ELLPACK = 7 ///< Sliced ELL
} cusparseFormat_t;

typedef enum {
    CUSPARSE_ORDER_COL = 1, ///< Column-Major Order - Matrix memory layout
    CUSPARSE_ORDER_ROW = 2  ///< Row-Major Order - Matrix memory layout
} cusparseOrder_t;

typedef enum {
    CUSPARSE_INDEX_16U = 1, ///< 16-bit unsigned integer for matrix/vector
                            ///< indices
    CUSPARSE_INDEX_32I = 2, ///< 32-bit signed integer for matrix/vector indices
    CUSPARSE_INDEX_64I = 3  ///< 64-bit signed integer for matrix/vector indices
} cusparseIndexType_t;

//------------------------------------------------------------------------------

struct cusparseSpVecDescr;
struct cusparseDnVecDescr;
struct cusparseSpMatDescr;
struct cusparseDnMatDescr;

typedef struct cusparseSpVecDescr* cusparseSpVecDescr_t;
typedef struct cusparseDnVecDescr* cusparseDnVecDescr_t;
typedef struct cusparseSpMatDescr* cusparseSpMatDescr_t;
typedef struct cusparseDnMatDescr* cusparseDnMatDescr_t;

// Delete some "const"-correctness cuda api is expected to have
#define cusparseConstSpVecDescr_t cusparseSpVecDescr_t
#define cusparseConstDnVecDescr_t cusparseDnVecDescr_t
#define cusparseConstSpMatDescr_t cusparseSpMatDescr_t
#define cusparseConstDnMatDescr_t cusparseDnMatDescr_t

 
typedef enum {
    CUSPARSE_SPMAT_FILL_MODE,
    CUSPARSE_SPMAT_DIAG_TYPE
} cusparseSpMatAttribute_t;

//
// SPARSE TO DENSE
//

typedef enum {
    CUSPARSE_SPARSETODENSE_ALG_DEFAULT = 0
} cusparseSparseToDenseAlg_t;

//
// DENSE TO SPARSE
//

typedef enum {
    CUSPARSE_DENSETOSPARSE_ALG_DEFAULT = 0
} cusparseDenseToSparseAlg_t;

//
// SPARSE MATRIX-VECTOR MULTIPLICATION
//

typedef enum {
    CUSPARSE_SPMV_ALG_DEFAULT = 0,
    CUSPARSE_SPMV_CSR_ALG1    = 2,
    CUSPARSE_SPMV_CSR_ALG2    = 3,
    CUSPARSE_SPMV_COO_ALG1    = 1,
    CUSPARSE_SPMV_COO_ALG2    = 4,
    CUSPARSE_SPMV_SELL_ALG1   = 5
} cusparseSpMVAlg_t;

//
// SPARSE TRIANGULAR VECTOR SOLVE
//

typedef enum {
    CUSPARSE_SPSV_ALG_DEFAULT = 0,
} cusparseSpSVAlg_t;

typedef enum {
    CUSPARSE_SPSV_UPDATE_GENERAL  = 0,
    CUSPARSE_SPSV_UPDATE_DIAGONAL = 1
} cusparseSpSVUpdate_t;

struct cusparseSpSVDescr;
typedef struct cusparseSpSVDescr* cusparseSpSVDescr_t;

//
// SPARSE TRIANGULAR MATRIX SOLVE
//

typedef enum {
    CUSPARSE_SPSM_ALG_DEFAULT = 0,
} cusparseSpSMAlg_t;

struct cusparseSpSMDescr;
typedef struct cusparseSpSMDescr* cusparseSpSMDescr_t;

//
// SPARSE MATRIX-MATRIX MULTIPLICATION
//

typedef enum {
    CUSPARSE_SPMM_ALG_DEFAULT      = 0,
    CUSPARSE_SPMM_COO_ALG1         = 1,
    CUSPARSE_SPMM_COO_ALG2         = 2,
    CUSPARSE_SPMM_COO_ALG3         = 3,
    CUSPARSE_SPMM_COO_ALG4         = 5,
    CUSPARSE_SPMM_CSR_ALG1         = 4,
    CUSPARSE_SPMM_CSR_ALG2         = 6,
    CUSPARSE_SPMM_CSR_ALG3         = 12,
    CUSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13
} cusparseSpMMAlg_t;

//
// SPARSE MATRIX - SPARSE MATRIX MULTIPLICATION (SpGEMM)
//

typedef enum {
    CUSPARSE_SPGEMM_DEFAULT                 = 0,
    CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC    = 1,
    CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC = 2,
    CUSPARSE_SPGEMM_ALG1                    = 3,
    CUSPARSE_SPGEMM_ALG2                    = 4,
    CUSPARSE_SPGEMM_ALG3                    = 5
} cusparseSpGEMMAlg_t;

// We use this structure for sanity checking
struct cusparseSpGEMMDescr {
    cusparseSpMatDescr_t matC;
};
typedef struct cusparseSpGEMMDescr* cusparseSpGEMMDescr_t;

//
// SAMPLED DENSE-DENSE MATRIX MULTIPLICATION
//

typedef enum {
    CUSPARSE_SDDMM_ALG_DEFAULT = 0
} cusparseSDDMMAlg_t;

//
// GENERIC APIs WITH CUSTOM OPERATORS (PREVIEW)
//

struct cusparseSpMMOpPlan;
typedef struct cusparseSpMMOpPlan*       cusparseSpMMOpPlan_t;

typedef enum {
    CUSPARSE_SPMM_OP_ALG_DEFAULT
} cusparseSpMMOpAlg_t;

#endif // MATHLIBS_SPARSE_TYPES_H
