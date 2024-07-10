
#include <vector>

#include "cusparse/export.h"

// Export everything
#define GPUSPARSE_EXPORT_C extern "C" GPUSPARSE_EXPORT 

#define SPARSE_INLINE_EVERYTHING

// Override to compile uninlined functions
#define BODY(X) { X }


#include "cusparse.h"


const char* cusparseGetErrorName(cusparseStatus_t status) {
    switch (status) {
        case CUSPARSE_STATUS_SUCCESS:
            return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE:
            return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_MAPPING_ERROR:
            return "CUSPARSE_STATUS_MAPPING_ERROR";
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "CUSPARSE_STATUS_INTERNAL_ERROR";
        case CUSPARSE_STATUS_NOT_SUPPORTED:
            return "CUSPARSE_STATUS_NOT_SUPPORTED";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSPARSE_STATUS_ZERO_PIVOT:
            return "CUSPARSE_STATUS_ZERO_PIVOT";
        case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:
            return "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
    }
}

const char* cusparseGetErrorString(cusparseStatus_t status) {
    switch (status) {
        case CUSPARSE_STATUS_SUCCESS:
            return "Success";
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "Library has not been initialised";
        case CUSPARSE_STATUS_NOT_SUPPORTED:
            return "Operation not supported";
        case CUSPARSE_STATUS_INVALID_VALUE:
            return "Invalid input value";
        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "Allocation failure";
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "Architecture mismatch";
        case CUSPARSE_STATUS_MAPPING_ERROR:
            return "Mapping error";
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "Execution failed";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "Matrix type not supported";
        case CUSPARSE_STATUS_ZERO_PIVOT:
            return "Zero pivot";
        case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:
            return "Insufficient resources";
        default:
            return "Internal error";
    }
}



// GPUSPARSE_EXPORT_C cusparseStatus_t
// cusparseSpGEMM_createDescr(cusparseSpGEMMDescr_t* descr);

// GPUSPARSE_EXPORT_C cusparseStatus_t
// cusparseSpGEMM_destroyDescr(cusparseSpGEMMDescr_t descr);

// GPUSPARSE_EXPORT_C cusparseStatus_t
// cusparseSpGEMM_workEstimation(
//     cusparseHandle_t          handle,
//     cusparseOperation_t       opA,
//     cusparseOperation_t       opB,
//     const void*               alpha,
//     cusparseConstSpMatDescr_t matA,
//     cusparseConstSpMatDescr_t matB,
//     const void*               beta,
//     cusparseSpMatDescr_t      matC,
//     cudaDataType              computeType,
//     cusparseSpGEMMAlg_t       alg,
//     cusparseSpGEMMDescr_t     spgemmDescr,
//     size_t*                   bufferSize1,
//     void*                     externalBuffer1);

// GPUSPARSE_EXPORT_C cusparseStatus_t
// cusparseSpGEMM_getNumProducts(cusparseSpGEMMDescr_t spgemmDescr,
//                               int64_t*              num_prods);


// GPUSPARSE_EXPORT_C cusparseStatus_t
// cusparseSpGEMM_estimateMemory(
//     cusparseHandle_t          handle,
//     cusparseOperation_t       opA,
//     cusparseOperation_t       opB,
//     const void*               alpha,
//     cusparseConstSpMatDescr_t matA,
//     cusparseConstSpMatDescr_t matB,
//     const void*               beta,
//     cusparseSpMatDescr_t      matC,
//     cudaDataType              computeType,
//     cusparseSpGEMMAlg_t       alg,
//     cusparseSpGEMMDescr_t     spgemmDescr,
//     float                     chunk_fraction,
//     size_t*                   bufferSize3,
//     void*                     externalBuffer3,
// size_t*                   bufferSize2);

// GPUSPARSE_EXPORT_C cusparseStatus_t
// cusparseSpGEMM_compute( 
//     cusparseHandle_t          handle,
//     cusparseOperation_t       opA,
//     cusparseOperation_t       opB,
//     const void*               alpha,
//     cusparseConstSpMatDescr_t matA,
//     cusparseConstSpMatDescr_t matB,
//     const void*               beta,
//     cusparseSpMatDescr_t      matC,
//     cudaDataType              computeType,
//     cusparseSpGEMMAlg_t       alg,
//     cusparseSpGEMMDescr_t     spgemmDescr,
//     size_t*                   bufferSize2,
//     void*                     externalBuffer2);

// cusparseStatus_t cusparseSpGEMM_copy(    
//     cusparseHandle_t          handle,
//     cusparseOperation_t       opA,
//     cusparseOperation_t       opB,
//     const void*               alpha,
//     cusparseConstSpMatDescr_t matA,
//     cusparseConstSpMatDescr_t matB,
//     const void*               beta,
//     cusparseSpMatDescr_t      matC,
//     cudaDataType              computeType,
//     cusparseSpGEMMAlg_t       alg,
//     cusparseSpGEMMDescr_t     spgemmDescr) {

//     // // Copy C matrix result back to host
//     // std::vector<int> hcsr_row_ptr_C(m + 1);
//     // std::vector<int> hcsr_col_ind_C(nnz_C);
//     // std::vector<float>  hcsr_val_C(nnz_C);

//     // hipMemcpy(hcsr_row_ptr_C.data(), dcsr_row_ptr_C, sizeof(int) * (m + 1), hipMemcpyDeviceToHost);
//     // hipMemcpy(hcsr_col_ind_C.data(), dcsr_col_ind_C, sizeof(int) * nnz_C, hipMemcpyDeviceToHost);
//     // hipMemcpy(hcsr_val_C.data(), dcsr_val_C, sizeof(float) * nnz_C, hipMemcpyDeviceToHost);

// }
