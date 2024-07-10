
#include "common.h"

/// Sparse Matrix-Vector Multiplication

GPUSPARSE_EXPORT_C cusparseStatus_t
cusparseSpMV(cusparseHandle_t          handle,
             cusparseOperation_t       opA,
             const void*               alpha,
             cusparseConstSpMatDescr_t matA,  // non-const descriptor supported
             cusparseConstDnVecDescr_t vecX,  // non-const descriptor supported
             const void*               beta,
             cusparseDnVecDescr_t      vecY,
             cudaDataType              computeType,
             cusparseSpMVAlg_t         alg,
             void*                     externalBuffer) 
    INLINE_BODY_STATUS(

    // Note: rocsparse code expects a non-nullptr bufferSize if externalBuffer is nullptr 
    // (which happens if bufferSize required is 0). This is a dummy to avoid this check.
    size_t dummy = 0;

    auto status = rocsparse_spmv(cuToRoc(handle), cuToRoc(opA), alpha, 
        cuToRoc(matA), cuToRoc(vecX), beta, 
        cuToRoc(vecY), cuToRoc(computeType), cuToRoc(alg),
        rocsparse_spmv_stage_preprocess, &dummy, externalBuffer);

    if (status != rocsparse_status_success)
        return status;

    return rocsparse_spmv(cuToRoc(handle), cuToRoc(opA), alpha, 
        cuToRoc(matA), cuToRoc(vecX), beta, 
        cuToRoc(vecY), cuToRoc(computeType), cuToRoc(alg),
        rocsparse_spmv_stage_compute, &dummy, externalBuffer);
)

GPUSPARSE_EXPORT_C cusparseStatus_t
cusparseSpMV_bufferSize(cusparseHandle_t          handle,
                        cusparseOperation_t       opA,
                        const void*               alpha,
                        cusparseConstSpMatDescr_t matA,  // non-const descriptor supported
                        cusparseConstDnVecDescr_t vecX,  // non-const descriptor supported
                        const void*               beta,
                        cusparseDnVecDescr_t      vecY,
                        cudaDataType              computeType,
                        cusparseSpMVAlg_t         alg,
                        size_t*                   bufferSize)
    INLINE_BODY_STATUS( 
    return rocsparse_spmv(cuToRoc(handle), cuToRoc(opA), alpha, 
        cuToRoc(matA), cuToRoc(vecX), beta, 
        cuToRoc(vecY), cuToRoc(computeType), cuToRoc(alg),
        rocsparse_spmv_stage_buffer_size, bufferSize, nullptr);
)

//
// Sparse Triangular Vector Solve
//


// GPUSPARSE_EXPORT_C cusparseStatus_t
// cusparseSpSV_createDescr(cusparseSpSVDescr_t* descr);

// SPARSE_API_DIRECT_NOHANDLE(SpSV_destroyDescr, please_Write_me,
//  cusparseSpSVDescr_t descr);

// SPARSE_API_DIRECT(SpSV_bufferSize, please_write, 
//                         cusparseOperation_t,       opA,
//                         const void*,               alpha,
//                         cusparseConstSpMatDescr_t, matA,
//                         cusparseConstDnVecDescr_t, vecX,
//                         cusparseDnVecDescr_t,      vecY,
//                         cudaDataType,              computeType,
//                         cusparseSpSVAlg_t,         alg,
//                         cusparseSpSVDescr_t,       spsvDescr,
//                         size_t*,                   bufferSize);

// SPARSE_API_DIRECT(SpSV_analysis, please_write,   
//                       cusparseOperation_t,       opA,
//                       const void*,               alpha,
//                       cusparseConstSpMatDescr_t, matA,
//                       cusparseConstDnVecDescr_t, vecX,
//                       cusparseDnVecDescr_t,      vecY,
//                       cudaDataType,              computeType,
//                       cusparseSpSVAlg_t,         alg,
//                       cusparseSpSVDescr_t,       spsvDescr,
//                       void*,                     externalBuffer);

// SPARSE_API_DIRECT(SpSV_solve, please_write,  
//                    cusparseOperation_t,       opA,
//                    const void*,               alpha,
//                    cusparseConstSpMatDescr_t, matA,
//                    cusparseConstDnVecDescr_t, vecX,
//                    cusparseDnVecDescr_t,      vecY,
//                    cudaDataType,              computeType,
//                    cusparseSpSVAlg_t,         alg,
//                    cusparseSpSVDescr_t,       spsvDescr);

// SPARSE_API_DIRECT(SpSV_updateMatrix, please_write,   
//                           cusparseSpSVDescr_t,   spsvDescr,
//                           void*,                 newValues,
//                           cusparseSpSVUpdate_t,  updatePart);

/// Sparse Triangular Matrix Solve

// GPUSPARSE_EXPORT_C cusparseStatus_t
// cusparseSpSM_createDescr(cusparseSpSMDescr_t* descr);

// SPARSE_API_DIRECT_NOHANDLE(SpSM_destroyDescr, please_Write_me,
//  cusparseSpSMDescr_t descr);

// SPARSE_API_DIRECT(SpSM_bufferSize, please_write, 
//                         cusparseOperation_t,       opA,
//                         cusparseOperation_t,       opB,
//                         const void*,               alpha,
//                         cusparseConstSpMatDescr_t, matA,
//                         cusparseConstDnMatDescr_t, matB,
//                         cusparseDnMatDescr_t,      matC,
//                         cudaDataType,              computeType,
//                         cusparseSpSMAlg_t,         alg,
//                         cusparseSpSMDescr_t,       spsmDescr,
//                         size_t*,                   bufferSize);

// SPARSE_API_DIRECT(SpSM_analysis, please_write,   
//                       cusparseOperation_t,       opA,
//                       cusparseOperation_t,       opB,
//                       const void*,               alpha,
//                       cusparseConstSpMatDescr_t, matA,
//                       cusparseConstDnMatDescr_t, matB,
//                       cusparseDnMatDescr_t,      matC,
//                       cudaDataType,              computeType,
//                       cusparseSpSMAlg_t,         alg,
//                       cusparseSpSMDescr_t,       spsmDescr,
//                       void*,                     externalBuffer);

// SPARSE_API_DIRECT(SpSM_solve, please_write,  
//                    cusparseOperation_t,       opA,
//                    cusparseOperation_t,       opB,
//                    const void*,               alpha,
//                    cusparseConstSpMatDescr_t, matA,
//                    cusparseConstDnMatDescr_t, matB,
//                    cusparseDnMatDescr_t,      matC,
//                    cudaDataType,              computeType,
//                    cusparseSpSMAlg_t,         alg,
//                    cusparseSpSMDescr_t,       spsmDescr);

