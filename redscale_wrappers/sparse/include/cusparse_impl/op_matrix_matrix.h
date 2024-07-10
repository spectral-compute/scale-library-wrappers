
#include "common.h"

/// Sparse Matrix-Matrix Multiplication

GPUSPARSE_EXPORT_C cusparseStatus_t 
cusparseSpMM_bufferSize(cusparseHandle_t          handle,
                        cusparseOperation_t       opA,
                        cusparseOperation_t       opB,
                        const void*               alpha,
                        cusparseConstSpMatDescr_t matA,
                        cusparseConstDnMatDescr_t matB,
                        const void*               beta,
                        cusparseDnMatDescr_t      matC,
                        cudaDataType              computeType,
                        cusparseSpMMAlg_t         alg,
                        size_t*                   bufferSize) 
    INLINE_BODY_STATUS(

    // Query SpMM buffer size
    return rocsparse_spmm(handle->handle,
                   cuToRoc(opA), cuToRoc(opB),
                   &alpha, matA, matB,
                   &beta, matC,
                   cuToRoc(computeType),
                   cuToRoc(alg),
                   rocsparse_spmm_stage_buffer_size,
                   bufferSize,
                   nullptr);
)

GPUSPARSE_EXPORT_C cusparseStatus_t
cusparseSpMM_preprocess(
    cusparseHandle_t handle,    
    cusparseOperation_t       opA,
    cusparseOperation_t       opB,
    const void*               alpha,
    cusparseConstSpMatDescr_t matA,
    cusparseConstDnMatDescr_t matB,
    const void*               beta,
    cusparseDnMatDescr_t      matC,
    cudaDataType              computeType,
    cusparseSpMMAlg_t         alg,
    void*                     externalBuffer) 
    INLINE_BODY_STATUS(

    return rocsparse_spmm(handle->handle,
                   cuToRoc(opA), cuToRoc(opB),
                   &alpha, matA, matB,
                   &beta, matC,
                   cuToRoc(computeType),
                   cuToRoc(alg),
                   rocsparse_spmm_stage_preprocess,
                   nullptr,
                   externalBuffer);
)

GPUSPARSE_EXPORT_C cusparseStatus_t 
cusparseSpMM(
    cusparseHandle_t handle,    
    cusparseOperation_t       opA,
    cusparseOperation_t       opB,
    const void*               alpha,
    cusparseConstSpMatDescr_t matA,
    cusparseConstDnMatDescr_t matB,
    const void*               beta,
    cusparseDnMatDescr_t      matC,
    cudaDataType              computeType,
    cusparseSpMMAlg_t         alg,
    void*                     externalBuffer)
    INLINE_BODY_STATUS(

    return rocsparse_spmm(handle->handle,
                   cuToRoc(opA), cuToRoc(opB),
                   &alpha, matA, matB,
                   &beta, matC,
                   cuToRoc(computeType),
                   cuToRoc(alg),
                   rocsparse_spmm_stage_compute,
                   nullptr,
                   externalBuffer);
)

/// Sparse General Matrix-Matrix Multiplication

// Usually, these are used as:
//  workEstimation(&buffer_size, buffer = nullptr)
//  workEstimation(&buffer_size, buffer)
//  compute(&buffer_size, buffer = nullptr)
//  compute(&buffer_size, buffer)

GPUSPARSE_EXPORT_C cusparseStatus_t
cusparseSpGEMM_createDescr(cusparseSpGEMMDescr_t* descr)
    INLINE_BODY_STATUS(
    *descr = new cusparseSpGEMMDescr;
    return rocsparse_status_success;
)

GPUSPARSE_EXPORT_C cusparseStatus_t
cusparseSpGEMM_destroyDescr(cusparseSpGEMMDescr_t descr)
    INLINE_BODY_STATUS(
    delete descr;
    return rocsparse_status_success;
)

GPUSPARSE_EXPORT_C cusparseStatus_t
cusparseSpGEMM_workEstimation(
    __CXX17UNUSED cusparseHandle_t          handle,
    __CXX17UNUSED cusparseOperation_t       opA,
    __CXX17UNUSED cusparseOperation_t       opB,
    __CXX17UNUSED const void*               alpha,
    __CXX17UNUSED cusparseConstSpMatDescr_t matA,
    __CXX17UNUSED cusparseConstSpMatDescr_t matB,
    __CXX17UNUSED const void*               beta,
    __CXX17UNUSED cusparseSpMatDescr_t      matC,
    __CXX17UNUSED cudaDataType              computeType,
    __CXX17UNUSED cusparseSpGEMMAlg_t       alg,
    __CXX17UNUSED cusparseSpGEMMDescr_t     spgemmDescr,
    __CXX17UNUSED size_t*                   bufferSize1,
    __CXX17UNUSED void*                     externalBuffer1)
    INLINE_BODY_STATUS(
    return rocsparse_status_success;
)

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
//     size_t*                   bufferSize2);

#define spgemm(stage) rocsparse_spgemm(cuToRoc(handle), cuToRoc(opA), cuToRoc(opB), \
        alpha, cuToRoc(matA), cuToRoc(matB), beta, matC, \
        matC, cuToRoc(computeType), cuToRoc(alg), stage, bufferSize2, externalBuffer2);

GPUSPARSE_EXPORT_C cusparseStatus_t
cusparseSpGEMM_compute( 
    cusparseHandle_t          handle,
    cusparseOperation_t       opA,
    cusparseOperation_t       opB,
    const void*               alpha,
    cusparseConstSpMatDescr_t matA,
    cusparseConstSpMatDescr_t matB,
    const void*               beta,
    cusparseSpMatDescr_t      matC,
    cudaDataType              computeType,
    cusparseSpGEMMAlg_t       alg,
    cusparseSpGEMMDescr_t     spgemmDescr,
    size_t*                   bufferSize2,
    void*                     externalBuffer2)
    INLINE_BODY_STATUS(
        if (externalBuffer2 == nullptr)
            MAYBE_ERROR(spgemm(rocsparse_spgemm_stage_buffer_size));

        MAYBE_ERROR(spgemm(rocsparse_spgemm_stage_nnz));

        // We will do sanity checking with this later in _copy()
        spgemmDescr->matC = matC;

        return spgemm(rocsparse_spgemm_stage_compute);
)
#undef spgemm

// As far as I understand, this should copy the result, out of the magical `spgemmDescr` into matC. I hate this API.
GPUSPARSE_EXPORT_C cusparseStatus_t
cusparseSpGEMM_copy(
    __CXX17UNUSED cusparseHandle_t          handle,
    __CXX17UNUSED cusparseOperation_t       opA,
    __CXX17UNUSED cusparseOperation_t       opB,
    __CXX17UNUSED const void*               alpha,
    __CXX17UNUSED cusparseConstSpMatDescr_t matA,
    __CXX17UNUSED cusparseConstSpMatDescr_t matB,
    __CXX17UNUSED const void*               beta,
    __CXX17UNUSED cusparseSpMatDescr_t      matC,
    __CXX17UNUSED cudaDataType              computeType,
    __CXX17UNUSED cusparseSpGEMMAlg_t       alg,
    __CXX17UNUSED cusparseSpGEMMDescr_t     spgemmDescr)
    INLINE_BODY_STATUS(
    // Assume the compute happened on the sane matrices
    // ... but also check
    if (spgemmDescr->matC != matC)
        return rocsparse_status_internal_error;
    return rocsparse_status_success;
)

//
// SPARSE MATRIX - SPARSE MATRIX MULTIPLICATION (SpGEMM) STRUCTURE REUSE
//

// SPARSE_API_DIRECT(SpGEMMreuse_workEstimation, please_write,  
//                                    cusparseOperation_t,       opA,
//                                    cusparseOperation_t,       opB,
//                                    cusparseConstSpMatDescr_t, matA,
//                                    cusparseConstSpMatDescr_t, matB,
//                                    cusparseSpMatDescr_t,      matC,
//                                    cusparseSpGEMMAlg_t,       alg,
//                                    cusparseSpGEMMDescr_t,     spgemmDescr,
//                                    size_t*,                   bufferSize1,
//                                    void*,                     externalBuffer1);

// SPARSE_API_DIRECT(SpGEMMreuse_nnz, please_write, 
//                         cusparseOperation_t,       opA,
//                         cusparseOperation_t,       opB,
//                         cusparseConstSpMatDescr_t, matA,
//                         cusparseConstSpMatDescr_t, matB,
//                         cusparseSpMatDescr_t,      matC,
//                         cusparseSpGEMMAlg_t,       alg,
//                         cusparseSpGEMMDescr_t,     spgemmDescr,
//                         size_t*,                   bufferSize2,
//                         void*,                     externalBuffer2,
//                         size_t*,                   bufferSize3,
//                         void*,                     externalBuffer3,
//                         size_t*,                   bufferSize4,
//                         void*,                     externalBuffer4);

// SPARSE_API_DIRECT(SpGEMMreuse_copy, please_write,    
//                          cusparseOperation_t,       opA,
//                          cusparseOperation_t,       opB,
//                          cusparseConstSpMatDescr_t, matA,
//                          cusparseConstSpMatDescr_t, matB,
//                          cusparseSpMatDescr_t,      matC,
//                          cusparseSpGEMMAlg_t,       alg,
//                          cusparseSpGEMMDescr_t,     spgemmDescr,
//                          size_t*,                   bufferSize5,
//                          void*,                     externalBuffer5);

// SPARSE_API_DIRECT(SpGEMMreuse_compute, please_write, 
//                             cusparseOperation_t,       opA,
//                             cusparseOperation_t,       opB,
//                             const void*,               alpha,
//                             cusparseConstSpMatDescr_t, matA,
//                             cusparseConstSpMatDescr_t, matB,
//                             const void*,               beta,
//                             cusparseSpMatDescr_t,      matC,
//                             cudaDataType,              computeType,
//                             cusparseSpGEMMAlg_t,       alg,
//                             cusparseSpGEMMDescr_t,     spgemmDescr);

/// Sampled Dense-Dense Matrix Multiplication


// SPARSE_API_DIRECT(SDDMM_bufferSize, please_write,    
//                          cusparseOperation_t,       opA,
//                          cusparseOperation_t,       opB,
//                          const void*,               alpha,
//                          cusparseConstDnMatDescr_t, matA,
//                          cusparseConstDnMatDescr_t, matB,
//                          const void*,               beta,
//                          cusparseSpMatDescr_t,      matC,
//                          cudaDataType,              computeType,
//                          cusparseSDDMMAlg_t,        alg,
//                          size_t*,                   bufferSize);

// SPARSE_API_DIRECT(SDDMM_preprocess, please_write,    
//                          cusparseOperation_t,       opA,
//                          cusparseOperation_t,       opB,
//                          const void*,               alpha,
//                          cusparseConstDnMatDescr_t, matA,
//                          cusparseConstDnMatDescr_t, matB,
//                          const void*,               beta,
//                          cusparseSpMatDescr_t,      matC,
//                          cudaDataType,              computeType,
//                          cusparseSDDMMAlg_t,        alg,
//                          void*,                     externalBuffer);

// SPARSE_API_DIRECT(SDDMM, please_write,   
//               cusparseOperation_t,       opA,
//               cusparseOperation_t,       opB,
//               const void*,               alpha,
//               cusparseConstDnMatDescr_t, matA,
//               cusparseConstDnMatDescr_t, matB,
//               const void*,               beta,
//               cusparseSpMatDescr_t,      matC,
//               cudaDataType,              computeType,
//               cusparseSDDMMAlg_t,        alg,
//               void*,                     externalBuffer);

//
// Generic APIs with custom operations
//


// SPARSE_API_DIRECT(SpMMOp_createPlan, please_write,   
//                           cusparseSpMMOpPlan_t*,     plan,
//                           cusparseOperation_t,       opA,
//                           cusparseOperation_t,       opB,
//                           cusparseConstSpMatDescr_t, matA,
//                           cusparseConstDnMatDescr_t, matB,
//                           cusparseDnMatDescr_t,      matC,
//                           cudaDataType,              computeType,
//                           cusparseSpMMOpAlg_t,       alg,
//                           const void*,               addOperationNvvmBuffer,
//                           size_t,                    addOperationBufferSize,
//                           const void*,               mulOperationNvvmBuffer,
//                           size_t,                    mulOperationBufferSize,
//                           const void*,               epilogueNvvmBuffer,
//                           size_t,                    epilogueBufferSize,
//                           size_t*,                   SpMMWorkspaceSize);

// SPARSE_API_DIRECT_NOHANDLE(SpMMOp, please_Write_me,
//  cusparseSpMMOpPlan_t plan,
//                void*,                externalBuffer);

// SPARSE_API_DIRECT_NOHANDLE(SpMMOp_destroyPlan, please_Write_me,
//  cusparseSpMMOpPlan_t plan);
