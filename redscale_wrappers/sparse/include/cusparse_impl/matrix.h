
#include "common.h"

/// Sparse Matrices

SPARSE_API_DIRECT_NOHANDLE(DestroySpMat, destroy_spmat_descr,
    cusparseConstSpMatDescr_t, spMatDescr)

// GPUSPARSE_EXPORT_C cusparseStatus_t
// cusparseSpMatGetFormat(cusparseConstSpMatDescr_t spMatDescr,
//                        cusparseFormat_t*         format);

// SPARSE_API_DIRECT_NOHANDLE(SpMatGetIndexBase, please_Write_me,
//  cusparseConstSpMatDescr_t spMatDescr,
//                           cusparseIndexBase_t*,      idxBase);

// SPARSE_API_DIRECT_NOHANDLE(SpMatGetValues, please_Write_me,
//  cusparseSpMatDescr_t spMatDescr,
//                        void**,               values);

// SPARSE_API_DIRECT_NOHANDLE(ConstSpMatGetValues, please_Write_me,
//  cusparseConstSpMatDescr_t spMatDescr,
//                             const void**,               values);

// SPARSE_API_DIRECT_NOHANDLE(SpMatSetValues, please_Write_me,
//  cusparseSpMatDescr_t spMatDescr,
//                        void*,                values);

SPARSE_API_DIRECT_NOHANDLE(SpMatGetSize, spmat_get_size,
    cusparseConstSpMatDescr_t, spMatDescr,
    int64_t*,                  rows,
    int64_t*,                  cols,
    int64_t*,                  nnz);

// SPARSE_API_DIRECT_NOHANDLE(SpMatGetStridedBatch, please_Write_me,
//  cusparseConstSpMatDescr_t spMatDescr,
//                              int*,                      batchCount);

// SPARSE_API_DIRECT_NOHANDLE(CooSetStridedBatch, please_Write_me,
//  cusparseSpMatDescr_t spMatDescr,
//                            int,                  batchCount,
//                            int64_t,              batchStride);

// SPARSE_API_DIRECT_NOHANDLE(CsrSetStridedBatch, please_Write_me,
//  cusparseSpMatDescr_t spMatDescr,
//                            int,                  batchCount,
//                            int64_t,              offsetsBatchStride,
//                            int64_t,              columnsValuesBatchStride);

// SPARSE_API_DIRECT_NOHANDLE(BsrSetStridedBatch, please_Write_me,
//  cusparseSpMatDescr_t spMatDescr,
//                            int,                  batchCount,
//                            int64_t,              offsetsBatchStride,
//                            int64_t,              columnsValuesBatchStride,
//                            int64_t,              ValuesBatchStride);


// GPUSPARSE_EXPORT_C cusparseStatus_t
// cusparseSpMatGetAttribute(cusparseConstSpMatDescr_t spMatDescr,
//                           cusparseSpMatAttribute_t,  attribute,
//                           void*,                     data,
//                           size_t,                    dataSize);

// SPARSE_API_DIRECT_NOHANDLE(SpMatSetAttribute, please_Write_me,
//  cusparseSpMatDescr_t     spMatDescr,
//                           cusparseSpMatAttribute_t, attribute,
//                           void*,                    data,
//                           size_t,                   dataSize);

//
// CSR 

//define CreateCsr_ARGS spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType
SPARSE_API_DIRECT_NOHANDLE(CreateCsr, create_csr_descr,
    cusparseSpMatDescr_t*, spMatDescr,
    int64_t,               rows,
    int64_t,               cols,
    int64_t,               nnz,
    void*,                 csrRowOffsets,
    void*,                 csrColInd,
    void*,                 csrValues,
    cusparseIndexType_t,   csrRowOffsetsType,
    cusparseIndexType_t,   csrColIndType,
    cusparseIndexBase_t,   idxBase,
    cudaDataType,          valueType);
//undef CreateCsr_ARGS

// SPARSE_API_DIRECT_NOHANDLE(CreateConstCsr, please_Write_me,
//  cusparseConstSpMatDescr_t* spMatDescr,
//                        int64_t,                    rows,
//                        int64_t,                    cols,
//                        int64_t,                    nnz,
//                        const void*,                csrRowOffsets,
//                        const void*,                csrColInd,
//                        const void*,                csrValues,
//                        cusparseIndexType_t,        csrRowOffsetsType,
//                        cusparseIndexType_t,        csrColIndType,
//                        cusparseIndexBase_t,        idxBase,
//                        cudaDataType,               valueType);

// SPARSE_API_DIRECT_NOHANDLE(CreateCsc, please_Write_me,
//  cusparseSpMatDescr_t* spMatDescr,
//                   int64_t,               rows,
//                   int64_t,               cols,
//                   int64_t,               nnz,
//                   void*,                 cscColOffsets,
//                   void*,                 cscRowInd,
//                   void*,                 cscValues,
//                   cusparseIndexType_t,   cscColOffsetsType,
//                   cusparseIndexType_t,   cscRowIndType,
//                   cusparseIndexBase_t,   idxBase,
//                   cudaDataType,          valueType);

// SPARSE_API_DIRECT_NOHANDLE(CreateConstCsc, please_Write_me,
//  cusparseConstSpMatDescr_t* spMatDescr,
//                        int64_t,                    rows,
//                        int64_t,                    cols,
//                        int64_t,                    nnz,
//                        const void*,                cscColOffsets,
//                        const void*,                cscRowInd,
//                        const void*,                cscValues,
//                        cusparseIndexType_t,        cscColOffsetsType,
//                        cusparseIndexType_t,        cscRowIndType,
//                        cusparseIndexBase_t,        idxBase,
//                        cudaDataType,               valueType);

// SPARSE_API_DIRECT_NOHANDLE(CsrGet, please_Write_me,
//  cusparseSpMatDescr_t spMatDescr,
//                int64_t*,             rows,
//                int64_t*,             cols,
//                int64_t*,             nnz,
//                void**,               csrRowOffsets,
//                void**,               csrColInd,
//                void**,               csrValues,
//                cusparseIndexType_t*, csrRowOffsetsType,
//                cusparseIndexType_t*, csrColIndType,
//                cusparseIndexBase_t*, idxBase,
//                cudaDataType*,        valueType);

// SPARSE_API_DIRECT_NOHANDLE(ConstCsrGet, please_Write_me,
//  cusparseConstSpMatDescr_t spMatDescr,
//                     int64_t*,                  rows,
//                     int64_t*,                  cols,
//                     int64_t*,                  nnz,
//                     const void**,              csrRowOffsets,
//                     const void**,              csrColInd,
//                     const void**,              csrValues,
//                     cusparseIndexType_t*,      csrRowOffsetsType,
//                     cusparseIndexType_t*,      csrColIndType,
//                     cusparseIndexBase_t*,      idxBase,
//                     cudaDataType*,             valueType);

// SPARSE_API_DIRECT_NOHANDLE(CscGet, please_Write_me,
//  cusparseSpMatDescr_t spMatDescr,
//                int64_t*,             rows,
//                int64_t*,             cols,
//                int64_t*,             nnz,
//                void**,               cscColOffsets,
//                void**,               cscRowInd,
//                void**,               cscValues,
//                cusparseIndexType_t*, cscColOffsetsType,
//                cusparseIndexType_t*, cscRowIndType,
//                cusparseIndexBase_t*, idxBase,
//                cudaDataType*,        valueType);

// SPARSE_API_DIRECT_NOHANDLE(ConstCscGet, please_Write_me,
//  cusparseConstSpMatDescr_t spMatDescr,
//                     int64_t*,                  rows,
//                     int64_t*,                  cols,
//                     int64_t*,                  nnz,
//                     const void**,              cscColOffsets,
//                     const void**,              cscRowInd,
//                     const void**,              cscValues,
//                     cusparseIndexType_t*,      cscColOffsetsType,
//                     cusparseIndexType_t*,      cscRowIndType,
//                     cusparseIndexBase_t*,      idxBase,
//                     cudaDataType*,             valueType);

SPARSE_API_DIRECT_NOHANDLE(CsrSetPointers, csr_set_pointers,
    cusparseSpMatDescr_t, spMatDescr,
    void*,                csrRowOffsets,
    void*,                csrColInd,
    void*,                csrValues);

// SPARSE_API_DIRECT_NOHANDLE(CscSetPointers, please_Write_me,
//  cusparseSpMatDescr_t spMatDescr,
//                        void*,                cscColOffsets,
//                        void*,                cscRowInd,
//                        void*,                cscValues);

//
// BSR 

// SPARSE_API_DIRECT_NOHANDLE(CreateBsr, please_Write_me,
//  cusparseSpMatDescr_t* spMatDescr,
//                   int64_t,               brows,
//                   int64_t,               bcols,
//                   int64_t,               bnnz,
//                   int64_t,               rowBlockDim,
//                   int64_t,               colBlockDim,
//                   void*,                 bsrRowOffsets,
//                   void*,                 bsrColInd,
//                   void*,                 bsrValues,
//                   cusparseIndexType_t,   bsrRowOffsetsType,
//                   cusparseIndexType_t,   bsrColIndType,
//                   cusparseIndexBase_t,   idxBase,
//                   cudaDataType,          valueType,
//                   cusparseOrder_t,       order);

// SPARSE_API_DIRECT_NOHANDLE(CreateConstBsr, please_Write_me,
//  cusparseConstSpMatDescr_t* spMatDescr,
//                        int64_t,                    brows,
//                        int64_t,                    bcols,
//                        int64_t,                    bnnz,
//                        int64_t,                    rowBlockDim,
//                        int64_t,                    colBlockDim,
//                        const void*,                bsrRowOffsets,
//                        const void*,                bsrColInd,
//                        const void*,                bsrValues,
//                        cusparseIndexType_t,        bsrRowOffsetsType,
//                        cusparseIndexType_t,        bsrColIndType,
//                        cusparseIndexBase_t,        idxBase,
//                        cudaDataType,               valueType,
//                        cusparseOrder_t,            order);

//
// COO 

// SPARSE_API_DIRECT_NOHANDLE(CreateCoo, please_Write_me,
//  cusparseSpMatDescr_t* spMatDescr,
//                   int64_t,               rows,
//                   int64_t,               cols,
//                   int64_t,               nnz,
//                   void*,                 cooRowInd,
//                   void*,                 cooColInd,
//                   void*,                 cooValues,
//                   cusparseIndexType_t,   cooIdxType,
//                   cusparseIndexBase_t,   idxBase,
//                   cudaDataType,          valueType);

// SPARSE_API_DIRECT_NOHANDLE(CreateConstCoo, please_Write_me,
//  cusparseConstSpMatDescr_t* spMatDescr,
//                        int64_t,                    rows,
//                        int64_t,                    cols,
//                        int64_t,                    nnz,
//                        const void*,                cooRowInd,
//                        const void*,                cooColInd,
//                        const void*,                cooValues,
//                        cusparseIndexType_t,        cooIdxType,
//                        cusparseIndexBase_t,        idxBase,
//                        cudaDataType,               valueType);

// SPARSE_API_DIRECT_NOHANDLE(CooGet, please_Write_me,
//  cusparseSpMatDescr_t spMatDescr,
//                int64_t*,             rows,
//                int64_t*,             cols,
//                int64_t*,             nnz,
//                void**               cooRowInd,  // COO row indices
//                void**               cooColInd,  // COO column indices
//                void**               cooValues,  // COO values
//                cusparseIndexType_t*, idxType,
//                cusparseIndexBase_t*, idxBase,
//                cudaDataType*,        valueType);

// SPARSE_API_DIRECT_NOHANDLE(ConstCooGet, please_Write_me,
//  cusparseConstSpMatDescr_t spMatDescr,
//                     int64_t*,                  rows,
//                     int64_t*,                  cols,
//                     int64_t*,                  nnz,
//                     const void**              cooRowInd,  // COO row indices
//                     const void**              cooColInd,  // COO column indices
//                     const void**              cooValues,  // COO values
//                     cusparseIndexType_t*,      idxType,
//                     cusparseIndexBase_t*,      idxBase,
//                     cudaDataType*,             valueType);

// SPARSE_API_DIRECT_NOHANDLE(CooSetPointers, please_Write_me,
//  cusparseSpMatDescr_t spMatDescr,
//                        void*,                cooRows,
//                        void*,                cooColumns,
//                        void*,                cooValues);

//
// BLOCKED ELL 

// SPARSE_API_DIRECT_NOHANDLE(CreateBlockedEll, please_Write_me,
//  cusparseSpMatDescr_t* spMatDescr,
//                          int64_t,               rows,
//                          int64_t,               cols,
//                          int64_t,               ellBlockSize,
//                          int64_t,               ellCols,
//                          void*,                 ellColInd,
//                          void*,                 ellValue,
//                          cusparseIndexType_t,   ellIdxType,
//                          cusparseIndexBase_t,   idxBase,
//                          cudaDataType,          valueType);

// SPARSE_API_DIRECT_NOHANDLE(CreateConstBlockedEll, please_Write_me,
//  cusparseConstSpMatDescr_t* spMatDescr,
//                               int64_t,                    rows,
//                               int64_t,                    cols,
//                               int64_t,                    ellBlockSize,
//                               int64_t,                    ellCols,
//                               const void*,                ellColInd,
//                               const void*,                ellValue,
//                               cusparseIndexType_t,        ellIdxType,
//                               cusparseIndexBase_t,        idxBase,
//                               cudaDataType,               valueType);

// SPARSE_API_DIRECT_NOHANDLE(BlockedEllGet, please_Write_me,
//  cusparseSpMatDescr_t spMatDescr,
//                       int64_t*,             rows,
//                       int64_t*,             cols,
//                       int64_t*,             ellBlockSize,
//                       int64_t*,             ellCols,
//                       void**,               ellColInd,
//                       void**,               ellValue,
//                       cusparseIndexType_t*, ellIdxType,
//                       cusparseIndexBase_t*, idxBase,
//                       cudaDataType*,        valueType);

// SPARSE_API_DIRECT_NOHANDLE(ConstBlockedEllGet, please_Write_me,
//  cusparseConstSpMatDescr_t spMatDescr,
//                            int64_t*,                  rows,
//                            int64_t*,                  cols,
//                            int64_t*,                  ellBlockSize,
//                            int64_t*,                  ellCols,
//                            const void**,              ellColInd,
//                            const void**,              ellValue,
//                            cusparseIndexType_t*,      ellIdxType,
//                            cusparseIndexBase_t*,      idxBase,
//                            cudaDataType*,             valueType);

//
// Sliced ELLPACK 

// SPARSE_API_DIRECT_NOHANDLE(CreateSlicedEll, please_Write_me,
//  cusparseSpMatDescr_t*   spMatDescr,
//                         int64_t,                 rows,
//                         int64_t,                 cols,
//                         int64_t,                 nnz,
//                         int64_t,                 sellValuesSize,
//                         int64_t,                 sliceSize,
//                     void*,                   sellSliceOffsets,
//                         void*,                   sellColInd,
//                         void*,                   sellValues,
//             cusparseIndexType_t,     sellSliceOffsetsType,
//                         cusparseIndexType_t,     sellColIndType,
//                         cusparseIndexBase_t,     idxBase,
//                         cudaDataType,            valueType);

// SPARSE_API_DIRECT_NOHANDLE(CreateConstSlicedEll, please_Write_me,
//  cusparseConstSpMatDescr_t*     spMatDescr,
//                              int64_t,                        rows,
//                              int64_t,                        cols,
//                              int64_t,                        nnz,
//                              int64_t,                        sellValuesSize,
//                              int64_t,                        sliceSize,
//                              void*,                          sellSliceOffsets,
//                              void*,                          sellColInd,
//                              void*,                          sellValues,
//                              cusparseIndexType_t,            sellSliceOffsetsType,
//                              cusparseIndexType_t,            sellColIndType,
//                              cusparseIndexBase_t,            idxBase,
//                              cudaDataType,                   valueType);

/// Dense Matrices

SPARSE_API_DIRECT_NOHANDLE(CreateDnMat, create_dnmat_descr,
    cusparseDnMatDescr_t*, dnMatDescr,
    int64_t,               rows,
    int64_t,               cols,
    int64_t,               ld,
    void*,                 values,
    cudaDataType,          valueType,
    cusparseOrder_t,       order);

// SPARSE_API_DIRECT_NOHANDLE(CreateConstDnMat, please_Write_me,
//  cusparseConstDnMatDescr_t* dnMatDescr,
//                          int64_t,                    rows,
//                          int64_t,                    cols,
//                          int64_t,                    ld,
//                          const void*,                values,
//                          cudaDataType,               valueType,
//                          cusparseOrder_t,            order);

SPARSE_API_DIRECT_NOHANDLE(DestroyDnMat, destroy_dnmat_descr,
    cusparseConstDnMatDescr_t, dnMatDescr);

// SPARSE_API_DIRECT_NOHANDLE(DnMatGet, please_Write_me,
//  cusparseDnMatDescr_t dnMatDescr,
//                  int64_t*,             rows,
//                  int64_t*,             cols,
//                  int64_t*,             ld,
//                  void**,               values,
//                  cudaDataType*,        type,
//                  cusparseOrder_t*,     order);

// SPARSE_API_DIRECT_NOHANDLE(ConstDnMatGet, please_Write_me,
//  cusparseConstDnMatDescr_t dnMatDescr,
//                       int64_t*,                  rows,
//                       int64_t*,                  cols,
//                       int64_t*,                  ld,
//                       const void**,              values,
//                       cudaDataType*,             type,
//                       cusparseOrder_t*,          order);

// SPARSE_API_DIRECT_NOHANDLE(DnMatGetValues, please_Write_me,
//  cusparseDnMatDescr_t dnMatDescr,
//                        void**,               values);

// SPARSE_API_DIRECT_NOHANDLE(ConstDnMatGetValues, please_Write_me,
//  cusparseConstDnMatDescr_t dnMatDescr,
//                             const void**,              values);

// SPARSE_API_DIRECT_NOHANDLE(DnMatSetValues, please_Write_me,
//  cusparseDnMatDescr_t dnMatDescr,
//                        void*,                values);

// SPARSE_API_DIRECT_NOHANDLE(DnMatSetStridedBatch, please_Write_me,
//  cusparseDnMatDescr_t dnMatDescr,
//                              int,                  batchCount,
//                              int64_t,              batchStride);

// SPARSE_API_DIRECT_NOHANDLE(DnMatGetStridedBatch, please_Write_me,
//  cusparseConstDnMatDescr_t dnMatDescr,
//                              int*,                      batchCount,
//                              int64_t*,                  batchStride);
