
#include "common.h"

#include "cusparse.h"

/// Sparse Operations Level 2 

// SPARSE_API_DIRECT(Sgemvi, please_write,  
//                cusparseOperation_t, transA,
//                int,                 m,
//                int,                 n,
//                const float*,        alpha,
//                const float*,        A,
//                int,                 lda,
//                int,                 nnz,
//                const float*,        xVal,
//                const int*,          xInd,
//                const float*,        beta,
//                float*,              y,
//                cusparseIndexBase_t, idxBase,
//                void*,               pBuffer)

// SPARSE_API_DIRECT(Sgemvi_bufferSize, please_write,
//                           cusparseOperation_t, transA,
//                           int,                 m,
//                           int,                 n,
//                           int,                 nnz,
//                           int*,                pBufferSize)

// SPARSE_API_DIRECT(Dgemvi, please_write,
//                cusparseOperation_t, transA,
//                int,                 m,
//                int,                 n,
//                const double*,       alpha,
//                const double*,       A,
//                int,                 lda,
//                int,                 nnz,
//                const double*,       xVal,
//                const int*,          xInd,
//                const double*,       beta,
//                double*,             y,
//                cusparseIndexBase_t, idxBase,
//                void*,               pBuffer)

// SPARSE_API_DIRECT(Dgemvi_bufferSize, please_write,
//                           cusparseOperation_t, transA,
//                           int,                 m,
//                           int,                 n,
//                           int,                 nnz,
//                           int*,                pBufferSize)

// SPARSE_API_DIRECT(Cgemvi, please_write,
//                cusparseOperation_t, transA,
//                int,                 m,
//                int,                 n,
//                const cuComplex*,    alpha,
//                const cuComplex*,    A,
//                int,                 lda,
//                int,                 nnz,
//                const cuComplex*,    xVal,
//                const int*,          xInd,
//                const cuComplex*,    beta,
//                cuComplex*,          y,
//                cusparseIndexBase_t, idxBase,
//                void*,               pBuffer)

// SPARSE_API_DIRECT(Cgemvi_bufferSize, please_write,
//                           cusparseOperation_t, transA,
//                           int,                 m,
//                           int,                 n,
//                           int,                 nnz,
//                           int*,                pBufferSize)

// SPARSE_API_DIRECT(Zgemvi, please_write,
//                cusparseOperation_t,    transA,
//                int,                    m,
//                int,                    n,
//                const cuDoubleComplex*, alpha,
//                const cuDoubleComplex*, A,
//                int,                    lda,
//                int,                    nnz,
//                const cuDoubleComplex*, xVal,
//                const int*,             xInd,
//                const cuDoubleComplex*, beta,
//                cuDoubleComplex*,       y,
//                cusparseIndexBase_t,    idxBase,
//                void*,                  pBuffer)

// SPARSE_API_DIRECT(Zgemvi_bufferSize, please_write,
//                           cusparseOperation_t, transA,
//                           int,                 m,
//                           int,                 n,
//                           int,                 nnz,
//                           int*,                pBufferSize)


#define bsrmv_ARGS dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, x, beta, y
#define bsrmv_body(ROCLETTER) \
    INLINE_BODY( \
    rocsparse_mat_info info; \
    rocsparse_create_mat_info(&info); \
    auto ret = rocsparse_##ROCLETTER##bsrmv(MAP(CU_TO_ROC, COMMA, handle->handle, bsrmv_ARGS)); \
    rocsparse_destroy_mat_info(info); \
    \
    return __redscale_cusparseStatus_t(ret); \
)

GPUSPARSE_EXPORT_C cusparseStatus_t
cusparseSbsrmv(cusparseHandle_t handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               int                      mb,
               int                      nb,
               int                      nnzb,
               const float*             alpha,
               const cusparseMatDescr_t descrA,
               const float*             bsrSortedValA,
               const int*               bsrSortedRowPtrA,
               const int*               bsrSortedColIndA,
               int                      blockDim,
               const float*             x,
               const float*             beta,
               float*                   y)
    bsrmv_body(s)

GPUSPARSE_EXPORT_C cusparseStatus_t
cusparseDbsrmv(cusparseHandle_t handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               int                      mb,
               int                      nb,
               int                      nnzb,
               const double*            alpha,
               const cusparseMatDescr_t descrA,
               const double*            bsrSortedValA,
               const int*               bsrSortedRowPtrA,
               const int*               bsrSortedColIndA,
               int                      blockDim,
               const double*            x,
               const double*            beta,
               double*                  y)
    bsrmv_body(d)

GPUSPARSE_EXPORT_C cusparseStatus_t
cusparseCbsrmv(cusparseHandle_t handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               int                      mb,
               int                      nb,
               int                      nnzb,
               const cuComplex*         alpha,
               const cusparseMatDescr_t descrA,
               const cuComplex*         bsrSortedValA,
               const int*               bsrSortedRowPtrA,
               const int*               bsrSortedColIndA,
               int                      blockDim,
               const cuComplex*         x,
               const cuComplex*         beta,
               cuComplex*               y)
    bsrmv_body(c)

GPUSPARSE_EXPORT_C cusparseStatus_t
cusparseZbsrmv(cusparseHandle_t handle,
               cusparseDirection_t      dirA,
               cusparseOperation_t      transA,
               int                      mb,
               int                      nb,
               int                      nnzb,
               const cuDoubleComplex*   alpha,
               const cusparseMatDescr_t descrA,
               const cuDoubleComplex*   bsrSortedValA,
               const int*               bsrSortedRowPtrA,
               const int*               bsrSortedColIndA,
               int                      blockDim,
               const cuDoubleComplex*   x,
               const cuDoubleComplex*   beta,
               cuDoubleComplex*         y)
    bsrmv_body(z)

#undef bsrmv_body
#undef bsrmv_ARGS

SPARSE_API(S, s, bsrxmv, Bsrxmv,
                cusparseDirection_t,      dirA,
                cusparseOperation_t,      transA,
                int,                      sizeOfMask,
                int,                      mb,
                int,                      nb,
                int,                      nnzb,
                const float*,             alpha,
                const cusparseMatDescr_t, descrA,
                const float*,             bsrSortedValA,
                const int*,               bsrSortedMaskPtrA,
                const int*,               bsrSortedRowPtrA,
                const int*,               bsrSortedEndPtrA,
                const int*,               bsrSortedColIndA,
                int,                      blockDim,
                const float*,             x,
                const float*,             beta,
                float*,                   y)
SPARSE_API(D, d, bsrxmv, Bsrxmv,
                cusparseDirection_t,      dirA,
                cusparseOperation_t,      transA,
                int,                      sizeOfMask,
                int,                      mb,
                int,                      nb,
                int,                      nnzb,
                const double*,            alpha,
                const cusparseMatDescr_t, descrA,
                const double*,            bsrSortedValA,
                const int*,               bsrSortedMaskPtrA,
                const int*,               bsrSortedRowPtrA,
                const int*,               bsrSortedEndPtrA,
                const int*,               bsrSortedColIndA,
                int,                      blockDim,
                const double*,            x,
                const double*,            beta,
                double*,                  y)

SPARSE_API(C, c, bsrxmv, Bsrxmv,
                cusparseDirection_t,      dirA,
                cusparseOperation_t,      transA,
                int,                      sizeOfMask,
                int,                      mb,
                int,                      nb,
                int,                      nnzb,
                const cuComplex*,         alpha,
                const cusparseMatDescr_t, descrA,
                const cuComplex*,         bsrSortedValA,
                const int*,               bsrSortedMaskPtrA,
                const int*,               bsrSortedRowPtrA,
                const int*,               bsrSortedEndPtrA,
                const int*,               bsrSortedColIndA,
                int,                      blockDim,
                const cuComplex*,         x,
                const cuComplex*,         beta,
                cuComplex*,               y)

SPARSE_API(Z, z, bsrxmv, Bsrxmv,
                cusparseDirection_t,      dirA,
                cusparseOperation_t,      transA,
                int,                      sizeOfMask,
                int,                      mb,
                int,                      nb,
                int,                      nnzb,
                const cuDoubleComplex*,   alpha,
                const cusparseMatDescr_t, descrA,
                const cuDoubleComplex*,   bsrSortedValA,
                const int*,               bsrSortedMaskPtrA,
                const int*,               bsrSortedRowPtrA,
                const int*,               bsrSortedEndPtrA,
                const int*,               bsrSortedColIndA,
                int,                      blockDim,
                const cuDoubleComplex*,   x,
                const cuDoubleComplex*,   beta,
                cuDoubleComplex*,         y)

// SPARSE_API_DIRECT(Xbsrsv2_zeroPivot, please_write,
//                           bsrsv2Info_t,     info,
//                           int*,             position)

// SPARSE_API_DIRECT(Sbsrsv2_bufferSize, please_write,
//                            cusparseDirection_t,      dirA,
//                            cusparseOperation_t,      transA,
//                            int,                      mb,
//                            int,                      nnzb,
//                            const cusparseMatDescr_t, descrA,
//                            float*,                   bsrSortedValA,
//                            const int*,               bsrSortedRowPtrA,
//                            const int*,               bsrSortedColIndA,
//                            int,                      blockDim,
//                            bsrsv2Info_t,             info,
//                            int*,                     pBufferSizeInBytes)

// SPARSE_API_DIRECT(Dbsrsv2_bufferSize, please_write,
//                            cusparseDirection_t,      dirA,
//                            cusparseOperation_t,      transA,
//                            int,                      mb,
//                            int,                      nnzb,
//                            const cusparseMatDescr_t, descrA,
//                            double*,                  bsrSortedValA,
//                            const int*,               bsrSortedRowPtrA,
//                            const int*,               bsrSortedColIndA,
//                            int,                      blockDim,
//                            bsrsv2Info_t,             info,
//                            int*,                     pBufferSizeInBytes)

// SPARSE_API_DIRECT(Cbsrsv2_bufferSize, please_write,
//                            cusparseDirection_t,      dirA,
//                            cusparseOperation_t,      transA,
//                            int,                      mb,
//                            int,                      nnzb,
//                            const cusparseMatDescr_t, descrA,
//                            cuComplex*,               bsrSortedValA,
//                            const int*,               bsrSortedRowPtrA,
//                            const int*,               bsrSortedColIndA,
//                            int,                      blockDim,
//                            bsrsv2Info_t,             info,
//                            int*,                     pBufferSizeInBytes)

// SPARSE_API_DIRECT(Zbsrsv2_bufferSize, please_write,
//                            cusparseDirection_t,      dirA,
//                            cusparseOperation_t,      transA,
//                            int,                      mb,
//                            int,                      nnzb,
//                            const cusparseMatDescr_t, descrA,
//                            cuDoubleComplex*,         bsrSortedValA,
//                            const int*,               bsrSortedRowPtrA,
//                            const int*,               bsrSortedColIndA,
//                            int,                      blockDim,
//                            bsrsv2Info_t,             info,
//                            int*,                     pBufferSizeInBytes)

// SPARSE_API_DIRECT(Sbsrsv2_bufferSizeExt, please_write,
//                               cusparseDirection_t,      dirA,
//                               cusparseOperation_t,      transA,
//                               int,                      mb,
//                               int,                      nnzb,
//                               const cusparseMatDescr_t, descrA,
//                               float*,                   bsrSortedValA,
//                               const int*,               bsrSortedRowPtrA,
//                               const int*,               bsrSortedColIndA,
//                               int,                      blockSize,
//                               bsrsv2Info_t,             info,
//                               size_t*,                  pBufferSize)

// SPARSE_API_DIRECT(Dbsrsv2_bufferSizeExt, please_write,
//                               cusparseDirection_t,      dirA,
//                               cusparseOperation_t,      transA,
//                               int,                      mb,
//                               int,                      nnzb,
//                               const cusparseMatDescr_t, descrA,
//                               double*,                  bsrSortedValA,
//                               const int*,               bsrSortedRowPtrA,
//                               const int*,               bsrSortedColIndA,
//                               int,                      blockSize,
//                               bsrsv2Info_t,             info,
//                               size_t*,                  pBufferSize)

// SPARSE_API_DIRECT(Cbsrsv2_bufferSizeExt, please_write,
//                               cusparseDirection_t,      dirA,
//                               cusparseOperation_t,      transA,
//                               int,                      mb,
//                               int,                      nnzb,
//                               const cusparseMatDescr_t, descrA,
//                               cuComplex*,               bsrSortedValA,
//                               const int*,               bsrSortedRowPtrA,
//                               const int*,               bsrSortedColIndA,
//                               int,                      blockSize,
//                               bsrsv2Info_t,             info,
//                               size_t*,                  pBufferSize)

// SPARSE_API_DIRECT(Zbsrsv2_bufferSizeExt, please_write,
//                               cusparseDirection_t,      dirA,
//                               cusparseOperation_t,      transA,
//                               int,                      mb,
//                               int,                      nnzb,
//                               const cusparseMatDescr_t, descrA,
//                               cuDoubleComplex*,         bsrSortedValA,
//                               const int*,               bsrSortedRowPtrA,
//                               const int*,               bsrSortedColIndA,
//                               int,                      blockSize,
//                               bsrsv2Info_t,             info,
//                               size_t*,                  pBufferSize)

// SPARSE_API_DIRECT(Sbsrsv2_analysis, please_write,
//                          cusparseDirection_t,      dirA,
//                          cusparseOperation_t,      transA,
//                          int,                      mb,
//                          int,                      nnzb,
//                          const cusparseMatDescr_t, descrA,
//                          const float*,             bsrSortedValA,
//                          const int*,               bsrSortedRowPtrA,
//                          const int*,               bsrSortedColIndA,
//                          int,                      blockDim,
//                          bsrsv2Info_t,             info,
//                          cusparseSolvePolicy_t,    policy,
//                          void*,                    pBuffer)

// SPARSE_API_DIRECT(Dbsrsv2_analysis, please_write,
//                          cusparseDirection_t,      dirA,
//                          cusparseOperation_t,      transA,
//                          int,                      mb,
//                          int,                      nnzb,
//                          const cusparseMatDescr_t, descrA,
//                          const double*,            bsrSortedValA,
//                          const int*,               bsrSortedRowPtrA,
//                          const int*,               bsrSortedColIndA,
//                          int,                      blockDim,
//                          bsrsv2Info_t,             info,
//                          cusparseSolvePolicy_t,    policy,
//                          void*,                    pBuffer)

// SPARSE_API_DIRECT(Cbsrsv2_analysis, please_write,
//                          cusparseDirection_t,      dirA,
//                          cusparseOperation_t,      transA,
//                          int,                      mb,
//                          int,                      nnzb,
//                          const cusparseMatDescr_t, descrA,
//                          const cuComplex*,         bsrSortedValA,
//                          const int*,               bsrSortedRowPtrA,
//                          const int*,               bsrSortedColIndA,
//                          int,                      blockDim,
//                          bsrsv2Info_t,             info,
//                          cusparseSolvePolicy_t,    policy,
//                          void*,                    pBuffer)

// SPARSE_API_DIRECT(Zbsrsv2_analysis, please_write,
//                          cusparseDirection_t,      dirA,
//                          cusparseOperation_t,      transA,
//                          int,                      mb,
//                          int,                      nnzb,
//                          const cusparseMatDescr_t, descrA,
//                          const cuDoubleComplex*,   bsrSortedValA,
//                          const int*,               bsrSortedRowPtrA,
//                          const int*,               bsrSortedColIndA,
//                          int,                      blockDim,
//                          bsrsv2Info_t,             info,
//                          cusparseSolvePolicy_t,    policy,
//                          void*,                    pBuffer)

// SPARSE_API_DIRECT(Sbsrsv2_solve, please_write,
//                       cusparseDirection_t,      dirA,
//                       cusparseOperation_t,      transA,
//                       int,                      mb,
//                       int,                      nnzb,
//                       const float*,             alpha,
//                       const cusparseMatDescr_t, descrA,
//                       const float*,             bsrSortedValA,
//                       const int*,               bsrSortedRowPtrA,
//                       const int*,               bsrSortedColIndA,
//                       int,                      blockDim,
//                       bsrsv2Info_t,             info,
//                       const float*,             f,
//                       float*,                   x,
//                       cusparseSolvePolicy_t,    policy,
//                       void*,                    pBuffer)

// SPARSE_API_DIRECT(Dbsrsv2_solve, please_write,
//                       cusparseDirection_t,      dirA,
//                       cusparseOperation_t,      transA,
//                       int,                      mb,
//                       int,                      nnzb,
//                       const double*,            alpha,
//                       const cusparseMatDescr_t, descrA,
//                       const double*,            bsrSortedValA,
//                       const int*,               bsrSortedRowPtrA,
//                       const int*,               bsrSortedColIndA,
//                       int,                      blockDim,
//                       bsrsv2Info_t,             info,
//                       const double*,            f,
//                       double*,                  x,
//                       cusparseSolvePolicy_t,    policy,
//                       void*,                    pBuffer)

// SPARSE_API_DIRECT(Cbsrsv2_solve, please_write,
//                       cusparseDirection_t,      dirA,
//                       cusparseOperation_t,      transA,
//                       int,                      mb,
//                       int,                      nnzb,
//                       const cuComplex*,         alpha,
//                       const cusparseMatDescr_t, descrA,
//                       const cuComplex*,         bsrSortedValA,
//                       const int*,               bsrSortedRowPtrA,
//                       const int*,               bsrSortedColIndA,
//                       int,                      blockDim,
//                       bsrsv2Info_t,             info,
//                       const cuComplex*,         f,
//                       cuComplex*,               x,
//                       cusparseSolvePolicy_t,    policy,
//                       void*,                    pBuffer)

// SPARSE_API_DIRECT(Zbsrsv2_solve, please_write,
//                       cusparseDirection_t,      dirA,
//                       cusparseOperation_t,      transA,
//                       int,                      mb,
//                       int,                      nnzb,
//                       const cuDoubleComplex*,   alpha,
//                       const cusparseMatDescr_t, descrA,
//                       const cuDoubleComplex*,   bsrSortedValA,
//                       const int*,               bsrSortedRowPtrA,
//                       const int*,               bsrSortedColIndA,
//                       int,                      blockDim,
//                       bsrsv2Info_t,             info,
//                       const cuDoubleComplex*,   f,
//                       cuDoubleComplex*,         x,
//                       cusparseSolvePolicy_t,    policy,
//                       void*,                    pBuffer)
