
#include "common.h"

// 
// SPARSE MATRIX SORTING
// 

// SPARSE_API_DIRECT(CreateIdentityPermutation, please_write,   
//                                   int,              n,
//                                   int*,             p);

// SPARSE_API_DIRECT(Xcoosort_bufferSizeExt, please_write,  
//                                int,              m,
//                                int,              n,
//                                int,              nnz,
//                                const int*,       cooRowsA,
//                                const int*,       cooColsA,
//                                size_t*,          pBufferSizeInBytes);

// SPARSE_API_DIRECT(XcoosortByRow, please_write,   
//                       int,              m,
//                       int,              n,
//                       int,              nnz,
//                       int*,             cooRowsA,
//                       int*,             cooColsA,
//                       int*,             P,
//                       void*,            pBuffer);

// SPARSE_API_DIRECT(XcoosortByColumn, please_write,    
//                          int,              m,
//                          int,              n,
//                          int,              nnz,
//                          int*,             cooRowsA,
//                          int*,             cooColsA,
//                          int*,             P,
//                          void*,            pBuffer);

// SPARSE_API_DIRECT(Xcsrsort_bufferSizeExt, please_write,  
//                                int,              m,
//                                int,              n,
//                                int,              nnz,
//                                const int*,       csrRowPtrA,
//                                const int*,       csrColIndA,
//                                size_t*,          pBufferSizeInBytes);

// SPARSE_API_DIRECT(Xcsrsort, please_write,    
//                  int,                      m,
//                  int,                      n,
//                  int,                      nnz,
//                  const cusparseMatDescr_t, descrA,
//                  const int*,               csrRowPtrA,
//                  int*,                     csrColIndA,
//                  int*,                     P,
//                  void*,                    pBuffer);

// SPARSE_API_DIRECT(Xcscsort_bufferSizeExt, please_write,  
//                                int,              m,
//                                int,              n,
//                                int,              nnz,
//                                const int*,       cscColPtrA,
//                                const int*,       cscRowIndA,
//                                size_t*,          pBufferSizeInBytes);

// SPARSE_API_DIRECT(Xcscsort, please_write,    
//                  int,                      m,
//                  int,                      n,
//                  int,                      nnz,
//                  const cusparseMatDescr_t, descrA,
//                  const int*,               cscColPtrA,
//                  int*,                     cscRowIndA,
//                  int*,                     P,
//                  void*,                    pBuffer);

// SPARSE_API_DIRECT(Scsru2csr_bufferSizeExt, please_write, 
//                                 int,              m,
//                                 int,              n,
//                                 int,              nnz,
//                                 float*,           csrVal,
//                                 const int*,       csrRowPtr,
//                                 int*,             csrColInd,
//                                 csru2csrInfo_t,   info,
//                                 size_t*,          pBufferSizeInBytes);

// SPARSE_API_DIRECT(Dcsru2csr_bufferSizeExt, please_write, 
//                                 int,              m,
//                                 int,              n,
//                                 int,              nnz,
//                                 double*,          csrVal,
//                                 const int*,       csrRowPtr,
//                                 int*,             csrColInd,
//                                 csru2csrInfo_t,   info,
//                                 size_t*,          pBufferSizeInBytes);

// SPARSE_API_DIRECT(Ccsru2csr_bufferSizeExt, please_write, 
//                                 int,              m,
//                                 int,              n,
//                                 int,              nnz,
//                                 cuComplex*,       csrVal,
//                                 const int*,       csrRowPtr,
//                                 int*,             csrColInd,
//                                 csru2csrInfo_t,   info,
//                                 size_t*,          pBufferSizeInBytes);

// SPARSE_API_DIRECT(Zcsru2csr_bufferSizeExt, please_write, 
//                                 int,              m,
//                                 int,              n,
//                                 int,              nnz,
//                                 cuDoubleComplex*, csrVal,
//                                 const int*,       csrRowPtr,
//                                 int*,             csrColInd,
//                                 csru2csrInfo_t,   info,
//                                 size_t*,          pBufferSizeInBytes);

// SPARSE_API_DIRECT(Scsru2csr, please_write,   
//                   int,                      m,
//                   int,                      n,
//                   int,                      nnz,
//                   const cusparseMatDescr_t, descrA,
//                   float*,                   csrVal,
//                   const int*,               csrRowPtr,
//                   int*,                     csrColInd,
//                   csru2csrInfo_t,           info,
//                   void*,                    pBuffer);

// SPARSE_API_DIRECT(Dcsru2csr, please_write,   
//                   int,                      m,
//                   int,                      n,
//                   int,                      nnz,
//                   const cusparseMatDescr_t, descrA,
//                   double*,                  csrVal,
//                   const int*,               csrRowPtr,
//                   int*,                     csrColInd,
//                   csru2csrInfo_t,           info,
//                   void*,                    pBuffer);

// SPARSE_API_DIRECT(Ccsru2csr, please_write,   
//                   int,                      m,
//                   int,                      n,
//                   int,                      nnz,
//                   const cusparseMatDescr_t, descrA,
//                   cuComplex*,               csrVal,
//                   const int*,               csrRowPtr,
//                   int*,                     csrColInd,
//                   csru2csrInfo_t,           info,
//                   void*,                    pBuffer);

// SPARSE_API_DIRECT(Zcsru2csr, please_write,   
//                   int,                      m,
//                   int,                      n,
//                   int,                      nnz,
//                   const cusparseMatDescr_t, descrA,
//                   cuDoubleComplex*,         csrVal,
//                   const int*,               csrRowPtr,
//                   int*,                     csrColInd,
//                   csru2csrInfo_t,           info,
//                   void*,                    pBuffer);

// SPARSE_API_DIRECT(Scsr2csru, please_write,   
//                   int,                      m,
//                   int,                      n,
//                   int,                      nnz,
//                   const cusparseMatDescr_t, descrA,
//                   float*,                   csrVal,
//                   const int*,               csrRowPtr,
//                   int*,                     csrColInd,
//                   csru2csrInfo_t,           info,
//                   void*,                    pBuffer);

// SPARSE_API_DIRECT(Dcsr2csru, please_write,   
//                   int,                      m,
//                   int,                      n,
//                   int,                      nnz,
//                   const cusparseMatDescr_t, descrA,
//                   double*,                  csrVal,
//                   const int*,               csrRowPtr,
//                   int*,                     csrColInd,
//                   csru2csrInfo_t,           info,
//                   void*,                    pBuffer);

// SPARSE_API_DIRECT(Ccsr2csru, please_write,   
//                   int,                      m,
//                   int,                      n,
//                   int,                      nnz,
//                   const cusparseMatDescr_t, descrA,
//                   cuComplex*,               csrVal,
//                   const int*,               csrRowPtr,
//                   int*,                     csrColInd,
//                   csru2csrInfo_t,           info,
//                   void*,                    pBuffer);

// SPARSE_API_DIRECT(Zcsr2csru, please_write,   
//                   int,                      m,
//                   int,                      n,
//                   int,                      nnz,
//                   const cusparseMatDescr_t, descrA,
//                   cuDoubleComplex*,         csrVal,
//                   const int*,               csrRowPtr,
//                   int*,                     csrColInd,
//                   csru2csrInfo_t,           info,
//                   void*,                    pBuffer);

//if defined(__cplusplus)
// SPARSE_API_DIRECT(HpruneDense2csr_bufferSizeExt, please_write,   
//                                       int,                      m,
//                                       int,                      n,
//                                       const __half*,            A,
//                                       int,                      lda,
//                                       const __half*,            threshold,
//                                       const cusparseMatDescr_t, descrC,
//                                       const __half*,            csrSortedValC,
//                                       const int*,               csrSortedRowPtrC,
//                                       const int*,               csrSortedColIndC,
//                                       size_t*, pBufferSizeInBytes);
//endif // defined(__cplusplus)

// SPARSE_API_DIRECT(SpruneDense2csr_bufferSizeExt, please_write,   
//                                       int,                      m,
//                                       int,                      n,
//                                       const float*,             A,
//                                       int,                      lda,
//                                       const float*,             threshold,
//                                       const cusparseMatDescr_t, descrC,
//                                       const float*,             csrSortedValC,
//                                       const int*,               csrSortedRowPtrC,
//                                       const int*,               csrSortedColIndC,
//                                       size_t*, pBufferSizeInBytes);

// SPARSE_API_DIRECT(DpruneDense2csr_bufferSizeExt, please_write,   
//                                       int,                      m,
//                                       int,                      n,
//                                       const double*,            A,
//                                       int,                      lda,
//                                       const double*,            threshold,
//                                       const cusparseMatDescr_t, descrC,
//                                       const double*,            csrSortedValC,
//                                       const int*,               csrSortedRowPtrC,
//                                       const int*,               csrSortedColIndC,
//                                       size_t*,               pBufferSizeInBytes);

//if defined(__cplusplus)
// SPARSE_API_DIRECT(HpruneDense2csrNnz, please_write,  
//                            int,                      m,
//                            int,                      n,
//                            const __half*,            A,
//                            int,                      lda,
//                            const __half*,            threshold,
//                            const cusparseMatDescr_t, descrC,
//                            int*,                     csrRowPtrC,
//                            int*,                     nnzTotalDevHostPtr,
//                            void*,                    pBuffer);
//endif // defined(__cplusplus)

// SPARSE_API_DIRECT(SpruneDense2csrNnz, please_write,  
//                            int,                      m,
//                            int,                      n,
//                            const float*,             A,
//                            int,                      lda,
//                            const float*,             threshold,
//                            const cusparseMatDescr_t, descrC,
//                            int*,                     csrRowPtrC,
//                            int*,                     nnzTotalDevHostPtr,
//                            void*,                    pBuffer);

// SPARSE_API_DIRECT(DpruneDense2csrNnz, please_write,  
//                            int,                      m,
//                            int,                      n,
//                            const double*,            A,
//                            int,                      lda,
//                            const double*,            threshold,
//                            const cusparseMatDescr_t, descrC,
//                            int*,                     csrSortedRowPtrC,
//                            int*,                     nnzTotalDevHostPtr,
//                            void*,                    pBuffer);

//if defined(__cplusplus)
// SPARSE_API_DIRECT(HpruneDense2csr, please_write, 
//                         int,                      m,
//                         int,                      n,
//                         const __half*,            A,
//                         int,                      lda,
//                         const __half*,            threshold,
//                         const cusparseMatDescr_t, descrC,
//                         __half*,                  csrSortedValC,
//                         const int*,               csrSortedRowPtrC,
//                         int*,                     csrSortedColIndC,
//                         void*,                    pBuffer);
//endif // defined(__cplusplus)

// SPARSE_API_DIRECT(SpruneDense2csr, please_write, 
//                         int,                      m,
//                         int,                      n,
//                         const float*,             A,
//                         int,                      lda,
//                         const float*,             threshold,
//                         const cusparseMatDescr_t, descrC,
//                         float*,                   csrSortedValC,
//                         const int*,               csrSortedRowPtrC,
//                         int*,                     csrSortedColIndC,
//                         void*,                    pBuffer);

// SPARSE_API_DIRECT(DpruneDense2csr, please_write, 
//                         int,                      m,
//                         int,                      n,
//                         const double*,            A,
//                         int,                      lda,
//                         const double*,            threshold,
//                         const cusparseMatDescr_t, descrC,
//                         double*,                  csrSortedValC,
//                         const int*,               csrSortedRowPtrC,
//                         int*,                     csrSortedColIndC,
//                         void*,                    pBuffer);

//if defined(__cplusplus)
// SPARSE_API_DIRECT(HpruneCsr2csr_bufferSizeExt, please_write, 
//                                     int,                      m,
//                                     int,                      n,
//                                     int,                      nnzA,
//                                     const cusparseMatDescr_t, descrA,
//                                     const __half*,            csrSortedValA,
//                                     const int*,               csrSortedRowPtrA,
//                                     const int*,               csrSortedColIndA,
//                                     const __half*,            threshold,
//                                     const cusparseMatDescr_t, descrC,
//                                     const __half*,            csrSortedValC,
//                                     const int*,               csrSortedRowPtrC,
//                                     const int*,               csrSortedColIndC,
//                                     size_t*, pBufferSizeInBytes);
//endif // defined(__cplusplus)

// SPARSE_API_DIRECT(SpruneCsr2csr_bufferSizeExt, please_write, 
//                                     int,                      m,
//                                     int,                      n,
//                                     int,                      nnzA,
//                                     const cusparseMatDescr_t, descrA,
//                                     const float*,             csrSortedValA,
//                                     const int*,               csrSortedRowPtrA,
//                                     const int*,               csrSortedColIndA,
//                                     const float*,             threshold,
//                                     const cusparseMatDescr_t, descrC,
//                                     const float*,             csrSortedValC,
//                                     const int*,               csrSortedRowPtrC,
//                                     const int*,               csrSortedColIndC,
//                                     size_t*,                 pBufferSizeInBytes);

// SPARSE_API_DIRECT(DpruneCsr2csr_bufferSizeExt, please_write, 
//                                     int,                      m,
//                                     int,                      n,
//                                     int,                      nnzA,
//                                     const cusparseMatDescr_t, descrA,
//                                     const double*,            csrSortedValA,
//                                     const int*,               csrSortedRowPtrA,
//                                     const int*,               csrSortedColIndA,
//                                     const double*,            threshold,
//                                     const cusparseMatDescr_t, descrC,
//                                     const double*,            csrSortedValC,
//                                     const int*,               csrSortedRowPtrC,
//                                     const int*,               csrSortedColIndC,
//                                     size_t*,                 pBufferSizeInBytes);

//if defined(__cplusplus)
// SPARSE_API_DIRECT(HpruneCsr2csrNnz, please_write,    
//                          int,                      m,
//                          int,                      n,
//                          int,                      nnzA,
//                          const cusparseMatDescr_t, descrA,
//                          const __half*,            csrSortedValA,
//                          const int*,               csrSortedRowPtrA,
//                          const int*,               csrSortedColIndA,
//                          const __half*,            threshold,
//                          const cusparseMatDescr_t, descrC,
//                          int*,                     csrSortedRowPtrC,
//                          int*,                     nnzTotalDevHostPtr,
//                          void*,                    pBuffer);
//endif // defined(__cplusplus)

// SPARSE_API_DIRECT(SpruneCsr2csrNnz, please_write,    
//                          int,                      m,
//                          int,                      n,
//                          int,                      nnzA,
//                          const cusparseMatDescr_t, descrA,
//                          const float*,             csrSortedValA,
//                          const int*,               csrSortedRowPtrA,
//                          const int*,               csrSortedColIndA,
//                          const float*,             threshold,
//                          const cusparseMatDescr_t, descrC,
//                          int*,                     csrSortedRowPtrC,
//                          int*,                     nnzTotalDevHostPtr,
//                          void*,                    pBuffer);

// GPUSPARSE_EXPORT_C cusparseStatus_t
//  cusparseDpruneCsr2csrNnz(cusparseHandle_t         handle,
//                           int,                      m,
//                           int,                      n,
//                           int,                      nnzA,
//                           const cusparseMatDescr_t, descrA,
//                           const double*,            csrSortedValA,
//                           const int*,               csrSortedRowPtrA,
//                           const int*,               csrSortedColIndA,
//                           const double*,            threshold,
//                           const cusparseMatDescr_t, descrC,
//                           int*,                     csrSortedRowPtrC,
//                           int*,                     nnzTotalDevHostPtr,
//                           void*,                    pBuffer);

//if defined(__cplusplus)
// SPARSE_API_DIRECT(HpruneCsr2csr, please_write,   
//                       int,                      m,
//                       int,                      n,
//                       int,                      nnzA,
//                       const cusparseMatDescr_t, descrA,
//                       const __half*,            csrSortedValA,
//                       const int*,               csrSortedRowPtrA,
//                       const int*,               csrSortedColIndA,
//                       const __half*,            threshold,
//                       const cusparseMatDescr_t, descrC,
//                       __half*,                  csrSortedValC,
//                       const int*,               csrSortedRowPtrC,
//                       int*,                     csrSortedColIndC,
//                       void*,                    pBuffer);
//endif // defined(__cplusplus)

// SPARSE_API_DIRECT(SpruneCsr2csr, please_write,   
//                       int,                      m,
//                       int,                      n,
//                       int,                      nnzA,
//                       const cusparseMatDescr_t, descrA,
//                       const float*,             csrSortedValA,
//                       const int*,               csrSortedRowPtrA,
//                       const int*,               csrSortedColIndA,
//                       const float*,             threshold,
//                       const cusparseMatDescr_t, descrC,
//                       float*,                   csrSortedValC,
//                       const int*,               csrSortedRowPtrC,
//                       int*,                     csrSortedColIndC,
//                       void*,                    pBuffer);

// SPARSE_API_DIRECT(DpruneCsr2csr, please_write,   
//                       int,                      m,
//                       int,                      n,
//                       int,                      nnzA,
//                       const cusparseMatDescr_t, descrA,
//                       const double*,            csrSortedValA,
//                       const int*,               csrSortedRowPtrA,
//                       const int*,               csrSortedColIndA,
//                       const double*,            threshold,
//                       const cusparseMatDescr_t, descrC,
//                       double*,                  csrSortedValC,
//                       const int*,               csrSortedRowPtrC,
//                       int*,                     csrSortedColIndC,
//                       void*,                    pBuffer);

//if defined(__cplusplus)
// SPARSE_API_DIRECT_NOHANDLE(HpruneDense2csrByPercentage_bufferSizeExt, please_Write_me,
    
//                                    cusparseHandle_t,         handle,
//                                    int,                      m,
//                                    int,                      n,
//                                    const __half*,            A,
//                                    int,                      lda,
//                                    float,                    percentage,
//                                    const cusparseMatDescr_t, descrC,
//                                    const __half*,            csrSortedValC,
//                                    const int*,               csrSortedRowPtrC,
//                                    const int*,               csrSortedColIndC,
//                                    pruneInfo_t,              info,
//                                    size_t*,                  pBufferSizeInBytes);
//endif // defined(__cplusplus)

// SPARSE_API_DIRECT_NOHANDLE(SpruneDense2csrByPercentage_bufferSizeExt, please_Write_me,
    
//                                    cusparseHandle_t,         handle,
//                                    int,                      m,
//                                    int,                      n,
//                                    const float*,             A,
//                                    int,                      lda,
//                                    float,                    percentage,
//                                    const cusparseMatDescr_t, descrC,
//                                    const float*,             csrSortedValC,
//                                    const int*,               csrSortedRowPtrC,
//                                    const int*,               csrSortedColIndC,
//                                    pruneInfo_t,              info,
//                                    size_t*,                  pBufferSizeInBytes);

// SPARSE_API_DIRECT_NOHANDLE(DpruneDense2csrByPercentage_bufferSizeExt, please_Write_me,
    
//                                    cusparseHandle_t,         handle,
//                                    int,                      m,
//                                    int,                      n,
//                                    const double*,            A,
//                                    int,                      lda,
//                                    float,                    percentage,
//                                    const cusparseMatDescr_t, descrC,
//                                    const double*,            csrSortedValC,
//                                    const int*,               csrSortedRowPtrC,
//                                    const int*,               csrSortedColIndC,
//                                    pruneInfo_t,              info,
//                                    size_t*,                  pBufferSizeInBytes);

//if defined(__cplusplus)
// SPARSE_API_DIRECT_NOHANDLE(HpruneDense2csrNnzByPercentage, please_Write_me,
    
//                                     cusparseHandle_t,         handle,
//                                     int,                      m,
//                                     int,                      n,
//                                     const __half*,            A,
//                                     int,                      lda,
//                                     float,                    percentage,
//                                     const cusparseMatDescr_t, descrC,
//                                     int*,                     csrRowPtrC,
//                                     int*,                     nnzTotalDevHostPtr,
//                                     pruneInfo_t,              info,
//                                     void*,                    pBuffer);
//endif // defined(__cplusplus)

// SPARSE_API_DIRECT_NOHANDLE(SpruneDense2csrNnzByPercentage, please_Write_me,
    
//                                     cusparseHandle_t,         handle,
//                                     int,                      m,
//                                     int,                      n,
//                                     const float*,             A,
//                                     int,                      lda,
//                                     float,                    percentage,
//                                     const cusparseMatDescr_t, descrC,
//                                     int*,                     csrRowPtrC,
//                                     int*,                     nnzTotalDevHostPtr,
//                                     pruneInfo_t,              info,
//                                     void*,                    pBuffer);

// SPARSE_API_DIRECT_NOHANDLE(DpruneDense2csrNnzByPercentage, please_Write_me,
    
//                                     cusparseHandle_t,         handle,
//                                     int,                      m,
//                                     int,                      n,
//                                     const double*,            A,
//                                     int,                      lda,
//                                     float,                    percentage,
//                                     const cusparseMatDescr_t, descrC,
//                                     int*,                     csrRowPtrC,
//                                     int*,                     nnzTotalDevHostPtr,
//                                     pruneInfo_t,              info,
//                                     void*,                    pBuffer);

//if defined(__cplusplus)
// SPARSE_API_DIRECT(HpruneDense2csrByPercentage, please_write, 
//                                     int,                      m,
//                                     int,                      n,
//                                     const __half*,            A,
//                                     int,                      lda,
//                                     float,                    percentage,
//                                     const cusparseMatDescr_t, descrC,
//                                     __half*,                  csrSortedValC,
//                                     const int*,               csrSortedRowPtrC,
//                                     int*,                     csrSortedColIndC,
//                                     pruneInfo_t,              info,
//                                     void*,                    pBuffer);
//endif // defined(__cplusplus)

// SPARSE_API_DIRECT(SpruneDense2csrByPercentage, please_write, 
//                                     int,                      m,
//                                     int,                      n,
//                                     const float*,             A,
//                                     int,                      lda,
//                                     float,                    percentage,
//                                     const cusparseMatDescr_t, descrC,
//                                     float*,                   csrSortedValC,
//                                     const int*,               csrSortedRowPtrC,
//                                     int*,                     csrSortedColIndC,
//                                     pruneInfo_t,              info,
//                                     void*,                    pBuffer);

// SPARSE_API_DIRECT(DpruneDense2csrByPercentage, please_write, 
//                                     int,                      m,
//                                     int,                      n,
//                                     const double*,            A,
//                                     int,                      lda,
//                                     float,                    percentage,
//                                     const cusparseMatDescr_t, descrC,
//                                     double*,                  csrSortedValC,
//                                     const int*,               csrSortedRowPtrC,
//                                     int*,                     csrSortedColIndC,
//                                     pruneInfo_t,              info,
//                                     void*,                    pBuffer);

//if defined(__cplusplus)

// SPARSE_API_DIRECT_NOHANDLE(HpruneCsr2csrByPercentage_bufferSizeExt, please_Write_me,
    
//                                    cusparseHandle_t,         handle,
//                                    int,                      m,
//                                    int,                      n,
//                                    int,                      nnzA,
//                                    const cusparseMatDescr_t, descrA,
//                                    const __half*,            csrSortedValA,
//                                    const int*,               csrSortedRowPtrA,
//                                    const int*,               csrSortedColIndA,
//                                    float,                    percentage,
//                                    const cusparseMatDescr_t, descrC,
//                                    const __half*,            csrSortedValC,
//                                    const int*,               csrSortedRowPtrC,
//                                    const int*,               csrSortedColIndC,
//                                    pruneInfo_t,              info,
//                                    size_t*,                  pBufferSizeInBytes);

//endif // defined(__cplusplus)

// SPARSE_API_DIRECT_NOHANDLE(SpruneCsr2csrByPercentage_bufferSizeExt, please_Write_me,
    
//                                    cusparseHandle_t,         handle,
//                                    int,                      m,
//                                    int,                      n,
//                                    int,                      nnzA,
//                                    const cusparseMatDescr_t, descrA,
//                                    const float*,             csrSortedValA,
//                                    const int*,               csrSortedRowPtrA,
//                                    const int*,               csrSortedColIndA,
//                                    float,                    percentage,
//                                    const cusparseMatDescr_t, descrC,
//                                    const float*,             csrSortedValC,
//                                    const int*,               csrSortedRowPtrC,
//                                    const int*,               csrSortedColIndC,
//                                    pruneInfo_t,              info,
//                                    size_t*,                  pBufferSizeInBytes);

// SPARSE_API_DIRECT_NOHANDLE(DpruneCsr2csrByPercentage_bufferSizeExt, please_Write_me,
    
//                                    cusparseHandle_t,         handle,
//                                    int,                      m,
//                                    int,                      n,
//                                    int,                      nnzA,
//                                    const cusparseMatDescr_t, descrA,
//                                    const double*,            csrSortedValA,
//                                    const int*,               csrSortedRowPtrA,
//                                    const int*,               csrSortedColIndA,
//                                    float,                    percentage,
//                                    const cusparseMatDescr_t, descrC,
//                                    const double*,            csrSortedValC,
//                                    const int*,               csrSortedRowPtrC,
//                                    const int*,               csrSortedColIndC,
//                                    pruneInfo_t,              info,
//                                    size_t*,                  pBufferSizeInBytes);

//if defined(__cplusplus)

// SPARSE_API_DIRECT_NOHANDLE(HpruneCsr2csrNnzByPercentage, please_Write_me,
    
//                                     cusparseHandle_t,         handle,
//                                     int,                      m,
//                                     int,                      n,
//                                     int,                      nnzA,
//                                     const cusparseMatDescr_t, descrA,
//                                     const __half*,            csrSortedValA,
//                                     const int*,               csrSortedRowPtrA,
//                                     const int*,               csrSortedColIndA,
//                                     float,                    percentage,
//                                     const cusparseMatDescr_t, descrC,
//                                     int*,                     csrSortedRowPtrC,
//                                     int*,                     nnzTotalDevHostPtr,
//                                     pruneInfo_t,              info,
//                                     void*,                    pBuffer);

//endif // defined(__cplusplus)

// SPARSE_API_DIRECT_NOHANDLE(SpruneCsr2csrNnzByPercentage, please_Write_me,
    
//                                     cusparseHandle_t,         handle,
//                                     int,                      m,
//                                     int,                      n,
//                                     int,                      nnzA,
//                                     const cusparseMatDescr_t, descrA,
//                                     const float*,             csrSortedValA,
//                                     const int*,               csrSortedRowPtrA,
//                                     const int*,               csrSortedColIndA,
//                                     float,                    percentage,
//                                     const cusparseMatDescr_t, descrC,
//                                     int*,                     csrSortedRowPtrC,
//                                     int*,                     nnzTotalDevHostPtr,
//                                     pruneInfo_t,              info,
//                                     void*,                    pBuffer);

// SPARSE_API_DIRECT_NOHANDLE(DpruneCsr2csrNnzByPercentage, please_Write_me,
    
//                                     cusparseHandle_t,         handle,
//                                     int,                      m,
//                                     int,                      n,
//                                     int,                      nnzA,
//                                     const cusparseMatDescr_t, descrA,
//                                     const double*,            csrSortedValA,
//                                     const int*,               csrSortedRowPtrA,
//                                     const int*,               csrSortedColIndA,
//                                     float,                    percentage,
//                                     const cusparseMatDescr_t, descrC,
//                                     int*,                     csrSortedRowPtrC,
//                                     int*,                     nnzTotalDevHostPtr,
//                                     pruneInfo_t,              info,
//                                     void*,                    pBuffer);

//if defined(__cplusplus)
// SPARSE_API_DIRECT(HpruneCsr2csrByPercentage, please_write,   
//                                   int,                      m,
//                                   int,                      n,
//                                   int,                      nnzA,
//                                   const cusparseMatDescr_t, descrA,
//                                   const __half*,            csrSortedValA,
//                                   const int*,               csrSortedRowPtrA,
//                                   const int*,               csrSortedColIndA,
//                                   float percentage, /* between 0 to 100 */
//                                   const cusparseMatDescr_t, descrC,
//                                   __half*,                  csrSortedValC,
//                                   const int*,               csrSortedRowPtrC,
//                                   int*,                     csrSortedColIndC,
//                                   pruneInfo_t,              info,
//                                   void*,                    pBuffer);

//endif // defined(__cplusplus)

// SPARSE_API_DIRECT(SpruneCsr2csrByPercentage, please_write,   
//                                   int,                      m,
//                                   int,                      n,
//                                   int,                      nnzA,
//                                   const cusparseMatDescr_t, descrA,
//                                   const float*,             csrSortedValA,
//                                   const int*,               csrSortedRowPtrA,
//                                   const int*,               csrSortedColIndA,
//                                   float,                    percentage,
//                                   const cusparseMatDescr_t, descrC,
//                                   float*,                   csrSortedValC,
//                                   const int*,               csrSortedRowPtrC,
//                                   int*,                     csrSortedColIndC,
//                                   pruneInfo_t,              info,
//                                   void*,                    pBuffer);

// SPARSE_API_DIRECT(DpruneCsr2csrByPercentage, please_write,   
//                                   int,                      m,
//                                   int,                      n,
//                                   int,                      nnzA,
//                                   const cusparseMatDescr_t, descrA,
//                                   const double*,            csrSortedValA,
//                                   const int*,               csrSortedRowPtrA,
//                                   const int*,               csrSortedColIndA,
//                                   float,                    percentage,
//                                   const cusparseMatDescr_t, descrC,
//                                   double*,                  csrSortedValC,
//                                   const int*,               csrSortedRowPtrC,
//                                   int*,                     csrSortedColIndC,
//                                   pruneInfo_t,              info,
//                                   void*,                    pBuffer);

// // 
// // CSR2CSC
// // 

#define csr2csc(ROCLETTER, TYPE) \
    rocsparse_## ROCLETTER ##csr2csc(cuToRoc(handle), m, n, nnz, \
        reinterpret_cast<const TYPE*>(csrVal), csrRowPtr, csrColInd, \
        reinterpret_cast<TYPE*>(cscVal), cscRowInd, cscColPtr, \
        cuToRoc(copyValues), cuToRoc(idxBase), buffer)

GPUSPARSE_EXPORT_C cusparseStatus_t
cusparseCsr2cscEx2(cusparseHandle_t     handle,
                   int                  m,
                   int                  n,
                   int                  nnz,
                   const void*          csrVal,
                   const int*           csrRowPtr,
                   const int*           csrColInd,
                   void*                cscVal,
                   int*                 cscColPtr,
                   int*                 cscRowInd,
                   cudaDataType         valType,
                   cusparseAction_t     copyValues,
                   cusparseIndexBase_t  idxBase,
                   cusparseCsr2CscAlg_t alg,
                   void*                buffer)
    INLINE_BODY_STATUS(
    (void)alg;
    switch(valType) {
        case CUDA_R_32F:
            return csr2csc(s, float);
        case CUDA_R_64F:
            return csr2csc(d, double);
        case CUDA_C_32F:
            return csr2csc(c, rocsparse_float_complex);
        case CUDA_C_64F:
            return csr2csc(z, rocsparse_double_complex);
        default:
            return rocsparse_status_invalid_value;
    }
)

#undef csr2csc

GPUSPARSE_EXPORT_C cusparseStatus_t
cusparseCsr2cscEx2_bufferSize(cusparseHandle_t     handle,
                              int                  m,
                              int                  n,
                              int                  nnz,
                              const void*          csrVal,
                              const int*           csrRowPtr,
                              const int*           csrColInd,
                              void*                cscVal,
                              int*                 cscColPtr,
                              int*                 cscRowInd,
                              cudaDataType         valType,
                              cusparseAction_t     copyValues,
                              cusparseIndexBase_t  idxBase,
                              cusparseCsr2CscAlg_t alg,
                              size_t*              bufferSize)
    INLINE_BODY_STATUS(
        (void)csrVal;
        (void)cscVal; (void)cscColPtr; (void)cscRowInd;
        (void)valType; (void)idxBase; (void)alg;
    return rocsparse_csr2csc_buffer_size(cuToRoc(handle), m, n, nnz,
        csrRowPtr, csrColInd, cuToRoc(copyValues), bufferSize);
)

// SPARSE_API_DIRECT(Csr2cscEx2_bufferSize, csr2csc_buffer_size,  
//     int,                  m,
//     int,                  n,
//     int,                  nnz,
//     const void*,          csrVal,
//     const int*,           csrRowPtr,
//     const int*,           csrColInd,
//     void*,                cscVal,
//     int*,                 cscColPtr,
//     int*,                 cscRowInd,
//     cudaDataType,         valType,
//     cusparseAction_t,     copyValues,
//     cusparseIndexBase_t,  idxBase,
//     cusparseCsr2CscAlg_t, alg,
//     size_t*,              bufferSize);

