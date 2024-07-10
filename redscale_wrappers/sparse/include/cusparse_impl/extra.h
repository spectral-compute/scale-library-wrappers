
#include "common.h"

/// Extra stuff

// SPARSE_API_DIRECT(Scsrgeam2_bufferSizeExt, please_write, 
//                                 int,                      m,
//                                 int,                      n,
//                                 const float*,             alpha,
//                                 const cusparseMatDescr_t, descrA,
//                                 int,                      nnzA,
//                                 const float*,             csrSortedValA,
//                                 const int*,               csrSortedRowPtrA,
//                                 const int*,               csrSortedColIndA,
//                                 const float*,             beta,
//                                 const cusparseMatDescr_t, descrB,
//                                 int,                      nnzB,
//                                 const float*,             csrSortedValB,
//                                 const int*,               csrSortedRowPtrB,
//                                 const int*,               csrSortedColIndB,
//                                 const cusparseMatDescr_t, descrC,
//                                 const float*,             csrSortedValC,
//                                 const int*,               csrSortedRowPtrC,
//                                 const int*,               csrSortedColIndC,
//                                 size_t*,                  pBufferSizeInBytes);

// SPARSE_API_DIRECT(Dcsrgeam2_bufferSizeExt, please_write, 
//                                 int,                      m,
//                                 int,                      n,
//                                 const double*,            alpha,
//                                 const cusparseMatDescr_t, descrA,
//                                 int,                      nnzA,
//                                 const double*,            csrSortedValA,
//                                 const int*,               csrSortedRowPtrA,
//                                 const int*,               csrSortedColIndA,
//                                 const double*,            beta,
//                                 const cusparseMatDescr_t, descrB,
//                                 int,                      nnzB,
//                                 const double*,            csrSortedValB,
//                                 const int*,               csrSortedRowPtrB,
//                                 const int*,               csrSortedColIndB,
//                                 const cusparseMatDescr_t, descrC,
//                                 const double*,            csrSortedValC,
//                                 const int*,               csrSortedRowPtrC,
//                                 const int*,               csrSortedColIndC,
//                                 size_t*,                  pBufferSizeInBytes);

// SPARSE_API_DIRECT(Ccsrgeam2_bufferSizeExt, please_write, 
//                                 int,                      m,
//                                 int,                      n,
//                                 const cuComplex*,         alpha,
//                                 const cusparseMatDescr_t, descrA,
//                                 int,                      nnzA,
//                                 const cuComplex*,         csrSortedValA,
//                                 const int*,               csrSortedRowPtrA,
//                                 const int*,               csrSortedColIndA,
//                                 const cuComplex*,         beta,
//                                 const cusparseMatDescr_t, descrB,
//                                 int,                      nnzB,
//                                 const cuComplex*,         csrSortedValB,
//                                 const int*,               csrSortedRowPtrB,
//                                 const int*,               csrSortedColIndB,
//                                 const cusparseMatDescr_t, descrC,
//                                 const cuComplex*,         csrSortedValC,
//                                 const int*,               csrSortedRowPtrC,
//                                 const int*,               csrSortedColIndC,
//                                 size_t*,                  pBufferSizeInBytes);

// SPARSE_API_DIRECT(Zcsrgeam2_bufferSizeExt, please_write, 
//                                 int,                      m,
//                                 int,                      n,
//                                 const cuDoubleComplex*,   alpha,
//                                 const cusparseMatDescr_t, descrA,
//                                 int,                      nnzA,
//                                 const cuDoubleComplex*,   csrSortedValA,
//                                 const int*,               csrSortedRowPtrA,
//                                 const int*,               csrSortedColIndA,
//                                 const cuDoubleComplex*,   beta,
//                                 const cusparseMatDescr_t, descrB,
//                                 int,                      nnzB,
//                                 const cuDoubleComplex*,   csrSortedValB,
//                                 const int*,               csrSortedRowPtrB,
//                                 const int*,               csrSortedColIndB,
//                                 const cusparseMatDescr_t, descrC,
//                                 const cuDoubleComplex*,   csrSortedValC,
//                                 const int*,               csrSortedRowPtrC,
//                                 const int*,               csrSortedColIndC,
//                                 size_t*,                  pBufferSizeInBytes);

// SPARSE_API_DIRECT(Xcsrgeam2Nnz, please_write,    
//                      int,                      m,
//                      int,                      n,
//                      const cusparseMatDescr_t, descrA,
//                      int,                      nnzA,
//                      const int*,               csrSortedRowPtrA,
//                      const int*,               csrSortedColIndA,
//                      const cusparseMatDescr_t, descrB,
//                      int,                      nnzB,
//                      const int*,               csrSortedRowPtrB,
//                      const int*,               csrSortedColIndB,
//                      const cusparseMatDescr_t, descrC,
//                      int*,                     csrSortedRowPtrC,
//                      int*,                     nnzTotalDevHostPtr,
//                      void*,                    workspace);

// SPARSE_API_DIRECT(Scsrgeam2, please_write,   
//                   int,                      m,
//                   int,                      n,
//                   const float*,             alpha,
//                   const cusparseMatDescr_t, descrA,
//                   int,                      nnzA,
//                   const float*,             csrSortedValA,
//                   const int*,               csrSortedRowPtrA,
//                   const int*,               csrSortedColIndA,
//                   const float*,             beta,
//                   const cusparseMatDescr_t, descrB,
//                   int,                      nnzB,
//                   const float*,             csrSortedValB,
//                   const int*,               csrSortedRowPtrB,
//                   const int*,               csrSortedColIndB,
//                   const cusparseMatDescr_t, descrC,
//                   float*,                   csrSortedValC,
//                   int*,                     csrSortedRowPtrC,
//                   int*,                     csrSortedColIndC,
//                   void*,                    pBuffer);

// SPARSE_API_DIRECT(Dcsrgeam2, please_write,   
//                   int,                      m,
//                   int,                      n,
//                   const double*,            alpha,
//                   const cusparseMatDescr_t, descrA,
//                   int,                      nnzA,
//                   const double*,            csrSortedValA,
//                   const int*,               csrSortedRowPtrA,
//                   const int*,               csrSortedColIndA,
//                   const double*,            beta,
//                   const cusparseMatDescr_t, descrB,
//                   int,                      nnzB,
//                   const double*,            csrSortedValB,
//                   const int*,               csrSortedRowPtrB,
//                   const int*,               csrSortedColIndB,
//                   const cusparseMatDescr_t, descrC,
//                   double*,                  csrSortedValC,
//                   int*,                     csrSortedRowPtrC,
//                   int*,                     csrSortedColIndC,
//                   void*,                    pBuffer);

// SPARSE_API_DIRECT(Ccsrgeam2, please_write,   
//                   int,                      m,
//                   int,                      n,
//                   const cuComplex*,         alpha,
//                   const cusparseMatDescr_t, descrA,
//                   int,                      nnzA,
//                   const cuComplex*,         csrSortedValA,
//                   const int*,               csrSortedRowPtrA,
//                   const int*,               csrSortedColIndA,
//                   const cuComplex*,         beta,
//                   const cusparseMatDescr_t, descrB,
//                   int,                      nnzB,
//                   const cuComplex*,         csrSortedValB,
//                   const int*,               csrSortedRowPtrB,
//                   const int*,               csrSortedColIndB,
//                   const cusparseMatDescr_t, descrC,
//                   cuComplex*,               csrSortedValC,
//                   int*,                     csrSortedRowPtrC,
//                   int*,                     csrSortedColIndC,
//                   void*,                    pBuffer);

// SPARSE_API_DIRECT(Zcsrgeam2, please_write,   
//                   int,                      m,
//                   int,                      n,
//                   const cuDoubleComplex*,   alpha,
//                   const cusparseMatDescr_t, descrA,
//                   int,                      nnzA,
//                   const cuDoubleComplex*,   csrSortedValA,
//                   const int*,               csrSortedRowPtrA,
//                   const int*,               csrSortedColIndA,
//                   const cuDoubleComplex*,   beta,
//                   const cusparseMatDescr_t, descrB,
//                   int,                      nnzB,
//                   const cuDoubleComplex*,   csrSortedValB,
//                   const int*,               csrSortedRowPtrB,
//                   const int*,               csrSortedColIndB,
//                   const cusparseMatDescr_t, descrC,
//                   cuDoubleComplex*,         csrSortedValC,
//                   int*,                     csrSortedRowPtrC,
//                   int*,                     csrSortedColIndC,
//                   void*,                    pBuffer);
