
#include "common.h"

/// Vector-Vector operations


// SPARSE_API_DIRECT(Axpby, please_write,   
//               const void*,               alpha,
//               cusparseConstSpVecDescr_t, vecX,
//               const void*,               beta,
//               cusparseDnVecDescr_t,      vecY);

// SPARSE_API_DIRECT(Gather, please_write,  
//                cusparseConstDnVecDescr_t, vecY,
//                cusparseSpVecDescr_t,      vecX);

// SPARSE_API_DIRECT(Scatter, please_write, 
//                 cusparseConstSpVecDescr_t, vecX,
//                 cusparseDnVecDescr_t,      vecY);

// SPARSE_API_DIRECT(Rot, please_write, 
//             const void*,          c_coeff,
//             const void*,          s_coeff,
//             cusparseSpVecDescr_t, vecX,
//             cusparseDnVecDescr_t, vecY);

// SPARSE_API_DIRECT(SpVV_bufferSize, please_write, 
//                         cusparseOperation_t,       opX,
//                         cusparseConstSpVecDescr_t, vecX,
//                         cusparseConstDnVecDescr_t, vecY,
//                         const void*,               result,
//                         cudaDataType,              computeType,
//                         size_t*,                   bufferSize);

// SPARSE_API_DIRECT(SpVV, please_write,    
//              cusparseOperation_t,       opX,
//              cusparseConstSpVecDescr_t, vecX,
//              cusparseConstDnVecDescr_t, vecY,
//              void*,                     result,
//              cudaDataType,              computeType,
//              void*,                     externalBuffer);

/// Sparse -> Dense

// SPARSE_API_DIRECT(SparseToDense_bufferSize, please_write,    
//                                  cusparseConstSpMatDescr_t,  matA,
//                                  cusparseDnMatDescr_t,       matB,
//                                  cusparseSparseToDenseAlg_t, alg,
//                                  size_t*,                    bufferSize);

// SPARSE_API_DIRECT(SparseToDense, please_write,   
//                       cusparseConstSpMatDescr_t,  matA,
//                       cusparseDnMatDescr_t,       matB,
//                       cusparseSparseToDenseAlg_t, alg,
//                       void*,                      externalBuffer);

/// Dense -> Sparse

// SPARSE_API_DIRECT(DenseToSparse_bufferSize, please_write,    
//                                  cusparseConstDnMatDescr_t,  matA,
//                                  cusparseSpMatDescr_t,       matB,
//                                  cusparseDenseToSparseAlg_t, alg,
//                                  size_t*,                    bufferSize);

// SPARSE_API_DIRECT(DenseToSparse_analysis, please_write,  
//                                cusparseConstDnMatDescr_t,  matA,
//                                cusparseSpMatDescr_t,       matB,
//                                cusparseDenseToSparseAlg_t, alg,
//                                void*,                      externalBuffer);

// SPARSE_API_DIRECT(DenseToSparse_convert, please_write,   
//                               cusparseConstDnMatDescr_t,  matA,
//                               cusparseSpMatDescr_t,       matB,
//                               cusparseDenseToSparseAlg_t, alg,
//                               void*,                      externalBuffer);
