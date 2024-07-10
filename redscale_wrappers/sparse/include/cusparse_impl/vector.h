
#include "common.h"

/// Sparse Vectors

// SPARSE_API_DIRECT_NOHANDLE(CreateSpVec, please_Write_me,
//  cusparseSpVecDescr_t* spVecDescr,
//                     int64_t,               size,
//                     int64_t,               nnz,
//                     void*,                 indices,
//                     void*,                 values,
//                     cusparseIndexType_t,   idxType,
//                     cusparseIndexBase_t,   idxBase,
//                     cudaDataType,          valueType);

// SPARSE_API_DIRECT_NOHANDLE(CreateConstSpVec, please_Write_me,
//  cusparseConstSpVecDescr_t* spVecDescr,
//                          int64_t,                    size,
//                          int64_t,                    nnz,
//                          const void*,                indices,
//                          const void*,                values,
//                          cusparseIndexType_t,        idxType,
//                          cusparseIndexBase_t,        idxBase,
//                          cudaDataType,               valueType);

// SPARSE_API_DIRECT_NOHANDLE(DestroySpVec, please_Write_me,
//  cusparseConstSpVecDescr_t spVecDescr);

// SPARSE_API_DIRECT_NOHANDLE(SpVecGet, please_Write_me,
//  cusparseSpVecDescr_t spVecDescr,
//                  int64_t*,             size,
//                  int64_t*,             nnz,
//                  void**,               indices,
//                  void**,               values,
//                  cusparseIndexType_t*, idxType,
//                  cusparseIndexBase_t*, idxBase,
//                  cudaDataType*,        valueType);

// SPARSE_API_DIRECT_NOHANDLE(ConstSpVecGet, please_Write_me,
//  cusparseConstSpVecDescr_t spVecDescr,
//                       int64_t*,             size,
//                       int64_t*,             nnz,
//                       const void**,         indices,
//                       const void**,         values,
//                       cusparseIndexType_t*, idxType,
//                       cusparseIndexBase_t*, idxBase,
//                       cudaDataType*,        valueType);

// SPARSE_API_DIRECT_NOHANDLE(SpVecGetIndexBase, please_Write_me,
//  cusparseConstSpVecDescr_t spVecDescr,
//                           cusparseIndexBase_t*,      idxBase);

// SPARSE_API_DIRECT_NOHANDLE(SpVecGetValues, please_Write_me,
//  cusparseSpVecDescr_t spVecDescr,
//                        void**,               values);

// SPARSE_API_DIRECT_NOHANDLE(ConstSpVecGetValues, please_Write_me,
//  cusparseConstSpVecDescr_t spVecDescr,
//                             const void**,              values);

// SPARSE_API_DIRECT_NOHANDLE(SpVecSetValues, please_Write_me,
//  cusparseSpVecDescr_t spVecDescr,
//                        void*,                values);

/// Dense Vectors

SPARSE_API_DIRECT_NOHANDLE(CreateDnVec, create_dnvec_descr,
    cusparseDnVecDescr_t*, dnVecDescr,
    int64_t,               size,
    void*,                 values,
    cudaDataType,          valueType);

// SPARSE_API_DIRECT_NOHANDLE(CreateConstDnVec, please_Write_me,
//  cusparseConstDnVecDescr_t* dnVecDescr,
//                          int64_t,                    size,
//                          const void*,                values,
//                          cudaDataType,               valueType);

SPARSE_API_DIRECT_NOHANDLE(DestroyDnVec, destroy_dnvec_descr,
    cusparseConstDnVecDescr_t, dnVecDescr);

// SPARSE_API_DIRECT_NOHANDLE(DnVecGet, please_Write_me,
//  cusparseDnVecDescr_t dnVecDescr,
//                  int64_t*,             size,
//                  void**,               values,
//                  cudaDataType*,        valueType);

// SPARSE_API_DIRECT_NOHANDLE(ConstDnVecGet, please_Write_me,
//  cusparseConstDnVecDescr_t dnVecDescr,
//                       int64_t*,                  size,
//                       const void**,              values,
//                       cudaDataType*,             valueType);

// SPARSE_API_DIRECT_NOHANDLE(DnVecGetValues, please_Write_me,
//  cusparseDnVecDescr_t dnVecDescr,
//                        void**,               values);

// SPARSE_API_DIRECT_NOHANDLE(ConstDnVecGetValues, please_Write_me,
//  cusparseConstDnVecDescr_t dnVecDescr,
//                             const void**,              values);

// SPARSE_API_DIRECT_NOHANDLE(DnVecSetValues, please_Write_me,
//  cusparseDnVecDescr_t dnVecDescr,
//                        void*,                values);
