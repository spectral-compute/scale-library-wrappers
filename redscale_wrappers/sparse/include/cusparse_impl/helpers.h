
#include "common.h"
#include "cusparse.h"

/// Helpers

SPARSE_API_DIRECT_NOHANDLE(CreateMatDescr, create_mat_descr,
    cusparseMatDescr_t*, descrA);

SPARSE_API_DIRECT_NOHANDLE(DestroyMatDescr, destroy_mat_descr,
    cusparseMatDescr_t, descrA);

//GPUSPARSE_EXPORT_C cusparseStatus_t
//cusparseCopyMatDescr(cusparseMatDescr_t       dest,
//                     const cusparseMatDescr_t src)

SPARSE_API_DIRECT_NOHANDLE(SetMatType, set_mat_type,
    cusparseMatDescr_t,   descrA,
    cusparseMatrixType_t, type)

GPUSPARSE_EXPORT_C cusparseMatrixType_t 
    cusparseGetMatType(const cusparseMatDescr_t descrA)
    INLINE_BODY(
    return from_rocsparse_matrix_type(rocsparse_get_mat_type(descrA));
)

SPARSE_API_DIRECT_NOHANDLE(SetMatFillMode, set_mat_fill_mode,
    cusparseMatDescr_t, descrA,
    cusparseFillMode_t, fillMode)

GPUSPARSE_EXPORT_C cusparseFillMode_t 
cusparseGetMatFillMode(const cusparseMatDescr_t descrA)
    INLINE_BODY(
        return from_rocsparse_fill_mode(rocsparse_get_mat_fill_mode(descrA));
)

// This one is fun. It no longer exists in CUDA oficially. 
// But that doesn't stop amgx (and maybe others) from linking against it, and using it.
// Let's just ignore it.
GPUSPARSE_EXPORT_C cusparseStatus_t cusparseSetMatFullPrecision(cusparseMatDescr_t descrA, bool fullprec)
    INLINE_BODY(
        (void)descrA;
        (void)fullprec;
        return CUSPARSE_STATUS_SUCCESS;
)

SPARSE_API_DIRECT_NOHANDLE(SetMatDiagType, set_mat_diag_type,
    cusparseMatDescr_t, descrA,
    cusparseDiagType_t, diagType)

GPUSPARSE_EXPORT_C cusparseDiagType_t 
cusparseGetMatDiagType(const cusparseMatDescr_t descrA)
INLINE_BODY(
    return from_rocsparse_diag_type(rocsparse_get_mat_diag_type(descrA));
)

SPARSE_API_DIRECT_NOHANDLE(SetMatIndexBase, set_mat_index_base,
    cusparseMatDescr_t,  descrA,
    cusparseIndexBase_t, base)

GPUSPARSE_EXPORT_C cusparseIndexBase_t 
cusparseGetMatIndexBase(const cusparseMatDescr_t descrA)
INLINE_BODY(
    return from_rocsparse_index_base(rocsparse_get_mat_index_base(descrA));
)

// TODO: maybe never
// SPARSE_API_DIRECT_NOHANDLE(CreateCsric02Info, please_Write_me,
//  csric02Info_t* info)

// SPARSE_API_DIRECT_NOHANDLE(DestroyCsric02Info, please_Write_me,
//  csric02Info_t info)

// SPARSE_API_DIRECT_NOHANDLE(CreateBsric02Info, please_Write_me,
//  bsric02Info_t* info)

// SPARSE_API_DIRECT_NOHANDLE(DestroyBsric02Info, please_Write_me,
//  bsric02Info_t info)

// SPARSE_API_DIRECT_NOHANDLE(CreateCsrilu02Info, please_Write_me,
//  csrilu02Info_t* info)

// SPARSE_API_DIRECT_NOHANDLE(DestroyCsrilu02Info, please_Write_me,
//  csrilu02Info_t info)

// SPARSE_API_DIRECT_NOHANDLE(CreateBsrilu02Info, please_Write_me,
//  bsrilu02Info_t* info)

// SPARSE_API_DIRECT_NOHANDLE(DestroyBsrilu02Info, please_Write_me,
//  bsrilu02Info_t info)

// SPARSE_API_DIRECT_NOHANDLE(CreateBsrsv2Info, please_Write_me,
//  bsrsv2Info_t* info)

// SPARSE_API_DIRECT_NOHANDLE(DestroyBsrsv2Info, please_Write_me,
//  bsrsv2Info_t info)

// SPARSE_API_DIRECT_NOHANDLE(CreateBsrsm2Info, please_Write_me,
//  bsrsm2Info_t* info)

// SPARSE_API_DIRECT_NOHANDLE(DestroyBsrsm2Info, please_Write_me,
//  bsrsm2Info_t info)

// SPARSE_API_DIRECT_NOHANDLE(CreateCsru2csrInfo, please_Write_me,
//  csru2csrInfo_t* info)

// SPARSE_API_DIRECT_NOHANDLE(DestroyCsru2csrInfo, please_Write_me,
//  csru2csrInfo_t info)

// SPARSE_API_DIRECT_NOHANDLE(CreateColorInfo, please_Write_me,
//  cusparseColorInfo_t* info)

// SPARSE_API_DIRECT_NOHANDLE(DestroyColorInfo, please_Write_me,
//  cusparseColorInfo_t info)

// SPARSE_API_DIRECT_NOHANDLE(CreatePruneInfo, please_Write_me,
//  pruneInfo_t* info)

// SPARSE_API_DIRECT_NOHANDLE(DestroyPruneInfo, please_Write_me,
//  pruneInfo_t info)
