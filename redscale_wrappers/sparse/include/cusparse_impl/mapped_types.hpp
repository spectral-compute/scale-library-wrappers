#ifndef MATHLIBS_SPARSE_MAPPED_TYPES_H
#define MATHLIBS_SPARSE_MAPPED_TYPES_H

#include "common.h"

#include "types.h"

#include "blas_impl/shared.hpp"
#include "cosplay_impl/enum_mapper.hpp"

#include "rocsparse/rocsparse-types.h"

// If we were to use rocsparse_float_complex, rocsparse_double_complex, we would get build problems 
// with any library including both cusolver and cusparse, due to cuToRoc() being a direct conversion.
// We do a slightly evil thing here.
// #ifndef ROCBLAS_COMPLEX_TYPES_H
// Instead of:
// CU_TO_ROC_COMPLEX_FLOAT(rocsparse_float_complex)
// CU_TO_ROC_COMPLEX_DOUBLE(rocsparse_double_complex)
// We do
// #else
#include <rocblas/rocblas.h>
#define rocsparse_float_complex rocblas_float_complex
#define rocsparse_double_complex rocblas_double_complex
// #endif // ROCBLAS_COMPLEX_TYPES_H

struct __redscale_cusparseStatus_t {
    cusparseStatus_t value;

    __redscale_cusparseStatus_t(rocsparse_status rocStatus) {
        switch (rocStatus) {
            case rocsparse_status_success:  /**< success. */
                value = CUSPARSE_STATUS_SUCCESS;
                break;
            case rocsparse_status_invalid_handle:  /**< handle not initialized, invalid or null. */
                value = CUSPARSE_STATUS_INVALID_VALUE;
                break;
            case rocsparse_status_not_implemented:  /**< function is not implemented. */
                value = CUSPARSE_STATUS_NOT_SUPPORTED;
                break;
            case rocsparse_status_invalid_pointer:  /**< invalid pointer parameter. */
                value = CUSPARSE_STATUS_INVALID_VALUE;
                break;
            case rocsparse_status_invalid_size:  /**< invalid size parameter. */
                value = CUSPARSE_STATUS_INVALID_VALUE;
                break;
            case rocsparse_status_memory_error:  /**< failed memory allocation, copy, dealloc. */
                value = CUSPARSE_STATUS_ALLOC_FAILED;
                break;
            case rocsparse_status_invalid_value:  /**< invalid value parameter. */
                value = CUSPARSE_STATUS_INVALID_VALUE;
                break;
            case rocsparse_status_arch_mismatch:  /**< device arch is not supported. */
                value = CUSPARSE_STATUS_ARCH_MISMATCH;
                break;
            case rocsparse_status_zero_pivot:  /**< encountered zero pivot. */
                value = CUSPARSE_STATUS_ZERO_PIVOT;
                break;
            case rocsparse_status_not_initialized:  /**< descriptor has not been initialized. */
                value = CUSPARSE_STATUS_NOT_INITIALIZED;
                break;
            case rocsparse_status_type_mismatch:  /**< index types do not match. */
                value = CUSPARSE_STATUS_INVALID_VALUE;
                break;
            case rocsparse_status_requires_sorted_storage: /** storage required. */
                value = CUSPARSE_STATUS_NOT_SUPPORTED;
                break;
            default: /** The front fell off. */
                value = CUSPARSE_STATUS_INTERNAL_ERROR;
                break;
        }
    }

    operator cusparseStatus_t() {
        return this->value;
    }
};

#include "cosplay_impl/cu_to_roc.hpp"
#include "cosplay_impl/cu_to_roc_cmplx.hpp"

MAP_ENUM_PARTIAL(rocsparse_indextype, cusparseIndexType_t,
    (CUSPARSE_INDEX_32I, rocsparse_indextype_i32),
    (CUSPARSE_INDEX_64I, rocsparse_indextype_i64)
)


MAP_ENUM_EXHAUSTIVE(rocsparse_index_base, cusparseIndexBase_t,
    (CUSPARSE_INDEX_BASE_ZERO, rocsparse_index_base_zero),
    (CUSPARSE_INDEX_BASE_ONE, rocsparse_index_base_one)
)

ASSERT_EQUAL(rocsparse_datatype, rocblas_datatype,
    (rocblas_datatype_i8_r, rocsparse_datatype_i8_r),
    (rocblas_datatype_u8_r, rocsparse_datatype_u8_r),
    (rocblas_datatype_i32_r, rocsparse_datatype_i32_r),
    (rocblas_datatype_u32_r, rocsparse_datatype_u32_r),
    (rocblas_datatype_f32_r, rocsparse_datatype_f32_r),
    (rocblas_datatype_f64_r, rocsparse_datatype_f64_r),
    (rocblas_datatype_f32_c, rocsparse_datatype_f32_c),
    (rocblas_datatype_f64_c, rocsparse_datatype_f64_c)
)


MAP_ENUM_EXHAUSTIVE(rocsparse_direction, cusparseDirection_t,
    (CUSPARSE_DIRECTION_ROW, rocsparse_direction_row),
    (CUSPARSE_DIRECTION_COLUMN, rocsparse_direction_column)
)

MAP_ENUM_EXHAUSTIVE(rocsparse_operation, cusparseOperation_t,
    (CUSPARSE_OPERATION_NON_TRANSPOSE, rocsparse_operation_none),
    (CUSPARSE_OPERATION_TRANSPOSE, rocsparse_operation_transpose),
    (CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, rocsparse_operation_conjugate_transpose)
)

MAP_ENUM_EXHAUSTIVE(rocsparse_order, cusparseOrder_t,
    (CUSPARSE_ORDER_ROW, rocsparse_order_row),
    (CUSPARSE_ORDER_COL, rocsparse_order_column)
)

MAP_ENUM_PARTIAL(rocsparse_spmm_alg, cusparseSpMMAlg_t,
    (CUSPARSE_SPMM_ALG_DEFAULT, rocsparse_spmm_alg_default)
)

CU_TO_ROC_POINTER(rocsparse_pointer_mode, cusparsePointerMode_t)
MAP_ENUM_EXHAUSTIVE(rocsparse_pointer_mode, cusparsePointerMode_t, 
    (CUSPARSE_POINTER_MODE_DEVICE, rocsparse_pointer_mode_device),
    (CUSPARSE_POINTER_MODE_HOST, rocsparse_pointer_mode_host)
)


MAP_ENUM_EXHAUSTIVE(rocsparse_matrix_type, cusparseMatrixType_t, 
    (CUSPARSE_MATRIX_TYPE_GENERAL, rocsparse_matrix_type_general),
    (CUSPARSE_MATRIX_TYPE_SYMMETRIC, rocsparse_matrix_type_symmetric),
    (CUSPARSE_MATRIX_TYPE_HERMITIAN, rocsparse_matrix_type_hermitian),
    (CUSPARSE_MATRIX_TYPE_TRIANGULAR, rocsparse_matrix_type_triangular)
)

MAP_ENUM_EXHAUSTIVE(rocsparse_fill_mode, cusparseFillMode_t, 
    (CUSPARSE_FILL_MODE_LOWER, rocsparse_fill_mode_lower),
    (CUSPARSE_FILL_MODE_UPPER, rocsparse_fill_mode_upper)
)

MAP_ENUM_EXHAUSTIVE(rocsparse_diag_type, cusparseDiagType_t, 
    (CUSPARSE_DIAG_TYPE_UNIT, rocsparse_diag_type_unit),
    (CUSPARSE_DIAG_TYPE_NON_UNIT, rocsparse_diag_type_non_unit)
)

MAP_ENUM_PARTIAL(rocsparse_spmv_alg, cusparseSpMVAlg_t, 
    (CUSPARSE_SPMV_ALG_DEFAULT, rocsparse_spmv_alg_default),
    (CUSPARSE_SPMV_COO_ALG1, rocsparse_spmv_alg_coo),
    (CUSPARSE_SPMV_COO_ALG2, rocsparse_spmv_alg_coo),
    (CUSPARSE_SPMV_CSR_ALG1, rocsparse_spmv_alg_csr_adaptive),
    (CUSPARSE_SPMV_CSR_ALG2, rocsparse_spmv_alg_csr_adaptive)
)

MAP_ENUM_EXHAUSTIVE(rocsparse_action, cusparseAction_t,
    (CUSPARSE_ACTION_NUMERIC, rocsparse_action_numeric),
    (CUSPARSE_ACTION_SYMBOLIC, rocsparse_action_symbolic)
)

MAP_ENUM_PARTIAL(rocsparse_spgemm_alg, cusparseSpGEMMAlg_t,
    (CUSPARSE_SPGEMM_DEFAULT, rocsparse_spgemm_alg_default)
)
#endif // MATHLIBS_SPARSE_MAPPED_TYPES_H
