#ifndef MATHLIBS_CU_TO_ROC_CMPLX_H
#define MATHLIBS_CU_TO_ROC_CMPLX_H

#include "cu_to_roc.hpp"
#include "rocblas/internal/rocblas-complex-types.h"


#define CU_TO_ROC_COMPLEX_FLOAT(ROC_TYPE) \
template <> \
struct CuToRoc<cuComplex *> final \
{ \
    [[maybe_unused]] __attribute__((always_inline)) \
    static ROC_TYPE *operator()(cuComplex *value) \
    { \
        return reinterpret_cast<ROC_TYPE *>(value); /*-fno-strict-aliasing is enabled*/ \
    } \
};

#define CU_TO_ROC_COMPLEX_DOUBLE(ROC_TYPE) \
template <>  \
struct CuToRoc<cuDoubleComplex *> final \
{ \
    [[maybe_unused]] __attribute__((always_inline)) \
    static ROC_TYPE *operator()(cuDoubleComplex *value) \
    { \
        return reinterpret_cast<ROC_TYPE *>(value); /*-fno-strict-aliasing is enabled*/ \
    } \
};

// Actually, since CuToRoc() cannot have multiple return types,
// this is the only conversion we can have with cuToRoc().
// As long as all mathlib use this, it should be fine...
CU_TO_ROC_COMPLEX_FLOAT(rocblas_float_complex)
CU_TO_ROC_COMPLEX_DOUBLE(rocblas_double_complex)

#endif //MATHLIBS_CU_TO_ROC_CMPLX_H
