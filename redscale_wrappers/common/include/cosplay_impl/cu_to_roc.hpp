#pragma once

#include "enum_mapper.hpp"
#include <cuComplex.h>


#define CU_TO_ROC_POINTER(ROCTYPE, CUTYPE) \
    __attribute__((always_inline))                  \
    inline ROCTYPE* to_ ## ROCTYPE(CUTYPE* arg) {   \
        return reinterpret_cast<ROCTYPE*>(arg);                      \
    }                                               \
    SPECIALISE_CU_TO_ROC_POINTER(CUTYPE, ROCTYPE)

namespace
{

/**
 * Cast an argument to the corresponding rocBLAS type.
 */
template <typename T>
struct CuToRoc final
{
    __attribute__((always_inline))
    static T operator()(T value)
    {
        return value;
    }
};
template <typename T>
struct CuToRoc<const T *> final
{
    __attribute__((always_inline))
    static auto operator()(const T *value) -> decltype(CuToRoc<T *>{}(const_cast<T *>(value)))
    {
        return CuToRoc<T *>{}(const_cast<T *>(value));
    }
};

template <typename T> __attribute__((always_inline))
auto cuToRoc(T value)
{
    return CuToRoc<T>{}(value);
}


} // namespace
