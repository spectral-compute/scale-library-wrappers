

#ifndef ENUM_MAPPER_H
#define ENUM_MAPPER_H

#include <sstream>
#include <stdexcept>

#include "Exception.hpp"

namespace CudaRocmWrapper {

class CosplayEnumError : public CudaRocmWrapper::Exception {
public:
    CosplayEnumError(const std::string& msg)
        : CudaRocmWrapper::Exception(msg) {}
};

inline CosplayEnumError __cosplay_build_enum_error(const char* enum_name, int value) {
    std::stringstream ss;

    ss << "Unexpected value for enum " << enum_name << ": " << value;

    return CosplayEnumError(ss.str());
}

} // namespace CudaRocmWrapper

#endif // ENUM_MAPPER_H


#include "redscale_impl/cppmagic/PushMagic.hpp"


namespace
{


#define ENUM_BLOCK(P) if (x == FIRST P) {return SECOND P;}
#define REV_ENUM_BLOCK(P) if (x == SECOND P) {return FIRST P;}

#define SPECIALISE_CU_TO_ROC(CUTYPE, ROCTYPE) \
    template <> \
    struct CuToRoc<CUTYPE> final \
    { \
        [[maybe_unused]] __attribute__((always_inline)) \
        static ROCTYPE operator()(CUTYPE value) \
        { \
            return to_ ## ROCTYPE(value); \
        } \
    };

#define SPECIALISE_CU_TO_ROC_POINTER(CUTYPE, ROCTYPE) \
    template <> \
    struct CuToRoc<CUTYPE*> final \
    { \
        [[maybe_unused]] __attribute__((always_inline)) \
        static ROCTYPE* operator()(CUTYPE* value) \
        { \
            return to_ ## ROCTYPE(value); \
        } \
    };


#define ENUM_CU_TO_ROC_EXHAUSTIVE(ROCTYPE, CUTYPE, ...) \
    __attribute__((always_inline))                 \
	inline ROCTYPE to_ ## ROCTYPE(CUTYPE x) {      \
        MAP_DENSE(ENUM_BLOCK, __VA_ARGS__)         \
        throw CudaRocmWrapper::__cosplay_build_enum_error(#CUTYPE, x); \
    }

#define ENUM_ROC_TO_CU_EXHAUSTIVE(ROCTYPE, CUTYPE, ...) \
    __attribute__((always_inline))                 \
	inline CUTYPE from_ ## ROCTYPE(ROCTYPE x) {    \
        MAP_DENSE(REV_ENUM_BLOCK, __VA_ARGS__)     \
        throw CudaRocmWrapper::__cosplay_build_enum_error(#CUTYPE, x); \
	}                                              \
    SPECIALISE_CU_TO_ROC(CUTYPE, ROCTYPE)

#define MAP_ENUM_EXHAUSTIVE(ROCTYPE, CUTYPE, ...) \
    ENUM_CU_TO_ROC_EXHAUSTIVE(ROCTYPE, CUTYPE, __VA_ARGS__) \
    ENUM_ROC_TO_CU_EXHAUSTIVE(ROCTYPE, CUTYPE, __VA_ARGS__)

#define ENUM_CU_TO_ROC_PARTIAL(ROCTYPE, CUTYPE, ...) \
    __attribute__((always_inline))                 \
	inline ROCTYPE to_ ## ROCTYPE(CUTYPE x) {      \
        MAP_DENSE(ENUM_BLOCK, __VA_ARGS__)         \
        throw CudaRocmWrapper::__cosplay_build_enum_error(#CUTYPE, x); \
    }                                              \
    SPECIALISE_CU_TO_ROC(CUTYPE, ROCTYPE)

#define ENUM_ROC_TO_CU_PARTIAL(ROCTYPE, CUTYPE, ...) \
    __attribute__((always_inline))                 \
	inline CUTYPE from_ ## ROCTYPE(ROCTYPE x) {    \
        MAP_DENSE(REV_ENUM_BLOCK, __VA_ARGS__)     \
        throw CudaRocmWrapper::__cosplay_build_enum_error(#CUTYPE, x); \
	}

#define MAP_ENUM_PARTIAL(ROCTYPE, CUTYPE, ...) \
    ENUM_CU_TO_ROC_PARTIAL(ROCTYPE, CUTYPE, __VA_ARGS__) \
    ENUM_ROC_TO_CU_PARTIAL(ROCTYPE, CUTYPE, __VA_ARGS__)


#define SEMICOLON() ;
#define __ASSERT(P) static_assert(int(FIRST P) == int(SECOND P))

#define ASSERT_EQUAL(ROCTYPE, CUTYPE, ...) MAP(__ASSERT, SEMICOLON, __VA_ARGS__);

} // namespace
