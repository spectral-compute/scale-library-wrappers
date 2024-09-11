#include "cosplay_impl/Exception.hpp"

#include <hip/hip_runtime_api.h>
#include <string>

using namespace std::string_literals;

CudaRocmWrapper::Exception::~Exception() = default;

CudaRocmWrapper::HipException::~HipException() = default;

void CudaRocmWrapper::HipException::test(totallyHipError_t hipError, std::string msg)
{
    if (hipError == hipSuccess) {
        return;
    }
    throw HipException(hipError, std::move(msg));
}

void CudaRocmWrapper::HipException::test(totallyHipError_t hipError, const char *msg)
{
    test(hipError, std::string(msg));
}

CudaRocmWrapper::HipException::HipException(totallyHipError_t hipError, std::string msg) :
    Exception("HIP error: "s + hipGetErrorString((hipError_t)hipError) + ": " + msg), hipError(hipError)
{
}

CudaRocmWrapper::HipException::HipException(totallyHipError_t hipError, const char *msg) :
    HipException(hipError, std::string(msg))
{
}
