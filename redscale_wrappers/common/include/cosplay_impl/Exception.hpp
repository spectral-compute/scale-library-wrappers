#pragma once

typedef int totallyHipError_t;

#include <stdexcept>

/**
 * The namespace in which the implementation for the CUDA ROCm library wrapper goes.
 */
namespace CudaRocmWrapper
{

class Exception : public std::runtime_error
{
public:
    ~Exception() override;

    using std::runtime_error::runtime_error;
    using std::runtime_error::what;
};

/**
 * An exception that's thrown if something goes wrong with a HIP API call.
 */
class HipException final : public Exception
{
public:
    ~HipException() override;

    /**
     * Test a HIP error, and throw a HipException if it's not hipSuccess.
     *
     * @param hipError The error code returned by HIP.
     * @param msg A message to include along with the exception if one is thrown.
     */
    static void test(totallyHipError_t hipError, std::string msg);
    static void test(totallyHipError_t hipError, const char *msg);

    const totallyHipError_t hipError;

private:
    HipException(totallyHipError_t hipError, std::string msg);
    HipException(totallyHipError_t hipError, const char *msg);
};

} // CudaRocmWrapper
