#include "gtest/gtest.h"
#include "cublas_v2.h"

TEST(BLAS, CreateDestroy) {
    cublasHandle_t handle;
    EXPECT_EQ(cublasCreate(&handle), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(cublasDestroy(handle), CUBLAS_STATUS_SUCCESS);
}

TEST(BLAS, Version) {
    cublasHandle_t handle;
    int version;
    EXPECT_EQ(cublasCreate(&handle), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(cublasGetVersion(handle, &version), CUBLAS_STATUS_SUCCESS);

#ifdef __REDSCALE__
    EXPECT_EQ(version, 42);
#endif // __REDSCALE__
}

TEST(BLAS, LibraryPropertyNonsense) {
    cublasHandle_t handle;
    int major;
    int minor;
    int patch;
    EXPECT_EQ(cublasCreate(&handle), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(cublasGetProperty(MAJOR_VERSION, &major), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(cublasGetProperty(MINOR_VERSION, &minor), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(cublasGetProperty(PATCH_LEVEL, &patch), CUBLAS_STATUS_SUCCESS);

#ifdef __REDSCALE__
    EXPECT_EQ(major, 42);
    EXPECT_EQ(minor, 0);
    EXPECT_EQ(patch, 0);
#endif // __REDSCALE__
}
