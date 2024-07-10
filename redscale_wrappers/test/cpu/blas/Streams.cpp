#include "gtest/gtest.h"
#include "cublas_v2.h"

// This test is specifically designed to have race conditions, so when it fails it is likely to
// fail nondeterministically. Enjoy.

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
    EXPECT_EQ(version, 42);
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

    EXPECT_EQ(major, 42);
    EXPECT_EQ(minor, 0);
    EXPECT_EQ(patch, 0);
}
