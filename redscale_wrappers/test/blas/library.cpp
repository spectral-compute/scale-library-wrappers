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

    EXPECT_EQ(version, CUBLAS_VERSION);
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

    EXPECT_EQ(major, CUBLAS_VER_MAJOR);
    EXPECT_EQ(minor, CUBLAS_VER_MINOR);
    EXPECT_EQ(patch, CUBLAS_VER_PATCH);
}
