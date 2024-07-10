include(GTest)

find_sources(BLAS_TEST_SOURCES "${CMAKE_CURRENT_LIST_DIR}")
set_source_files_properties(${BLAS_TEST_SOURCES} PROPERTIES LANGUAGE CXX)

add_gtest_executable(cosplay_blas_gtest "${BLAS_TEST_SOURCES}")
target_link_libraries(cosplay_blas_gtest PRIVATE cublas redscale)

# Build the nvidia version.
if (WithNvidia)
    add_gtest_executable(nvidia_blas_gtest ${BLAS_TEST_SOURCES})

    find_package(CUDAToolkit 8.0 REQUIRED)
    target_link_libraries(nvidia_blas_gtest PRIVATE CUDA::cudart CUDA::cublas)
endif()
