find_sources(PP_SOURCES "${CMAKE_CURRENT_LIST_DIR}/src" "${CMAKE_CURRENT_LIST_DIR}/include")

add_library(gpupp ${PP_SOURCES})
target_link_libraries(gpupp PUBLIC roc::rocpp)
add_export_header(gpupp BASE_NAME "gpupp" EXPORT_FILE_NAME "gpupp/export.h")
target_link_libraries(gpupp PRIVATE redscale)

# We need to be able to cast from cuComplex * to rocblas_float_complex *.
target_compile_options(gpublas PRIVATE -fno-strict-aliasing)
