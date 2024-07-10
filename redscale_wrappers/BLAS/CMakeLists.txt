include(../common.cmake)

add_cosplay_lib(
    NAME cublas
    AMD_NAME rocblas
    MACRO_NAME "GPUBLAS"
    DEPSCAN_HEADERS
        ${CMAKE_CURRENT_LIST_DIR}/include/cublas_v2.h
)

add_cosplay_lib(
    NAME cublasLt
    SRC_DIR lt
    MACRO_NAME "GPUBLAS_LT"
)

# Copy over random data-files rocblas needs if we're copying the ROCm libraries.
get_filename_component(ROC_LIBDIR ${ROCBLAS_LIBRARY} DIRECTORY)
list(APPEND ROCM_LIB_DEPS "${ROC_LIBDIR}/rocblas")
list(REMOVE_DUPLICATES ROCM_LIB_DEPS)
set(ROCM_LIB_DEPS "${ROCM_LIB_DEPS}" CACHE INTERNAL "")
