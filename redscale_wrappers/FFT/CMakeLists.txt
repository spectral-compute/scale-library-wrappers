include(../common.cmake)

add_cosplay_lib(
    NAME cufft
    AMD_NAME rocfft
    MACRO_NAME "GPUFFT"
    DEPSCAN_HEADERS
        ${CMAKE_CURRENT_LIST_DIR}/include/cufft.h
)

# Copy over random data-files rocfft needs if we're copying the ROCm libraries.
get_filename_component(ROC_LIBDIR ${ROCBLAS_LIBRARY} DIRECTORY)
list(APPEND ROCM_LIB_DEPS "${ROC_LIBDIR}/rocfft")
list(REMOVE_DUPLICATES ROCM_LIB_DEPS)
set(ROCM_LIB_DEPS "${ROCM_LIB_DEPS}" CACHE INTERNAL "")
