# This is a CMake finder script for rocBLAS that only depends on C, and not HIP.

include(FindPackageHandleStandardArgs)

find_path(ROCBLAS_INCLUDE_DIR rocblas/rocblas.h PATH_SUFFIXES include)
find_library(ROCBLAS_LIBRARY rocblas PATH_SUFFIXES lib)
find_package_handle_standard_args(Crocblas DEFAULT_MSG ROCBLAS_LIBRARY ROCBLAS_INCLUDE_DIR)

add_library(roc::rocblas SHARED IMPORTED)
set_target_properties(roc::rocblas PROPERTIES
                      INTERFACE_INCLUDE_DIRECTORIES "${ROCBLAS_INCLUDE_DIR}"
                      IMPORTED_LOCATION "${ROCBLAS_LIBRARY}")
