# This is a CMake finder script for rocsparse that only depends on C, and not HIP.

include(FindPackageHandleStandardArgs)

if (NOT ROCM_PATH)
    set(ROCM_PATH "/opt/rocm")
endif()

find_path(ROCSPARSE_INCLUDE_DIR rocsparse/rocsparse.h PATH_SUFFIXES include)
find_library(ROCSPARSE_LIBRARY rocsparse PATH_SUFFIXES lib)
find_package_handle_standard_args(Crocsparse DEFAULT_MSG ROCSPARSE_LIBRARY ROCSPARSE_INCLUDE_DIR)

add_library(roc::rocsparse SHARED IMPORTED)
set_target_properties(roc::rocsparse PROPERTIES
                      INTERFACE_INCLUDE_DIRECTORIES "${ROCSPARSE_INCLUDE_DIR}"
                      IMPORTED_LOCATION "${ROCSPARSE_LIBRARY}")
