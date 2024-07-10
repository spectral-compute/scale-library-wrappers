# This is a CMake finder script for rocsolver that only depends on C, and not HIP.

include(FindPackageHandleStandardArgs)

find_path(ROCSOLVER_INCLUDE_DIR rocsolver/rocsolver.h PATH_SUFFIXES include)
find_library(ROCSOLVER_LIBRARY rocsolver PATH_SUFFIXES lib)
find_package_handle_standard_args(Crocsolver DEFAULT_MSG ROCSOLVER_LIBRARY ROCSOLVER_INCLUDE_DIR)

add_library(roc::rocsolver SHARED IMPORTED)
set_target_properties(roc::rocsolver PROPERTIES
                      INTERFACE_INCLUDE_DIRECTORIES "${ROCSOLVER_INCLUDE_DIR}"
                      IMPORTED_LOCATION "${ROCSOLVER_LIBRARY}")
