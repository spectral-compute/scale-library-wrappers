# This is a CMake finder script for rocRAND that only depends on C, and not HIP.

include(FindPackageHandleStandardArgs)

find_path(ROCRAND_INCLUDE_DIR rocrand/rocrand.h PATH_SUFFIXES include)
find_library(ROCRAND_LIBRARY rocrand PATH_SUFFIXES lib)
find_package_handle_standard_args(Crocrand DEFAULT_MSG ROCRAND_LIBRARY ROCRAND_INCLUDE_DIR)

add_library(roc::rocrand SHARED IMPORTED)
set_target_properties(roc::rocrand PROPERTIES
                      INTERFACE_INCLUDE_DIRECTORIES "${ROCRAND_INCLUDE_DIR}"
                      IMPORTED_LOCATION "${ROCRAND_LIBRARY}")
