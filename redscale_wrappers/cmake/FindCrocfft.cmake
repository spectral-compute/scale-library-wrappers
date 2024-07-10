# This is a CMake finder script for rocFFT that only depends on C, and not HIP.

include(FindPackageHandleStandardArgs)

find_path(ROCFFT_INCLUDE_DIR rocfft/rocfft.h PATH_SUFFIXES include)
find_library(ROCFFT_LIBRARY rocfft PATH_SUFFIXES lib)
find_package_handle_standard_args(Crocfft DEFAULT_MSG ROCFFT_LIBRARY ROCFFT_INCLUDE_DIR)

add_library(roc::rocfft SHARED IMPORTED)
set_target_properties(roc::rocfft PROPERTIES
                      INTERFACE_INCLUDE_DIRECTORIES "${ROCFFT_INCLUDE_DIR}"
                      IMPORTED_LOCATION "${ROCFFT_LIBRARY}")
