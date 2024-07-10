# This is a CMake finder script for the HIP host library that only depends on C, and not the HIP language.

include(FindPackageHandleStandardArgs)

find_path(HIP_INCLUDE_DIR hip/hip_runtime_api.h PATH_SUFFIXES include)
find_library(HIP_LIBRARY amdhip64 PATH_SUFFIXES lib)
find_package_handle_standard_args(Chip DEFAULT_MSG HIP_LIBRARY HIP_INCLUDE_DIR)

add_library(hip::host SHARED IMPORTED)
set_target_properties(hip::host PROPERTIES
                      INTERFACE_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIR}"
                      IMPORTED_LOCATION "${HIP_LIBRARY}")
target_compile_definitions(hip::host INTERFACE __HIP_PLATFORM_AMD__=1)
