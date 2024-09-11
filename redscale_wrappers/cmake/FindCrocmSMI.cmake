# This is a CMake finder script for rocmSMI that only depends on C, and not HIP.

include(FindPackageHandleStandardArgs)

find_path(ROCMSMI_INCLUDE_DIR rocm_smi/rocm_smi.h PATH_SUFFIXES include)
find_library(ROCMSMI_LIBRARY rocm_smi64 PATH_SUFFIXES lib)
find_package_handle_standard_args(CrocmSMI DEFAULT_MSG ROCMSMI_LIBRARY ROCMSMI_INCLUDE_DIR)

add_library(roc::rocmSMI SHARED IMPORTED)
set_target_properties(roc::rocmSMI PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ROCMSMI_INCLUDE_DIR}"
    IMPORTED_LOCATION "${ROCMSMI_LIBRARY}"
)
