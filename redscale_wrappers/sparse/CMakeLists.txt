include(../common.cmake)

add_cosplay_lib(
    NAME cusparse
    AMD_NAME rocsparse
    MACRO_NAME "GPUSPARSE"
    DEPSCAN_HEADERS
        ${CMAKE_CURRENT_LIST_DIR}/include/cusparse_v2.h
)

add_cosplay_lib(
    NAME cusparseLt
    SRC_DIR lt
    MACRO_NAME "GPUSPARSE_LT"
)
