include(../common.cmake)

add_cosplay_lib(
    NAME cusolver
    AMD_NAME rocsolver
    MACRO_NAME "GPUSOLVER"
    DEPSCAN_HEADERS
        ${CMAKE_CURRENT_LIST_DIR}/include/cusolverDn.h
)

target_link_libraries(cusolver PRIVATE cusparse cublas)
