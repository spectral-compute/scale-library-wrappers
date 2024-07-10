include(../common.cmake)

add_cosplay_lib(
    NAME curand
    AMD_NAME rocrand
    MACRO_NAME "GPURAND"
    DEPSCAN_HEADERS
        ${CMAKE_CURRENT_LIST_DIR}/include/curand_kernel.h
        ${CMAKE_CURRENT_LIST_DIR}/include/curand.h
)
