find_sources(COMMON_SOURCES "src" "include")
add_library(cosplaycommon STATIC NOINSTALL "${COMMON_SOURCES}")
target_include_directories(cosplaycommon PUBLIC "include")
target_link_libraries(cosplaycommon PUBLIC redscale)

# Add AMD dependencies.
find_package(hsa-runtime64 REQUIRED)
find_package(Chip)
target_link_libraries(cosplaycommon PUBLIC hip::host hsa-runtime64::hsa-runtime64)

# We need to be able to cast from cuComplex * to rocblas_float_complex *.
target_compile_options(cosplaycommon PUBLIC -fno-strict-aliasing)

install(DIRECTORY include/ DESTINATION include/redscale/)
