find_sources(RAND_TEST_SOURCES "${CMAKE_CURRENT_LIST_DIR}")
list(APPEND RAND_TEST_SOURCES "${CMAKE_CURRENT_LIST_DIR}/../main.cpp")
set_source_files_properties(${RAND_TEST_SOURCES} PROPERTIES LANGUAGE CXX)

add_wrapper_gtest(cosplay_rand_gtest "${RAND_TEST_SOURCES}")
target_link_libraries(cosplay_rand_gtest PRIVATE curand)

# Build the nvidia version.
if (WithNvidia)
    add_nvidia_gtest_executable(nvidia_rand_gtest ${RAND_TEST_SOURCES})
    target_link_libraries(nvidia_rand_gtest PRIVATE CUDA::curand)
endif()
