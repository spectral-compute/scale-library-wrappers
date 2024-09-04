find_sources(COMMON_TEST_SOURCES "${CMAKE_CURRENT_LIST_DIR}")
list(APPEND COMMON_TEST_SOURCES "${CMAKE_CURRENT_LIST_DIR}/../main.cpp")
add_gtest_executable(cosplay_common_gtest "${COMMON_TEST_SOURCES}")
target_link_libraries(cosplay_common_gtest PRIVATE cosplaycommon)
