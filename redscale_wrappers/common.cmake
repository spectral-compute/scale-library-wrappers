function(_record_lib_deps TARGET)
    # Get the imported library.
    get_target_property(IMPORTED_LOCATION ${TARGET} IMPORTED_LOCATION)

    # Add the imported library to the list of dependencies.
    list(APPEND ROCM_LIB_DEPS "${IMPORTED_LOCATION}")
    list(REMOVE_DUPLICATES ROCM_LIB_DEPS)
    set(ROCM_LIB_DEPS "${ROCM_LIB_DEPS}" CACHE INTERNAL "")
endfunction()

function(add_cosplay_lib)
    set(flags)
    set(oneValueArgs NAME AMD_NAME MACRO_NAME SRC_DIR)
    set(multiValueArgs DEPSCAN_HEADERS)
    cmake_parse_arguments("d" "${flags}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if ("${d_SRC_DIR}" STREQUAL "")
        set(d_SRC_DIR src)
    endif()

    find_sources(SRC_FILES "${CMAKE_CURRENT_LIST_DIR}/${d_SRC_DIR}" "${CMAKE_CURRENT_LIST_DIR}/include")

    add_library(${d_NAME} ${SRC_FILES})

    if (NOT "${d_AMD_NAME}" STREQUAL "")
        find_package("C${d_AMD_NAME}" REQUIRED)
        target_link_libraries(${d_NAME} PUBLIC roc::${d_AMD_NAME})
        _record_lib_deps(roc::${d_AMD_NAME})
    endif ()

    add_export_header(${d_NAME} BASE_NAME "${d_MACRO_NAME}" EXPORT_FILE_NAME "redscale/${d_NAME}/export.h" INCLUDE_PATH_SUFFIX "redscale")
    target_include_directories(${d_NAME}
        PRIVATE
            "${PROJECT_SOURCE_DIR}/BLAS/include"

        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
            "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>"
    )


    target_link_libraries(${d_NAME} PRIVATE cosplaycommon redscale)
    install(DIRECTORY include/ DESTINATION ./include/redscale)

    # Install dependency headers. We need to do this even if the shared libraries aren't installed because there's no
    # rpath equivalent for headers unless we rewrite our headers to include by absolute path.
    if (NOT "${d_DEPSCAN_HEADERS}" STREQUAL "")
        get_target_property(IINC roc::${d_AMD_NAME} INTERFACE_INCLUDE_DIRECTORIES)
        get_cpp_dependencies(RESULT SOLVER_DEPS
            TARGETS ${d_NAME} roc::${d_AMD_NAME} redscale cosplaycommon
            INCLUDE_PATHS ${PROJECT_SOURCE_DIR}/build_tools/depinc
            SOURCES ${d_DEPSCAN_HEADERS}
            RELATIVE "${IINC}"
        )

        foreach (_H IN LISTS SOLVER_DEPS)
            get_filename_component(DIR "${_H}" DIRECTORY)
            install(FILES "${IINC}/${_H}" DESTINATION "./include/redscale/${DIR}")
        endforeach ()
    endif()
endfunction()
