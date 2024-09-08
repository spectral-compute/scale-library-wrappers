cmake_minimum_required(VERSION 3.19)
include(../xcmake/scripts/Init.cmake)
project(redscale_mathswrappers LANGUAGES C CXX
        VERSION 1.0.2.0)
include(XCMake)
target_compile_options(ASSERTIONS_EFFECTS INTERFACE -UNDEBUG)

# Validate generator. We need a cmake generator that understands C++ include dependency tracking to properly
# excise the bits of rocm we want. To build on Windows, use the Visual Studio Toolchain, but the "Makefiles JOM"
# or Ninja cmake generator during compilation. This does not prevent use of the Visual Studio IDE.
string(FIND "${CMAKE_GENERATOR}" "Makefile" MF)
string(FIND "${CMAKE_GENERATOR}" "Ninja" NJ)
if (${NJ} STREQUAL "-1" AND "${MF}" STREQUAL "-1")
    fatal_error("Only Makefile-based or Ninja-based cmake generators are supported")
endif()

# Search in /opt/rocm (last), as this is where the ROCm libraries are located by default.
list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")

# The ROCm library finder scripts depend on HIP, which has trouble being found properly. But we're only interested in
# the C interface. So use finder scripts that only depend on that.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Find the RedSCALE library when building the wrappers on their own. Also check that we got the expected compiler.
if (NOT TARGET redscale)
    # Check that we got the RedSCALE compiler.
    if (NOT "${CMAKE_CXX_COMPILER}" MATCHES "/targets/gfx[^/]+/bin/clang\\+\\+")
		fatal_error("CMAKE_CXX_COMPILER does not point to RedSCALE's Clang. Did you forget to set CMAKE_CXX_COMPILER? Ensure it points to the clang++ binary.")
    endif()

    # Add the corresponding RedSCALE directory to the CMake prefix path.
    get_filename_component(REDSCALE_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(REDSCALE_DIR "${REDSCALE_DIR}" DIRECTORY)
    get_filename_component(REDSCALE_DIR "${REDSCALE_DIR}" DIRECTORY)
    get_filename_component(REDSCALE_DIR "${REDSCALE_DIR}" DIRECTORY)
    list(APPEND CMAKE_PREFIX_PATH "${REDSCALE_DIR}")

    # Find the RedSCALE CMake package.
    find_package(RedSCALE)

    # Set up to build the tests.
    if (XCMAKE_ENABLE_TESTS)
        # Make sure a target is set.
        if (NOT DEFINED DeviceTargets)
            fatal_error("Cannot build tests without targets")
        endif()

        list(LENGTH DeviceTargets DeviceTargetsLength)
        if (DeviceTargetsLength GREATER 1)
            fatal_error(FATAL_ERROR "Currently, only one GPU device is supported at once when building tests. Please "
                                    "set DeviceTargets accordingly.")
        endif()
    endif()
endif()

function (add_wrapper_gtest TARGET)
    add_gtest_executable(${TARGET} ${ARGN})
    if (TARGET redscale)
        apply_scale_amd_options(${TARGET})
    else()
        target_compile_options(${TARGET} PRIVATE -x cuda "--cuda-path=${PSEUDO_INSTALL}")
    endif()

    foreach(_target IN LISTS DeviceTargets)
        target_compile_options(${TARGET} PRIVATE --cuda-gpu-arch=${_target})
    endforeach()
    target_link_libraries(${TARGET} PRIVATE redscale)
endfunction()


# GCC libs explode in the presence of the scale-addrspace parser. PRobably need to fix that...
set(XCMAKE_SCALE OFF)

add_subdirectory(common)

add_subdirectory(BLAS)
#add_subdirectory(DNN) Currently mostly empty.
#add_subdirectory(FFT) Currently mostly empty.
add_subdirectory(RAND)
add_subdirectory(sparse)
add_subdirectory(solver)

if (XCMAKE_ENABLE_TESTS)
    add_subdirectory(test)
endif()
