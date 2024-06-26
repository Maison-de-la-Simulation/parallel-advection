cmake_minimum_required (VERSION 3.20)

project(SYCL-Advection LANGUAGES CXX)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(preventBuildInSource) # Prevent from build-in source tree
include(initBuildType) # Init build type: Release, Debug, ...
include(print_target_properties) # useful for debugging cmake

if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM") OR (NOT ${DPCPP_FSYCL_TARGETS} STREQUAL ""))
    set(SYCL_IS_ACPP OFF)
else()
    set(SYCL_IS_ACPP ON)
    find_package(AdaptiveCpp REQUIRED)
endif()

add_library(dpcpp_opt INTERFACE)
if(NOT SYCL_IS_ACPP)
    message(STATUS "Setting flags for IntelLLVM compiler")
    # find_package(IntelDPCPP REQUIRED)
    target_link_options(dpcpp_opt INTERFACE -fno-sycl-id-queries-fit-in-int -fsycl ${DPCPP_FSYCL_TARGETS})
    target_compile_options(dpcpp_opt INTERFACE -fno-sycl-id-queries-fit-in-int -fsycl ${DPCPP_FSYCL_TARGETS})
else()
    message(STATUS "To add correct flags for DPCPP compiler, use the "
    "DPCPP_FSYCL_TARGETS' variable. Example: "
    "-DDPCPP_FSYCL_TARGETS='-fsycl-targets=nvptx64-nvidia-cuda'")
endif()

add_subdirectory(src)

################################################################################
if("${ADVECTION_BUILD_BENCHMARKS}")
    find_package(benchmark REQUIRED)
    add_subdirectory(benchmark)
else()
    message(STATUS "Skipping benchmark build, use ADVECTION_BUILD_BENCHMARKS=ON to build benchmarks.")
endif()

################################################################################
if("${ADVECTION_BUILD_TESTS}")
    message(STATUS "Building tests")
    enable_testing()
    add_subdirectory(thirdparty/googletest)
    add_subdirectory(tests)
else()
    message(STATUS "Skipping tests, use ADVECTION_BUILD_TESTS=ON to build tests.")
endif()
