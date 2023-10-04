cmake_minimum_required (VERSION 3.21)

project(SYCL-Advection LANGUAGES CXX)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Prevent from build-in source tree
include(preventBuildInSource)
# Init build type: Release, Debug, ...
include(initBuildType)
# useful for debugging cmake
include(print_target_properties)

add_library(compiler_wrapper INTERFACE)
message(STATUS "My compiler is    : ${CMAKE_CXX_COMPILER}")
message(STATUS "My compiler ID is : ${CMAKE_CXX_COMPILER_ID}")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
    message(STATUS "Setting flags for icpx compiler")
    find_package(IntelDPCPP REQUIRED)
    target_link_options(compiler_wrapper INTERFACE -v)
    target_compile_options(compiler_wrapper INTERFACE -v)
endif()

add_subdirectory(src)
