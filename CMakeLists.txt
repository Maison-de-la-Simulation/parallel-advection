cmake_minimum_required (VERSION 3.20)

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
if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM") OR (NOT ${DPCPP_FSYCL_TARGETS} STREQUAL ""))
    message(STATUS "Setting flags for IntelLLVM compiler")
    # find_package(IntelDPCPP REQUIRED)
    # target_link_options(compiler_wrapper INTERFACE -fsycl)
    # target_compile_options(compiler_wrapper INTERFACE -fsycl)
    target_link_options(compiler_wrapper INTERFACE -fsycl ${DPCPP_FSYCL_TARGETS})
    target_compile_options(compiler_wrapper INTERFACE -fsycl ${DPCPP_FSYCL_TARGETS})
else()
    message(STATUS "To add correct flags for DPCPP compiler, use the DPCPP_FSYCL_TARGETS' variable. Example: -DDPCPP_FSYCL_TARGETS='-fsycl-targets=nvptx64-nvidia-cuda'")
endif()

add_subdirectory(src)
