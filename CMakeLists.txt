cmake_minimum_required (VERSION 3.21)

project(SYCL-Advection LANGUAGES CXX)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Prevent from build-in source tree
include(preventBuildInSource)
# Init build type: Release, Debug, ...
include(initBuildType)
# useful for debugging cmake
include(print_target_properties)


set(AVAILABLE_BACKENDS "sycl;kokkos")
set(BACKEND "" CACHE STRING "Choose the dimension, options are ${AVAILABLE_BACKENDS}")
set_property(CACHE BACKEND PROPERTY STRINGS "${AVAILABLE_BACKENDS}")
if(NOT ("${BACKEND}"  IN_LIST AVAILABLE_BACKENDS))
message(FATAL_ERROR "BACKEND is not defined, available backends are ${AVAILABLE_BACKENDS}")
endif()

# we will copy the .ini file and binaries inside BIN_DIR

if("${BACKEND}" STREQUAL "sycl")
    set(INIFILE_LOCATION "${CMAKE_BINARY_DIR}/src/sycl")
    endif()
if("${BACKEND}" STREQUAL "kokkos")
    set(INIFILE_LOCATION "${CMAKE_BINARY_DIR}/src/kokkos")
endif()

add_subdirectory(src)

if("${BACKEND}" STREQUAL "sycl")
    add_subdirectory(src/sycl)
endif()
if("${BACKEND}" STREQUAL "kokkos")
    add_subdirectory(vendor/kokkos)
    add_subdirectory(src/kokkos)
endif()