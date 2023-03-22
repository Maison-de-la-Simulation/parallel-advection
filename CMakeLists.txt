cmake_minimum_required (VERSION 3.21)

project(SYCL-Advection LANGUAGES CXX)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Prevent from build-in source tree
include(preventBuildInSource)
# Init build type: Release, Debug, ...
include(initBuildType)
# useful for debugging cmake
include(print_target_properties)


add_subdirectory(config)

################################################
#            SYCL Compiler options             #
################################################
get_filename_component(NAME "${CMAKE_CXX_COMPILER}"
                       PROGRAM [PROGRAM_ARGS <ARG_VAR>]
                       [CACHE])

add_library(compiler_wrapper INTERFACE)
message(STATUS "My compiler is : ${CMAKE_CXX_COMPILER}")
if("${CMAKE_CXX_COMPILER}" STREQUAL "icpx")
    message(STATUS "Setting flags for icpx compiler")
    target_compile_options(compiler_wrapper INTERFACE -fsycl)
    target_link_options(compiler_wrapper INTERFACE -fsycl)
endif()

################################################
#                  Executables                 #
################################################
add_executable(main main.cpp)
target_link_libraries(main PUBLIC compiler_wrapper adv1d::config)

configure_file(advection.ini advection.ini COPYONLY)

