cmake_minimum_required (VERSION 3.21)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

set(CMAKE_CXX_COMPILER /local/home/am273028/source/hipSYCL/build/install/bin/syclcc) 

project(SYCL-Advection LANGUAGES CXX)


if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()


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
target_link_libraries(main PUBLIC compiler_wrapper)

add_executable(reduction reduction.cpp)
target_link_libraries(reduction PUBLIC compiler_wrapper)