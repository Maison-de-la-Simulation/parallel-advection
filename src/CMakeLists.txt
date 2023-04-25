add_subdirectory(config)
add_subdirectory(core)
add_subdirectory(tools)

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
add_executable(advection main.cpp)
target_link_libraries(advection
    PUBLIC
    compiler_wrapper
    adv1d::config
    adv1d::core
    adv1d::tools)

configure_file(advection.ini advection.ini COPYONLY)

