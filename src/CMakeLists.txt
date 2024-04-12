add_subdirectory(config)
add_subdirectory(core)
add_subdirectory(tools)

################################################
#            SYCL Compiler options             #
################################################
get_filename_component(NAME "${CMAKE_CXX_COMPILER}"
                       PROGRAM [PROGRAM_ARGS <ARG_VAR>]
                       [CACHE])

################################################
#                  Executables                 #
################################################
add_executable(advection main.cpp)

if(SYCL_IS_ACPP)
    add_sycl_to_target(TARGET advection)
endif()

target_link_libraries(advection
    PUBLIC
    compiler_wrapper
    adv1d::config
    adv1d::core
    adv1d::tools)

configure_file(advection.ini advection.ini COPYONLY)
