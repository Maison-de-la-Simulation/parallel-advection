add_subdirectory(config)
add_subdirectory(core)
add_subdirectory(tools)

add_executable(advection main.cpp)

target_link_libraries(advection
PUBLIC
adv1d::config
adv1d::core
adv1d::tools)

if(SYCL_IS_ACPP)
    add_sycl_to_target(TARGET advection)
else()
    target_link_libraries(advection PUBLIC dpcpp_opt)
endif()

configure_file(advection.ini advection.ini COPYONLY)
