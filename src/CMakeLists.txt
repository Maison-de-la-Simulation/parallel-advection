add_subdirectory(config)
add_subdirectory(core)

add_executable(advection main.cpp)

target_link_libraries(advection
PUBLIC
adv1d::config)

target_include_directories(advection
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/tools
  ${CMAKE_SOURCE_DIR}/src/core
)

if(SYCL_IS_ACPP)
    add_sycl_to_target(TARGET advection)
else()
    target_link_libraries(advection PUBLIC dpcpp_opt)
endif()

configure_file(advection.ini advection.ini COPYONLY)
