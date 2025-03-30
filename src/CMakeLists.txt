add_subdirectory(config)
add_subdirectory(core)

function(add_bkma_executable name)
  add_executable(${name} ${name}.cpp)

  target_link_libraries(${name}
  PUBLIC
  bkma::config
  bkma::core)

  target_include_directories(${name}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src/tools
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/solvers
    ${CMAKE_SOURCE_DIR}/src
  )

  if(SYCL_IS_ACPP)
      add_sycl_to_target(TARGET ${name})
  else()
      target_link_libraries(${name} PUBLIC dpcpp_opt)
  endif()

  configure_file(${name}.ini ${name}.ini COPYONLY)
endfunction()

# Add executables
add_bkma_executable(advection)
add_bkma_executable(conv1d)


### Advection
# add_executable(advection advection.cpp)

# target_link_libraries(advection
# PUBLIC
# bkma::config)

# target_include_directories(advection
#   PUBLIC
#   ${CMAKE_SOURCE_DIR}/src/tools
#   ${CMAKE_SOURCE_DIR}/src/core
#   ${CMAKE_SOURCE_DIR}/src/solvers
# )

# if(SYCL_IS_ACPP)
#     add_sycl_to_target(TARGET advection)
# else()
#     target_link_libraries(advection PUBLIC dpcpp_opt)
# endif()

# configure_file(advection.ini advection.ini COPYONLY)

# ### Conv1d
# add_executable(conv1d conv1d.cpp)

# target_link_libraries(conv1d
# PUBLIC
# bkma::config)

# target_include_directories(conv1d
#   PUBLIC
#   ${CMAKE_SOURCE_DIR}/src/tools
#   ${CMAKE_SOURCE_DIR}/src/core
#   ${CMAKE_SOURCE_DIR}/src/solvers
# )

# if(SYCL_IS_ACPP)
#     add_sycl_to_target(TARGET conv1d)
# else()
#     target_link_libraries(conv1d PUBLIC dpcpp_opt)
# endif()

# configure_file(conv1d.ini conv1d.ini COPYONLY)
