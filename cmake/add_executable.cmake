function(add_bench_executable name)
    add_executable(${name} ${name}.cpp)
  
    target_link_libraries(${name}
    PUBLIC
    benchmark::benchmark
    bkma::config)
  
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
endfunction()


function(add_bkma_executable name)
    add_executable(${name} ${name}.cpp)
  
    target_link_libraries(${name}
    PUBLIC
    bkma::config)
  
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
