add_executable(conv1d-bench conv1d_bench.cpp)

target_link_libraries(conv1d-bench PUBLIC
benchmark::benchmark
bkma::config
bkma::core)

target_include_directories(conv1d-bench
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/tools
  ${CMAKE_SOURCE_DIR}/src/config
  ${CMAKE_SOURCE_DIR}/src/core
  ${CMAKE_SOURCE_DIR}/benchmark
)

if(SYCL_IS_ACPP)
    add_sycl_to_target(TARGET conv1d-bench)
else()
    target_link_libraries(conv1d-bench PUBLIC dpcpp_opt)
endif()