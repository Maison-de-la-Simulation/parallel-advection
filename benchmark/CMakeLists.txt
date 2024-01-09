find_package(benchmark REQUIRED)
# add_subdirectory(thirdparty/benchmark)


add_executable(main-bench run_bench.cpp)
target_link_libraries(main-bench benchmark::benchmark)