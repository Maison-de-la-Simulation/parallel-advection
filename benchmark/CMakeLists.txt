find_package(benchmark REQUIRED)
# add_subdirectory(thirdparty/benchmark)


add_executable(main-bench run_bench.cpp)

target_link_libraries(main-bench PUBLIC
    benchmark::benchmark
    compiler_wrapper
    adv1d::config
    adv1d::core
    adv1d::tools)