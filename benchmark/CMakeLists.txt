find_package(benchmark REQUIRED)

add_executable(main-bench main_bench.cpp)

target_link_libraries(main-bench PUBLIC
benchmark::benchmark
compiler_wrapper
adv1d::config
adv1d::core
adv1d::tools)

#
add_executable(submission-bench submission_bench.cpp)

target_link_libraries(submission-bench PUBLIC
benchmark::benchmark
compiler_wrapper
adv1d::config
adv1d::core
adv1d::tools)

#
add_executable(stream-bench stream_bench.cpp)

target_link_libraries(stream-bench PUBLIC
benchmark::benchmark
compiler_wrapper
adv1d::config
adv1d::core
adv1d::tools)

#print sycl infos
add_executable(print-sycl-infos print_sycl_infos.cpp)
target_link_libraries(print-sycl-infos PUBLIC compiler_wrapper)
