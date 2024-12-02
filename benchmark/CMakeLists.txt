add_executable(main-bench main_bench.cpp)

target_link_libraries(main-bench PUBLIC
benchmark::benchmark
dpcpp_opt
adv1d::config
adv1d::core
adv1d::tools)

if(SYCL_IS_ACPP)
    add_sycl_to_target(TARGET main-bench)
else()
    target_link_libraries(main-bench PUBLIC dpcpp_opt)
endif()
#
# add_executable(submission-bench submission_bench.cpp)

# target_link_libraries(submission-bench PUBLIC
# benchmark::benchmark
# dpcpp_opt
# adv1d::config
# adv1d::core
# adv1d::tools)


# add_executable(stream-bench stream_bench.cpp)

# target_link_libraries(stream-bench PUBLIC
# benchmark::benchmark
# dpcpp_opt
# adv1d::config
# adv1d::core
# adv1d::tools)

# #print sycl infos
# add_executable(print-sycl-infos print_sycl_infos.cpp)
# target_link_libraries(print-sycl-infos PUBLIC dpcpp_opt)
