# add_executable(paper-bench paper-bench.cpp)

# target_link_libraries(paper-bench PUBLIC
# benchmark::benchmark
# dpcpp_opt
# adv1d::config
# adv1d::core
# adv1d::tools)

# if(SYCL_IS_ACPP)
#     add_sycl_to_target(TARGET paper-bench)
# else()
#     target_link_libraries(paper-bench PUBLIC dpcpp_opt)
# endif()
#


add_executable(conv1d-bench conv1d_bench.cpp)

target_link_libraries(conv1d-bench PUBLIC
benchmark::benchmark
# dpcpp_opt
adv1d::config
adv1d::core
adv1d::tools)

if(SYCL_IS_ACPP)
    add_sycl_to_target(TARGET conv1d-bench)
else()
    target_link_libraries(conv1d-bench PUBLIC dpcpp_opt)
endif()