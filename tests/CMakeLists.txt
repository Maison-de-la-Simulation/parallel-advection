add_executable(advection-tests
init_unittests.cpp
validation_unittests.cpp
impl_unittests.cpp
)

target_link_libraries(advection-tests PUBLIC
    GTest::gtest_main
    bkma::config
    bkma::core
)

target_include_directories(advection-tests
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/tools
)

if(SYCL_IS_ACPP)
    add_sycl_to_target(TARGET advection-tests)
else()
    target_link_libraries(advection-tests PUBLIC dpcpp_opt)
endif()

include(GoogleTest)
gtest_discover_tests(advection-tests)
