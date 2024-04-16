#include "validation.h"
#include "gtest/gtest.h"
#include <sycl/sycl.hpp>
#include <AdvectionParams.h>
#include <init.h>


TEST(Impl, BasicRange){
    ADVParams params;
    sycl::queue Q;

    sycl::range<3> r3d(params.nvx,params.nx, params.nz);
    sycl::buffer<double, 3> buff_fdistrib(r3d);
    fill_buffer(Q, buff_fdistrib, params);

    auto advector = sref::make_unique<AdvX::BasicRange>(params.nx, params.nvx, params.nz);

    for (int i = 0; i < params.maxIter; ++i)
        advector(Q, buff_fdistrib, params).wait_and_throw();

    auto err = validate_result(Q, buff_fdistrib, params);
    ASSERT_NEAR(err, 0, 1e-6);
}

TEST(Impl, Hierarchical){
    ADVParams params;
    sycl::queue Q;

    sycl::range<3> r3d(params.nvx,params.nx, params.nz);
    sycl::buffer<double, 3> buff_fdistrib(r3d);
    fill_buffer(Q, buff_fdistrib, params);

    auto advector = sref::make_unique<AdvX::Hierarchical>();

    for (int i = 0; i < params.maxIter; ++i)
        advector(Q, buff_fdistrib, params).wait_and_throw();

    auto err = validate_result(Q, buff_fdistrib, params);
    ASSERT_NEAR(err, 0, 1e-6);
}

TEST(Impl, NDRange){
    ADVParams params;
    sycl::queue Q;

    sycl::range<3> r3d(params.nvx,params.nx, params.nz);
    sycl::buffer<double, 3> buff_fdistrib(r3d);
    fill_buffer(Q, buff_fdistrib, params);

    auto advector = sref::make_unique<AdvX::NDRange>();

    for (int i = 0; i < params.maxIter; ++i)
        advector(Q, buff_fdistrib, params).wait_and_throw();

    auto err = validate_result(Q, buff_fdistrib, params);
    ASSERT_NEAR(err, 0, 1e-6);
}

TEST(Impl, Scoped){
    ADVParams params;
    sycl::queue Q;

    sycl::range<3> r3d(params.nvx,params.nx, params.nz);
    sycl::buffer<double, 3> buff_fdistrib(r3d);
    fill_buffer(Q, buff_fdistrib, params);

    auto advector = sref::make_unique<AdvX::Scoped>();

    for (int i = 0; i < params.maxIter; ++i)
        advector(Q, buff_fdistrib, params).wait_and_throw();

    auto err = validate_result(Q, buff_fdistrib, params);
    ASSERT_NEAR(err, 0, 1e-6);
}
