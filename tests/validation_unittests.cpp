#include "advectors.h"
#include "validation.h"
#include "gtest/gtest.h"
#include <sycl/sycl.hpp>
#include <AdvectionParams.h>
#include <init.h>

static constexpr double EPS = 1e-6;

// =============================================================================
TEST(Validation, ValidateNoIteration){
    std::srand(static_cast<unsigned>(std::time(0)));

    ADVParams params;
    params.nx  = 1 + std::rand() % 1024;
    params.ny = 1 + std::rand() % 128;
    params.ny1  = 1 + std::rand() % 64;

    params.maxIter = 0;
    params.update_deltas();

    const sycl::range<3> r3d(params.ny, params.nx, params.ny1);
    sycl::buffer<double, 3> buff_fdistrib(r3d);

    sycl::queue Q;
    fill_buffer(Q, buff_fdistrib, params);

    auto res = validate_result(Q, buff_fdistrib, params, false);

    EXPECT_EQ(res, 0);
}

// =============================================================================
TEST(Validation, ValidateEachIterFor10Iterations){
    ADVParams params;
    params.nx  = 512;
    params.ny = 1 + std::rand() % 30;
    params.ny1  = 1 + std::rand() % 30;
    params.update_deltas();

    const sycl::range<3> r3d(params.ny, params.nx, params.ny1);
    sycl::buffer<double, 3> buff_fdistrib(r3d);

    sycl::queue Q;
    fill_buffer(Q, buff_fdistrib, params);

    /* Creating a BasicRange advector */
    auto advector = sref::make_unique<AdvX::BasicRange>(params.nx, params.ny, params.ny1);

    double err;
    params.maxIter = 0;
    for(size_t it=0; it < 10; ++it){
        params.maxIter++;

        advector(Q, buff_fdistrib, params).wait_and_throw();

        err = validate_result(Q, buff_fdistrib, params, false);
        Q.wait();
        EXPECT_NEAR(err, 0, EPS);
    }

    err = validate_result(Q, buff_fdistrib, params, false);
    EXPECT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Validation, ValidateNIterations){
    ADVParams params;
    params.nx  = 1024;
    params.ny = 16;
    params.ny1  = 16;

    params.maxIter = 1 + std::rand() % 100;
    params.update_deltas();

    const sycl::range<3> r3d(params.ny, params.nx, params.ny1);
    sycl::buffer<double, 3> buff_fdistrib(r3d);

    sycl::queue Q;
    fill_buffer(Q, buff_fdistrib, params);

    auto advector = sref::make_unique<AdvX::BasicRange>(params.nx, params.ny, params.ny1);

    for(size_t it=0; it<params.maxIter; ++it)
        advector(Q, buff_fdistrib, params).wait_and_throw();

    auto err = validate_result(Q, buff_fdistrib, params, false);
    Q.wait();

    EXPECT_NEAR(err, 0, EPS);
}
