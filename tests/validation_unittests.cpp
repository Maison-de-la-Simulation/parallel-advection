#include "advectors.hpp"
#include "validation.hpp"
#include "gtest/gtest.h"
#include <sycl/sycl.hpp>
#include <AdvectionParams.hpp>
#include <init.hpp>

static constexpr double EPS = 1e-6;

// =============================================================================
TEST(Validation, ValidateNoIteration){
    std::srand(static_cast<unsigned>(std::time(0)));

    ADVParams params;
    params.n1  = 1 + std::rand() % 1024;
    params.n0 = 1 + std::rand() % 128;
    params.n2  = 1 + std::rand() % 64;

    params.maxIter = 0;
    params.update_deltas();

    sycl::queue Q;
    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);

    fill_buffer(Q, fdist, params);

    auto res = validate_result(Q, fdist, params, false);

    EXPECT_EQ(res, 0);
}

// =============================================================================
TEST(Validation, ValidateEachIterFor10Iterations){
    ADVParams params;
    params.n1  = 512;
    params.n0 = 1 + std::rand() % 30;
    params.n2  = 1 + std::rand() % 30;
    params.update_deltas();

    sycl::queue Q;
    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    Solver solver(params);

    /* Creating a BasicRange advector */
    auto advector = sref::make_unique<AdvX::BasicRange>(params.n1, params.n0, params.n2);

    double err;
    params.maxIter = 0;
    for(size_t it=0; it < 10; ++it){
        params.maxIter++;

        advector(Q, fdist, solver).wait_and_throw();

        err = validate_result(Q, fdist, params, false);
        Q.wait();
        EXPECT_NEAR(err, 0, EPS);
    }

    err = validate_result(Q, fdist, params, false);
    EXPECT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Validation, ValidateNIterations){
    ADVParams params;
    params.n1  = 1024;
    params.n0 = 16;
    params.n2  = 16;

    params.maxIter = 1 + std::rand() % 100;
    params.update_deltas();

    sycl::queue Q;
    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);

    fill_buffer(Q, fdist, params);
    
    Solver solver(params);

    auto advector = sref::make_unique<AdvX::BasicRange>(params.n1, params.n0, params.n2);

    for(size_t it=0; it<params.maxIter; ++it)
        advector(Q, fdist, solver).wait_and_throw();

    auto err = validate_result(Q, fdist, params, false);
    Q.wait();

    EXPECT_NEAR(err, 0, EPS);
}
