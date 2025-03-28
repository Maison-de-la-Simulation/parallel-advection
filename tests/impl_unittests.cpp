#include "advectors.hpp"
#include "validation.hpp"
#include "gtest/gtest.h"
#include <AdvectionParams.hpp>
#include <init.hpp>
#include <sycl/sycl.hpp>

static constexpr double EPS = 1e-6;

// =============================================================================
TEST(Impl, BasicRange) {
    ADVParams params;
    sycl::queue Q;
    Solver solver(params);

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector =
        sref::make_unique<AdvX::BasicRange>(params.n1, params.n0, params.n2);

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, solver).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, Hierarchical) {
    ADVParams params;
    sycl::queue Q;
    Solver solver(params);

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::Hierarchical>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, solver).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, NDRange) {
    ADVParams params;
    sycl::queue Q;
    Solver solver(params);

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::NDRange>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, solver).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, HybridMem) {
    ADVParams params;
    sycl::queue Q;
    Solver solver(params);

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::HybridMem>(params, Q);

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, solver).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}
