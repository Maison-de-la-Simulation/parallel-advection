#include "advectors.h"
#include "validation.h"
#include "gtest/gtest.h"
#include <AdvectionParams.h>
#include <init.h>
#include <sycl/sycl.hpp>

static constexpr double EPS = 1e-6;

// =============================================================================
TEST(Impl, BasicRange) {
    ADVParams params;
    sycl::queue Q;

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector =
        sref::make_unique<AdvX::BasicRange>(params.n1, params.n0, params.n2);

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, params).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, Hierarchical) {
    ADVParams params;
    sycl::queue Q;

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::Hierarchical>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, params).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, NDRange) {
    ADVParams params;
    sycl::queue Q;

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::NDRange>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, params).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, Scoped) {
    ADVParams params;
    sycl::queue Q;

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::Scoped>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, params).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, ReverseIndexes) {
    ADVParams params;
    sycl::queue Q;

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::ReverseIndexes>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, params).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, SeqTwoDimWG) {
    ADVParams params;
    sycl::queue Q;

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::SeqTwoDimWG>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, params).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, StraddledMalloc) {
    ADVParams params;
    sycl::queue Q;

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::StraddledMalloc>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, params).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, StreamY) {
    ADVParams params;
    sycl::queue Q;

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::StreamY>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, params).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, TwoDimWG) {
    ADVParams params;
    sycl::queue Q;

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::TwoDimWG>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, params).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, Exp1) {
    ADVParams params;
    sycl::queue Q;

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::Exp1>(params, Q);

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, params).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}
