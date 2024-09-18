#include "advectors.h"
#include "gtest/gtest.h"
#include <AdvectionParams.h>
#include <sycl/sycl.hpp>
#include <init.h>
#include "validation.h"

static constexpr double EPS = 1e-6;

// =============================================================================
TEST(Impl, BasicRange) {
    ADVParams params;
    sycl::queue Q;

    sycl::range<3> r3d(params.ny, params.nx, params.ny1);
    sycl::buffer<double, 3> buff_fdistrib(r3d);
    fill_buffer(Q, buff_fdistrib, params);

    auto advector =
        sref::make_unique<AdvX::BasicRange>(params.nx, params.ny, params.ny1);

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, buff_fdistrib, params).wait_and_throw();

    auto err = validate_result(Q, buff_fdistrib, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, Hierarchical) {
    ADVParams params;
    sycl::queue Q;

    sycl::range<3> r3d(params.ny, params.nx, params.ny1);
    sycl::buffer<double, 3> buff_fdistrib(r3d);
    fill_buffer(Q, buff_fdistrib, params);

    auto advector = sref::make_unique<AdvX::Hierarchical>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, buff_fdistrib, params).wait_and_throw();

    auto err = validate_result(Q, buff_fdistrib, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, NDRange) {
    ADVParams params;
    sycl::queue Q;

    sycl::range<3> r3d(params.ny, params.nx, params.ny1);
    sycl::buffer<double, 3> buff_fdistrib(r3d);
    fill_buffer(Q, buff_fdistrib, params);

    auto advector = sref::make_unique<AdvX::NDRange>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, buff_fdistrib, params).wait_and_throw();

    auto err = validate_result(Q, buff_fdistrib, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, Scoped) {
    ADVParams params;
    sycl::queue Q;

    sycl::range<3> r3d(params.ny, params.nx, params.ny1);
    sycl::buffer<double, 3> buff_fdistrib(r3d);
    fill_buffer(Q, buff_fdistrib, params);

    auto advector = sref::make_unique<AdvX::Scoped>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, buff_fdistrib, params).wait_and_throw();

    auto err = validate_result(Q, buff_fdistrib, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, ReverseIndexes) {
    ADVParams params;
    sycl::queue Q;

    sycl::range<3> r3d(params.ny, params.nx, params.ny1);
    sycl::buffer<double, 3> buff_fdistrib(r3d);
    fill_buffer(Q, buff_fdistrib, params);

    auto advector = sref::make_unique<AdvX::ReverseIndexes>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, buff_fdistrib, params).wait_and_throw();

    auto err = validate_result(Q, buff_fdistrib, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, SeqTwoDimWG) {
    ADVParams params;
    sycl::queue Q;

    sycl::range<3> r3d(params.ny, params.nx, params.ny1);
    sycl::buffer<double, 3> buff_fdistrib(r3d);
    fill_buffer(Q, buff_fdistrib, params);

    auto advector = sref::make_unique<AdvX::SeqTwoDimWG>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, buff_fdistrib, params).wait_and_throw();

    auto err = validate_result(Q, buff_fdistrib, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, StraddledMalloc) {
    ADVParams params;
    sycl::queue Q;

    sycl::range<3> r3d(params.ny, params.nx, params.ny1);
    sycl::buffer<double, 3> buff_fdistrib(r3d);
    fill_buffer(Q, buff_fdistrib, params);

    auto advector = sref::make_unique<AdvX::StraddledMalloc>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, buff_fdistrib, params).wait_and_throw();

    auto err = validate_result(Q, buff_fdistrib, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, StreamY) {
    ADVParams params;
    sycl::queue Q;

    sycl::range<3> r3d(params.ny, params.nx, params.ny1);
    sycl::buffer<double, 3> buff_fdistrib(r3d);
    fill_buffer(Q, buff_fdistrib, params);

    auto advector = sref::make_unique<AdvX::StreamY>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, buff_fdistrib, params).wait_and_throw();

    auto err = validate_result(Q, buff_fdistrib, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, TwoDimWG) {
    ADVParams params;
    sycl::queue Q;

    sycl::range<3> r3d(params.ny, params.nx, params.ny1);
    sycl::buffer<double, 3> buff_fdistrib(r3d);
    fill_buffer(Q, buff_fdistrib, params);

    auto advector = sref::make_unique<AdvX::TwoDimWG>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, buff_fdistrib, params).wait_and_throw();

    auto err = validate_result(Q, buff_fdistrib, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, Exp1) {
    ADVParams params;
    sycl::queue Q;

    sycl::range<3> r3d(params.ny, params.nx, params.ny1);
    sycl::buffer<double, 3> buff_fdistrib(r3d);
    fill_buffer(Q, buff_fdistrib, params);

    auto advector = sref::make_unique<AdvX::Exp1>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, buff_fdistrib, params).wait_and_throw();

    auto err = validate_result(Q, buff_fdistrib, params);
    ASSERT_NEAR(err, 0, EPS);
}
