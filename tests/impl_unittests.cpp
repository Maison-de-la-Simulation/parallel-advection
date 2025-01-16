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
TEST(Impl, Scoped) {
    ADVParams params;
    sycl::queue Q;
    Solver solver(params);

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::Scoped>();

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, solver).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, Exp1) {
    ADVParams params;
    sycl::queue Q;
    Solver solver(params);

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::Exp1>(params, Q);

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, solver).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, Exp2) {
    ADVParams params;
    sycl::queue Q;
    Solver solver(params);

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::Exp2>(params, params.percent_loc, Q);

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, solver).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, Exp3) {
    ADVParams params;
    sycl::queue Q;
    Solver solver(params);

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::Exp3>(params, Q);

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, solver).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, Exp4) {
    ADVParams params;
    sycl::queue Q;
    Solver solver(params);

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::Exp4>(params);

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, solver).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, Exp5) {
    ADVParams params;
    sycl::queue Q;
    Solver solver(params);

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::Exp5>(params);

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, solver).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, Exp6) {
    ADVParams params;
    sycl::queue Q;
    Solver solver(params);

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::Exp6>(params, Q);

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, solver).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
TEST(Impl, Exp7) {
    ADVParams params;
    sycl::queue Q;
    Solver solver(params);

    double *fdist =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
    fill_buffer(Q, fdist, params);

    auto advector = sref::make_unique<AdvX::Exp7>(params, Q);

    for (size_t i = 0; i < params.maxIter; ++i)
        advector(Q, fdist, solver).wait_and_throw();

    auto err = validate_result(Q, fdist, params);
    ASSERT_NEAR(err, 0, EPS);
}

// =============================================================================
// TEST(Impl, StraddledMalloc) {
//     ADVParams params;
//     sycl::queue Q;
//     Solver solver(params);

//     double *fdist =
//         sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
//     fill_buffer(Q, fdist, params);

//     auto advector = sref::make_unique<AdvX::StraddledMalloc>();

//     for (size_t i = 0; i < params.maxIter; ++i)
//         advector(Q, fdist, solver).wait_and_throw();

//     auto err = validate_result(Q, fdist, params);
//     ASSERT_NEAR(err, 0, EPS);
// }

// =============================================================================
// TEST(Impl, StreamY) {
//     ADVParams params;
//     sycl::queue Q;
//     Solver solver(params);

//     double *fdist =
//         sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
//     fill_buffer(Q, fdist, params);

//     auto advector = sref::make_unique<AdvX::StreamY>();

//     for (size_t i = 0; i < params.maxIter; ++i)
//         advector(Q, fdist, solver).wait_and_throw();

//     auto err = validate_result(Q, fdist, params);
//     ASSERT_NEAR(err, 0, EPS);
// }

// =============================================================================
// TEST(Impl, Exp1) {
//     ADVParams params;
//     sycl::queue Q;
//     Solver solver(params);

//     double *fdist =
//         sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);
//     fill_buffer(Q, fdist, params);

//     auto advector = sref::make_unique<AdvX::Exp1>(params, Q);

//     for (size_t i = 0; i < params.maxIter; ++i)
//         advector(Q, fdist, solver).wait_and_throw();

//     auto err = validate_result(Q, fdist, params);
//     ASSERT_NEAR(err, 0, EPS);
// }
