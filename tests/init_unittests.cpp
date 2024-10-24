#include "advectors.h"
#include "gtest/gtest.h"
#include <AdvectionParams.h>
#include <init.h>
#include <sycl/sycl.hpp>

static constexpr double EPS = 1e-6;

// =============================================================================
TEST(Init, FillBufferWithDefaultParams) {
    ADVParams params;
    const auto n1 = params.n1;
    const auto n0 = params.n0;
    const auto n2 = params.n2;
    params.update_deltas();
    sycl::queue Q;

    double *fdist_host =
        sycl::malloc_host<double>(n0 * n1 * n2, Q);
    mdspan3d_t fdist(fdist_host, n0, n1, n2);

    fill_buffer(Q, fdist_host, params);

    // sycl::host_accessor fdist(buff_fdistrib, sycl::read_only);

    for (size_t i2 = 0; i2 < n2; ++i2) {
        for (size_t iv = 0; iv < n0; ++iv) {
            for (size_t i1 = 0; i1 < n1; ++i1) {
                double x = params.minRealX + i1 * params.dx;
                EXPECT_NEAR(fdist(iv, i1, i2), sycl::sin(4 * x * M_PI), EPS);
            }
        }
    }
}

// =============================================================================
TEST(Init, FillBufferWithRandomParams) {
    std::srand(static_cast<unsigned>(std::time(0)));

    ADVParams params;
    params.n1 = 1 + std::rand() % 1024;
    params.n0 = 1 + std::rand() % 256;
    params.n2 = 1 + std::rand() % 256;

    params.update_deltas();

    sycl::queue Q;
    const sycl::range<3> r3d(params.n0, params.n1, params.n2);
    double *fdist_host =
        sycl::malloc_host<double>(params.n0 * params.n1 * params.n2, Q);
    mdspan3d_t fdist(fdist_host, params.n0, params.n1, params.n2);

    fill_buffer(Q, fdist_host, params);

    for (size_t i2 = 0; i2 < r3d.get(2); ++i2) {
        for (size_t i1 = 0; i1 < r3d.get(1); ++i1) {
            for (size_t iv = 0; iv < r3d.get(0); ++iv) {
                double x = params.minRealX + i1 * params.dx;
                EXPECT_NEAR(fdist(iv, i1, i2), sycl::sin(4 * x * M_PI), EPS);
            }
        }
    }
}
