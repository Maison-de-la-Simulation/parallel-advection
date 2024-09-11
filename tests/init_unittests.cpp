#include "gtest/gtest.h"
#include <sycl/sycl.hpp>
#include <AdvectionParams.h>
#include <init.h>

static constexpr double EPS = 1e-6;

// =============================================================================
TEST(Init, FillBufferWithDefaultParams){
    ADVParams params;
    const auto nx  = params.nx;
    const auto ny = params.ny;
    const auto ny1  = params.ny1;
    params.update_deltas();

    sycl::buffer<double, 3> buff_fdistrib(sycl::range<3>(ny, nx, ny1));

    sycl::queue Q;
    fill_buffer(Q, buff_fdistrib, params);

    sycl::host_accessor fdist(buff_fdistrib, sycl::read_only);

    for (auto iy1 = 0; iy1 < ny1; ++iy1) {
        for (auto iv = 0; iv < ny; ++iv) {
            for (auto ix = 0; ix < nx; ++ix) {
                double x = params.minRealX + ix * params.dx;
                EXPECT_NEAR(fdist[iv][ix][iy1], sycl::sin(4 * x * M_PI), EPS);
            }
        }
    }
}

// =============================================================================
TEST(Init, FillBufferWithRandomParams){
    std::srand(static_cast<unsigned>(std::time(0)));

    ADVParams params;
    params.nx  = 1 + std::rand() % 1024;
    params.ny = 1 + std::rand() % 256;
    params.ny1  = 1 + std::rand() % 256;

    params.update_deltas();

    const sycl::range<3> r3d(params.ny, params.nx, params.ny1);
    sycl::buffer<double, 3> buff_fdistrib(r3d);

    sycl::queue Q;
    fill_buffer(Q, buff_fdistrib, params);

    sycl::host_accessor fdist(buff_fdistrib, sycl::read_only);

    for (auto iy1 = 0; iy1 < r3d.get(2); ++iy1) {
        for (auto ix = 0; ix < r3d.get(1); ++ix) {
            for (auto iv = 0; iv < r3d.get(0); ++iv) {
                double x = params.minRealX + ix * params.dx;
                EXPECT_NEAR(fdist[iv][ix][iy1], sycl::sin(4 * x * M_PI), EPS);
            }
        }
    }
}
