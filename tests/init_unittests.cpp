#include "gtest/gtest.h"
#include <sycl/sycl.hpp>
#include <AdvectionParams.h>
#include <init.h>

// =============================================================================
TEST(Init, FillBufferWithDefaultParams){
    sycl::device d = sycl::device{sycl::cpu_selector_v}; // tests on the CPU

    ADVParams params;
    const auto nx  = params.nx;
    const auto nvx = params.nvx;
    const auto nz  = params.nz;
    params.update_deltas();

    sycl::buffer<double, 3> buff_fdistrib(sycl::range<3>(nvx, nx, nz));

    sycl::queue Q{d};
    fill_buffer(Q, buff_fdistrib, params);

    sycl::host_accessor fdist(buff_fdistrib, sycl::read_only);

    for (auto iz = 0; iz < nz; ++iz) {
        for (auto iv = 0; iv < nvx; ++iv) {
            for (auto ix = 0; ix < nx; ++ix) {
                double x = params.minRealX + ix * params.dx;
                EXPECT_EQ(fdist[iv][ix][iz], sycl::sin(4 * x * M_PI));
            }
        }
    }
}

// =============================================================================
TEST(Init, FillBufferWithRandomParams){
    sycl::device d = sycl::device{sycl::cpu_selector_v}; // tests on the CPU
    std::srand(static_cast<unsigned>(std::time(0)));

    ADVParams params;
    params.nx  = 1 + std::rand() % 1024;
    params.nvx = 1 + std::rand() % 256;
    params.nz  = 1 + std::rand() % 256;

    params.update_deltas();

    const sycl::range<3> r3d(params.nvx, params.nx, params.nz);
    sycl::buffer<double, 3> buff_fdistrib(r3d);

    sycl::queue Q{d};
    fill_buffer(Q, buff_fdistrib, params);

    sycl::host_accessor fdist(buff_fdistrib, sycl::read_only);

    for (auto iz = 0; iz < r3d.get(2); ++iz) {
        for (auto ix = 0; ix < r3d.get(1); ++ix) {
            for (auto iv = 0; iv < r3d.get(0); ++iv) {
                double x = params.minRealX + ix * params.dx;
                EXPECT_EQ(fdist[iv][ix][iz], sycl::sin(4 * x * M_PI));
            }
        }
    }
}
