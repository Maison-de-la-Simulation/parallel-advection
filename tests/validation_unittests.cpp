#include "advectors.h"
#include "validation.h"
#include "gtest/gtest.h"
#include <hipSYCL/sycl/libkernel/accessor.hpp>
#include <sycl/sycl.hpp>
#include <AdvectionParams.h>
#include <init.h>

TEST(Validation, ValidateNoIteration){
    sycl::device d = sycl::device{sycl::cpu_selector_v}; // tests on the CPU
    std::srand(static_cast<unsigned>(std::time(0)));

    ADVParams params;
    params.nx  = 1 + std::rand() % 1024;
    params.nvx = 1 + std::rand() % 128;
    params.nz  = 1 + std::rand() % 64;

    params.maxIter = 0;
    params.update_deltas();

    const sycl::range<3> r3d(params.nvx, params.nx, params.nz);
    sycl::buffer<double, 3> buff_fdistrib(r3d);

    sycl::queue Q{d};
    fill_buffer(Q, buff_fdistrib, params);

    auto res = validate_result(Q, buff_fdistrib, params, false);

    EXPECT_EQ(res, 0);
}

TEST(Validation, ValidateEachIterFor10Iterations){
    std::srand(static_cast<unsigned>(std::time(0)));

    ADVParams params;
    params.nx  = 512;
    params.nvx = 32;
    params.nz  = 32;
    params.update_deltas();

    const sycl::range<3> r3d(params.nvx, params.nx, params.nz);
    sycl::buffer<double, 3> buff_fdistrib(r3d);

    sycl::queue Q;
    fill_buffer(Q, buff_fdistrib, params);

    /* Creating a BasicRange advector */
    auto advector = sref::make_unique<AdvX::BasicRange>(params.nx, params.nvx, params.nz);

    double err;

    std::cerr << "(nvx, nx, nz) = (" << params.nvx << ", " << params.nx << ", " << params.nz << ")"  << std::endl;

    params.maxIter = 1;
    for(auto it=0; it < 10; ++it){
        advector(Q, buff_fdistrib, params).wait_and_throw();

        err = validate_result(Q, buff_fdistrib, params, false);
        Q.wait();
        EXPECT_NEAR(err, 0, 1e-4);

        params.maxIter++;
        std::cerr << err << std::endl;
    }

    err = validate_result(Q, buff_fdistrib, params, false);
    EXPECT_NEAR(err, 0, 1e-4);
}

TEST(Validation, ValidateNIterations){
    std::srand(static_cast<unsigned>(std::time(0)));

    ADVParams params;
    params.nx  = 1024;
    params.nvx = 16;
    params.nz  = 16;

    params.maxIter = 1 + std::rand() % 100;
    params.update_deltas();

    const sycl::range<3> r3d(params.nvx, params.nx, params.nz);
    sycl::buffer<double, 3> buff_fdistrib(r3d);

    sycl::queue Q;
    fill_buffer(Q, buff_fdistrib, params);

    auto advector = sref::make_unique<AdvX::BasicRange>(params.nx, params.nvx, params.nz);

    for(auto it=0; it<params.maxIter; ++it)
        advector(Q, buff_fdistrib, params).wait_and_throw();

    auto err = validate_result(Q, buff_fdistrib, params, false);

    EXPECT_NEAR(err, 0, 1e-6);
}