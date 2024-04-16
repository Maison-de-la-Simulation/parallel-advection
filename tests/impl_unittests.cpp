#include "gtest/gtest.h"
#include <sycl/sycl.hpp>
#include <AdvectionParams.h>
#include <init.h>

struct TestParams {
    ADVParams params;

    void setup_params(){
        params.nx  = 512;
        params.nvx = 32;
        params.nz  = 32;

        params.maxIter = 50;
        params.update_deltas();
    }

    TestParams() : params{} {
        setup_params();
    }
};


TEST(Impl, BasicRange){
    TestParams tstp;

    const auto nx = tp.params.nx;
    const auto nvx = tp.params.nvx;
    const auto nz = params.nz;
    const auto maxIter = params.maxIter;

    sycl::buffer<double, 3> buff_fdistrib(sycl::range<3>(nvx, nx, nz));

    sycl::queue Q{d};
    fill_buffer(Q, buff_fdistrib, params);
}
