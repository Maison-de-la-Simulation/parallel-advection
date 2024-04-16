// #include "gtest/gtest.h"
// #include <sycl/sycl.hpp>
// #include <AdvectionParams.h>
// #include <init.h>

// TEST(Impl, BasicRange){
//     ConfigMap configMap("advection_test.ini"); //might not exist and use default
//     ADVParamsNonCopyable strParams;
//     strParams.setup(configMap);

//     sycl::device d = sycl::device{sycl::cpu_selector_v}; // tests on the CPU

//     ADVParams params(strParams);
//     const auto nx = params.nx;
//     const auto nvx = params.nvx;
//     const auto nz = params.nz;
//     const auto maxIter = params.maxIter;

//     sycl::buffer<double, 3> buff_fdistrib(sycl::range<3>(nvx, nx, nz));

//     sycl::queue Q{d};
//     fill_buffer(Q, buff_fdistrib, params);
// }


