#pragma once

#include "init.h"
#include <AdvectionParams.h>
#include <iostream>
#include <sycl/sycl.hpp>

// ==========================================
// ==========================================
void
validate_result(sycl::queue &Q, sycl::buffer<double, 2> &buff_fdistrib,
                const ADVParams &params) noexcept {

    const double acceptable_error = 1e-5;

    int errCount = 0;
    double errorL1 = 0.0;
    {
        sycl::buffer<double> buff_errorL1{&errorL1, 1};

        Q.submit([&](sycl::handler &cgh) {
             // Input values to reductions are standard accessors
             auto fdist =
                 buff_fdistrib.get_access<sycl::access_mode::read>(cgh);


#ifdef __INTEL_LLVM_COMPILER   // for DPCPP
             auto reduc_errorL1 =
                 sycl::reduction(buff_errorL1, cgh, sycl::plus<>());
#else   // for openSYCL
             auto acc_errorL1 =
                 buff_errorL1.get_access<sycl::access_mode::write>(cgh);
             auto reduc_errorL1 =
                 sycl::reduction(acc_errorL1, sycl::plus<double>());
#endif

             cgh.parallel_for(
                 buff_fdistrib.get_range(), reduc_errorL1,
                 [=](auto itm, auto &reduc_errorL1) {
                     auto ix = itm[1];
                     auto ivx = itm[0];
                     auto f = fdist[itm];

                     double const x = params.minRealX + ix * params.dx;
                     double const v = params.minRealVx + ivx * params.dvx;
                     double const t = params.maxIter * params.dt;

                     auto value = sycl::sin(4 * M_PI * (x - v * t));

                     auto err = sycl::fabs(f - value);
                     reduc_errorL1 += err;
                 });
         }).wait_and_throw();
    }

    std::cout << "Total cumulated error: "
              << errorL1 * params.dx * params.dvx << "\n"
              << std::endl;

}   // end validate_results