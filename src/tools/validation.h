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



    auto const dx = params.dx;
    auto const dvx = params.dvx;
    auto const dt = params.dt;
    auto const minRealX = params.minRealX;
    auto const minRealVx = params.minRealVx;
    auto const maxIter = params.maxIter;
    double errorL1 = 0.0;
    {

        sycl::buffer<double> errorl1_buff(&errorL1, 1);

        Q.submit([&](sycl::handler &cgh) {
             // Input values to reductions are standard accessors
             auto fdist =
                 buff_fdistrib.get_access<sycl::access_mode::read>(cgh);


#ifdef SYCL_IMPLEMENTATION_ONEAPI   // for DPCPP
             auto errorl1_reduc =
                 sycl::reduction(errorl1_buff, cgh, sycl::plus<>());
#else   // for openSYCL
             sycl::accessor errorl1_acc(errorl1_buff, cgh, sycl::read_write);
             auto errorl1_reduc =
                 sycl::reduction(errorl1_acc, sycl::plus<double>());
#endif

             cgh.parallel_for(
                 buff_fdistrib.get_range(), errorl1_reduc,
                 [=](auto itm, auto &errorl1_reduc) {
                     auto ix = itm[1];
                     auto ivx = itm[0];
                     auto f = fdist[itm];

                     double const x = minRealX + ix * dx;
                     double const v = minRealVx + ivx * dvx;
                     double const t = maxIter * dt;

                     auto value = sycl::sin(4 * M_PI * (x - v * t));

                     auto err = sycl::fabs(f - value);
                     errorl1_reduc += err;
                 });
         }).wait_and_throw();
    }

    std::cout << "Total cumulated error: "
              << errorL1 * dx * dvx << "\n"
              << std::endl;

}   // end validate_results