#include "validation.h"
#include <algorithm>
#include <iostream>
// ==========================================
// ==========================================
double
validate_result(sycl::queue &Q, sycl::buffer<double, 3> &buff_fdistrib,
                const ADVParams &params, bool do_print) {

    auto const dx = params.dx;
    auto const dvx = params.dvx;
    auto const dt = params.dt;
    auto const minRealX = params.minRealX;
    auto const minRealVx = params.minRealVx;
    auto const maxIter = params.maxIter;

    std::vector<double> all_l1_errors(params.ns);
    for (size_t iz=0; iz < params.ns; iz++) {

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
                 sycl::accessor errorl1_acc(errorl1_buff, cgh,
                                            sycl::read_write);
                 auto errorl1_reduc =
                     sycl::reduction(errorl1_acc, sycl::plus<double>());
#endif

                 cgh.parallel_for(buff_fdistrib.get_range(), errorl1_reduc,
                                  [=](auto itm, auto &errorl1_reduc) {
                                      auto ix = itm[1];
                                      auto ivx = itm[0];
                                      auto f = fdist[itm];

                                      double const x = minRealX + ix * dx;
                                      double const v = minRealVx + ivx * dvx;
                                      double const t = maxIter * dt;

                                      auto value =
                                          sycl::sin(4 * M_PI * (x - v * t));

                                      auto err = sycl::fabs(f - value);
                                      errorl1_reduc += err;
                                  });
             });
        }

        all_l1_errors[iz] = errorL1 / (params.nx * params.nb);
    }

    auto highest_l1 =
        *std::max_element(all_l1_errors.begin(), all_l1_errors.end());

    auto lowest_l1 =
        *std::min_element(all_l1_errors.begin(), all_l1_errors.end());

    if(do_print){
        std::cout << "Highest L1 error found: "
                << highest_l1 << " (lowest is " << lowest_l1 << ")\n"
                << std::endl;
    }

    return highest_l1;
}   // end validate_results
