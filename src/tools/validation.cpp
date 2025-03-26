#include "validation.h"
#include "advectors.h"
#include <iostream>

// ==========================================
// ==========================================
double
validate_result(sycl::queue &Q, double *fdist_dev, const ADVParams &params,
                bool do_print) {

    auto const dx = params.dx;
    auto const dvx = params.dvx;
    auto const dt = params.dt;
    auto const minRealX = params.minRealX;
    auto const minRealVx = params.minRealVx;
    auto const maxIter = params.maxIter;

    sycl::range const r2d(params.n0, params.n1);

    std::vector<double> all_l1_errors(params.n2);
    for (size_t i2 = 0; i2 < params.n2; i2++) {

        double errorL1 = 0.0;

        std::vector<double> errorsL1(params.n2);
        {

            sycl::buffer<double> errorl1_buff(&errorL1, 1);

            Q.submit([&](sycl::handler &cgh) {

#ifdef SYCL_IMPLEMENTATION_ONEAPI   // for DPCPP
                 auto errorl1_reduc =
                     sycl::reduction(errorl1_buff, cgh, sycl::plus<>());
#else   // for openSYCL
                 sycl::accessor errorl1_acc(errorl1_buff, cgh,
                                            sycl::read_write);
                 auto errorl1_reduc =
                     sycl::reduction(errorl1_acc, sycl::plus<double>());
#endif

                 cgh.parallel_for(
                     r2d, errorl1_reduc, [=](auto itm, auto &errorl1_reduc) {
                         mdspan3d_t fdist(fdist_dev, params.n0, params.n1,
                                          params.n2);
                         auto i1 = itm[1];
                         auto i0 = itm[0];
                         auto f = fdist(i0, i1, i2);

                         double const x = minRealX + i1 * dx;
                         double const v = minRealVx + i0 * dvx;
                         double const t = maxIter * dt;

                         auto value = sycl::sin(4 * M_PI * (x - v * t));

                         auto err = sycl::fabs(f - value);
                         errorl1_reduc += err;
                     });
             }).wait();
        }

        all_l1_errors[i2] = errorL1 / (params.n1 * params.n0);
    }

    auto highest_l1 =
        *std::max_element(all_l1_errors.begin(), all_l1_errors.end());

    auto lowest_l1 =
        *std::min_element(all_l1_errors.begin(), all_l1_errors.end());

    if (do_print) {
        std::cout << "Highest L1 error found: " << highest_l1 << " (lowest is "
                  << lowest_l1 << ")\n"
                  << std::endl;
    }

    return highest_l1;
}   // end validate_results
