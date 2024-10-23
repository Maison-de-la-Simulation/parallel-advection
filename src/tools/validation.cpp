#include "validation.h"
#include "advectors.h"
#include <iostream>

// ==========================================
// ==========================================
double
validate_result(sycl::queue &Q, double* fdist_dev,
                const ADVParams &params, bool do_print) {

    auto const dx = params.dx;
    auto const dvx = params.dvx;
    auto const dt = params.dt;
    auto const minRealX = params.minRealX;
    auto const minRealVx = params.minRealVx;
    auto const maxIter = params.maxIter;

    sycl::range const r2d(params.n0, params.n1);

    std::vector<double> all_l1_errors(params.n2);
    for (size_t i2=0; i2 < params.n2; i2++) {

        double errorL1 = 0.0;

        std::vector<double> errorsL1(params.n2);
        {

            sycl::buffer<double> errorl1_buff(&errorL1, 1);

            Q.submit([&](sycl::handler &cgh) {
                 // Input values to reductions are standard accessors
                //  auto fdist =
                //      buff_fdistrib.get_access<sycl::access_mode::read>(cgh);

#ifdef SYCL_IMPLEMENTATION_ONEAPI   // for DPCPP
                 auto errorl1_reduc =
                     sycl::reduction(errorl1_buff, cgh, sycl::plus<>());
#else // for openSYCL
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

    if(do_print){
        std::cout << "Highest L1 error found: "
                << highest_l1 << " (lowest is " << lowest_l1 << ")\n"
                << std::endl;
    }

    return highest_l1;
}   // end validate_results

// ==========================================
// ==========================================
// double
// validate_result(sycl::queue &Q, double* fdist_dev,
//                 const ADVParams &params, bool do_print) {

//     auto const dx = params.dx;
//     auto const dvx = params.dvx;
//     auto const dt = params.dt;
//     auto const minRealX = params.minRealX;
//     auto const minRealVx = params.minRealVx;
//     auto const maxIter = params.maxIter;
//     auto const n2 = params.n2;

//     // Contains all the sum reductions in the n2 dimension
//     sycl::buffer<double, 1> buff_sums(sycl::range(params.n2));
//     // min at 0 max at 1
//     sycl::buffer<double, 1> buff_min_max(sycl::range(2));

//     sycl::buffer<double, 3> buff_err(buff_fdistrib.get_range());

//     sycl::nd_range<3> ndr(sycl::range(params.n0, params.n1, params.n2),
//                           sycl::range(params.n0, params.n1, 1));

//     Q.submit([&](sycl::handler &cgh) {
//          sycl::accessor fdist{buff_fdistrib, cgh, sycl::read_only};
//          sycl::accessor ferr{buff_err, cgh, sycl::read_write,
//                              sycl::no_init};
//          sycl::accessor outputValues{buff_sums, cgh, sycl::write_only,
//                                      sycl::no_init};
//          sycl::accessor min_max{buff_min_max, cgh, sycl::write_only,
//                                 sycl::no_init};

//          cgh.parallel_for(ndr, [=](sycl::nd_item<3> itm) {
//              const size_t i0 = itm.get_global_id(0);
//              const size_t i1 = itm.get_global_id(1);
//              const size_t i2 = itm.get_group(2);

//              double const x = minRealX + i1 * dx;
//              double const v = minRealVx + i0 * dvx;
//              double const t = maxIter * dt;

//              auto value = sycl::sin(4 * M_PI * (x - v * t));
//              auto err = sycl::fabs(fdist[i0][i1][i2] - value);
//              ferr[i0][i1][i2] = err;

            //  sycl::group_barrier(itm.get_group());

            // //  //  outputValues[i2] =
            // //  //  reduce_over_group(itm.get_group(), ferr[i0][i1][i2],
            // //  //  sycl::plus<double>());
            // //  outputValues[i2] = 10e-16;

            // //  sycl::group_barrier(itm.get_group());

//             //  double *first = outputValues.get_pointer();
//             //  double *last = first + n2;
//             //  double min = joint_reduce(itm.get_group(), first, last,
//             //                            sycl::minimum<double>());
//             //  double max = joint_reduce(itm.get_group(), first, last,
//             //                            sycl::maximum<double>());
//             //  min_max[0] = min;
//             //  min_max[1] = max;
//          });
//      }).wait();

//     auto highest_l1 = -1;
//     auto lowest_l1 = -1;
//     // sycl::host_accessor host_minmax{buff_min_max};
//     // highest_l1 = host_minmax[1];
//     // lowest_l1 = host_minmax[0];

//     if (do_print) {
//         std::cout << "Highest L1 error found: " << highest_l1 << " (lowest is "
//                   << lowest_l1 << ")\n"
//                   << std::endl;
//     }

//     return highest_l1;
// }   // end validate_results
