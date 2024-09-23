#include "validation.h"
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

    std::vector<double> all_l1_errors(params.ny1);
    for (size_t iy1=0; iy1 < params.ny1; iy1++) {

        double errorL1 = 0.0;

        std::vector<double> errorsL1(params.ny1);
        {

            sycl::buffer<double> errorl1_buff(&errorL1, 1);

            Q.submit([&](sycl::handler &cgh) {
                 // Input values to reductions are standard accessors
                 auto fdist =
                     buff_fdistrib.get_access<sycl::access_mode::read>(cgh);

#ifdef SYCL_IMPLEMENTATION_ONEAPI   // for DPCPP
                 auto errorl1_reduc =
                     sycl::reduction(errorl1_buff, cgh, sycl::plus<>());
#else // for openSYCL
                 sycl::accessor errorl1_acc(errorl1_buff, cgh,
                                            sycl::read_write);
                 auto errorl1_reduc =
                     sycl::reduction(errorl1_acc, sycl::plus<double>());
#endif

                 cgh.parallel_for(buff_fdistrib.get_range(), errorl1_reduc,
                                  [=](auto itm, auto &errorl1_reduc) {
                                      auto ix = itm[1];
                                      auto iy = itm[0];
                                      auto f = fdist[iy][ix][iy1];

                                      double const x = minRealX + ix * dx;
                                      double const v = minRealVx + iy * dvx;
                                      double const t = maxIter * dt;

                                      auto value =
                                          sycl::sin(4 * M_PI * (x - v * t));

                                      auto err = sycl::fabs(f - value);
                                      errorl1_reduc += err;
                                  });
             });
        }

        all_l1_errors[iy1] = errorL1 / (params.nx * params.ny);
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
// validate_result(sycl::queue &Q, sycl::buffer<double, 3> &buff_fdistrib,
//                 const ADVParams &params, bool do_print) {

//     auto const dx = params.dx;
//     auto const dvx = params.dvx;
//     auto const dt = params.dt;
//     auto const minRealX = params.minRealX;
//     auto const minRealVx = params.minRealVx;
//     auto const maxIter = params.maxIter;
//     auto const ny1 = params.ny1;

//     // Contains all the sum reductions in the ny1 dimension
//     sycl::buffer<double, 1> buff_sums(sycl::range(params.ny1));
//     // min at 0 max at 1
//     sycl::buffer<double, 1> buff_min_max(sycl::range(2));

//     sycl::buffer<double, 3> buff_err(buff_fdistrib.get_range());

//     sycl::nd_range<3> ndr(sycl::range(params.ny, params.nx, params.ny1),
//                           sycl::range(params.ny, params.nx, 1));

//     Q.submit([&](sycl::handler &cgh) {
//          sycl::accessor fdist{buff_fdistrib, cgh, sycl::read_only};
//          sycl::accessor ferr{buff_err, cgh, sycl::read_write,
//                              sycl::no_init};
//          sycl::accessor outputValues{buff_sums, cgh, sycl::write_only,
//                                      sycl::no_init};
//          sycl::accessor min_max{buff_min_max, cgh, sycl::write_only,
//                                 sycl::no_init};

//          cgh.parallel_for(ndr, [=](sycl::nd_item<3> itm) {
//              const size_t iy = itm.get_global_id(0);
//              const size_t ix = itm.get_global_id(1);
//              const size_t iy1 = itm.get_group(2);

//              double const x = minRealX + ix * dx;
//              double const v = minRealVx + iy * dvx;
//              double const t = maxIter * dt;

//              auto value = sycl::sin(4 * M_PI * (x - v * t));
//              auto err = sycl::fabs(fdist[iy][ix][iy1] - value);
//              ferr[iy][ix][iy1] = err;

            //  sycl::group_barrier(itm.get_group());

            // //  //  outputValues[iy1] =
            // //  //  reduce_over_group(itm.get_group(), ferr[iy][ix][iy1],
            // //  //  sycl::plus<double>());
            // //  outputValues[iy1] = 10e-16;

            // //  sycl::group_barrier(itm.get_group());

//             //  double *first = outputValues.get_pointer();
//             //  double *last = first + ny1;
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
