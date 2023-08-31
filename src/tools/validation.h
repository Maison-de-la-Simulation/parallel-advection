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
             auto acc_errorL1 =
                 buff_errorL1.get_access<sycl::access_mode::write>(cgh);

             auto reduc_errorL1 =
                 sycl::reduction(acc_errorL1, sycl::plus<double>());

             cgh.parallel_for(
                 buff_fdistrib.get_range(), reduc_errorL1,
                 [=](auto itm, auto &reduc_errorL1) {
                     auto ix = itm[1];
                     auto ivx = itm[0];
                     auto f = fdist[itm];

                     double const x = params.minRealx + ix * params.dx;
                     double const v = params.minRealVx + ivx * params.dVx;
                     double const t = params.maxIter * params.dt;

                     auto value = sycl::sin(4 * M_PI * (x - v * t));

                     auto err = sycl::fabs(f - value);
                     reduc_errorL1 += err;
                 });
         }).wait_and_throw();
    }

    std::cout << "Total cumulated error: "
              << errorL1 * params.dx * params.dVx << "\n"
              << std::endl;

}   // end validate_results

// // ==========================================
// // ==========================================
void
validate_distr_result(celerity::distr_queue &Q,
                      celerity::buffer<double, 2> &buff_fdistrib,
                      const ADVParams &params) noexcept {
    /* for now, get the celerity data and transform to a sycl buffer and
    then do the usual SYCL reduction in the host */

    Q.submit([&](celerity::handler &cgh) {
        celerity::accessor acc{buff_fdistrib, cgh, celerity::access::all{},
                               celerity::read_only_host_task};
        cgh.host_task(celerity::on_master_node, [=] {
            sycl::queue q;
            
            auto ptr = acc.get_pointer();
            sycl::buffer<double, 2> syclbuff(
                ptr, sycl::range<2>(params.nVx, params.nx));
            
            validate_result(q, syclbuff, params);
            q.wait();
        });
    });

}   // end validate_distr_result
// void
// validate_distr_result(celerity::distr_queue &Q,
//                       celerity::buffer<double, 2> &buff_fdistrib,
//                       const ADVParams &params) noexcept {

//     double errorL1 = 0.0;
//     {
//         // sycl::buffer<double> buff_errorL1{&errorL1, 1};

//         celerity::buffer<double, 1> buff_errorL1{&errorL1,
//                                                  celerity::range<1>{1}};

//         Q.submit([&](celerity::handler &cgh) {
//              celerity::accessor fdist(buff_fdistrib, cgh,
//                                       celerity::access::one_to_one{},
//                                       celerity::read_only);

//              auto reduc_errorL1 = celerity::reduction(
//                  buff_errorL1, cgh, sycl::plus<double>{});
//                 //  celerity::property::reduction::initialize_to_identity{});

//              cgh.parallel_for(
//                  buff_fdistrib.get_range(), reduc_errorL1,
//                  [=](celerity::item<2> itm, auto &reduc_errorL1) {
//                      auto ix = itm[1];
//                      auto ivx = itm[0];
//                      auto f = fdist[itm];

//                      double const x = params.minRealx + ix * params.dx;
//                      double const v = params.minRealVx + ivx * params.dVx;
//                      double const t = params.maxIter * params.dt;

//                      auto value = sycl::sin(4 * M_PI * (x - v * t));

//                      auto err = sycl::fabs(f - value);
//                      reduc_errorL1 += err;
//                  });
//          }).wait_and_throw();
//     }

//     std::cout << "Total cumulated error: "
//               << errorL1 * params.dx * params.dVx << "\n"
//               << std::endl;

// }   // end validate_distr_result

// ==========================================
// ==========================================
[[nodiscard]] double
check_result(sycl::queue &Q, sycl::buffer<double, 2> &buff_fdistrib,
             const ADVParams &params) noexcept {
    /* Fill a buffer the same way we filled fdist at init */
    sycl::buffer<double, 2> buff_init(sycl::range<2>(params.nx, params.nVx));
    fill_buffer(Q, buff_init, params);

    /* Check norm of difference, should be 0 */
    sycl::buffer<double, 2> buff_res(buff_init.get_range());
    Q.submit([&](sycl::handler &cgh) {
         auto A = buff_init.get_access<sycl::access::mode::read>(cgh);
         auto B = buff_fdistrib.get_access<sycl::access::mode::read>(cgh);
         sycl::accessor C(buff_res, cgh, sycl::write_only, sycl::no_init);

         cgh.parallel_for(buff_init.get_range(), [=](auto itm) {
             C[itm] = A[itm] - B[itm];
             C[itm] *= C[itm];   // We square each elements
         });
     }).wait_and_throw();

    double sumResult = 0;
    {
        sycl::buffer<double> buff_sum{&sumResult, 1};

        Q.submit([&](sycl::handler &cgh) {
             // Input values to reductions are standard accessors
             auto inputValues =
                 buff_res.get_access<sycl::access_mode::read>(cgh);

#ifdef __INTEL_LLVM_COMPILER   // for DPCPP
             auto sumReduction = sycl::reduction(buff_sum, cgh, sycl::plus<>());
#else   // for openSYCL
             auto sumAcc = buff_sum.get_access<sycl::access_mode::write>(cgh);
             auto sumReduction = sycl::reduction(sumAcc, sycl::plus<double>());
#endif
             cgh.parallel_for(buff_res.get_range(), sumReduction,
                              [=](auto idx, auto &sum) {
                                  // plus<>() corresponds to += operator, so sum
                                  // can be updated via += or combine()
                                  sum += inputValues[idx];
                              });
         }).wait_and_throw();
    }

    return std::sqrt(sumResult);
}   // end check_result