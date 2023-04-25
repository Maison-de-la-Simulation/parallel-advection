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
    double totalError = 0.0;
    {
        sycl::buffer<int> buff_errCount{&errCount, 1};
        sycl::buffer<double> buff_sumTotalErr{&totalError, 1};

        Q.submit([&](sycl::handler &cgh) {
             // Input values to reductions are standard accessors
             auto errAcc =
                 buff_errCount.get_access<sycl::access_mode::write>(cgh);
             auto fdist =
                 buff_fdistrib.get_access<sycl::access_mode::read>(cgh);
             auto totErrAcc =
                 buff_sumTotalErr.get_access<sycl::access_mode::write>(cgh);

             auto errReduction = sycl::reduction(errAcc, sycl::plus<int>());
             auto totalSumError =
                 sycl::reduction(totErrAcc, sycl::plus<double>());

             cgh.parallel_for(
                 buff_fdistrib.get_range(), errReduction, totalSumError,
                 [=](auto itm, auto &totalErr, auto &totalSumError) {
                     auto ix = itm[1];
                     auto ivx = itm[0];
                     auto f = fdist[itm];

                     double const x = params.minRealx + ix * params.dx;
                     double const v = params.minRealVx + ivx * params.dVx;
                     double const t = params.maxIter * params.dt;

                     auto value = sycl::sin(4 * M_PI * (x - v * t));

                     auto err = sycl::fabs(f - value);
                     totalSumError += err;

                     if (err > acceptable_error)
                         totalErr += 1;
                 });
         }).wait_and_throw();
    }

    double const totalCells = params.nx * params.nVx;
    double const fractionErr = errCount / totalCells * 100;

    std::cout << "fraction of cells with error > " << acceptable_error << ": "
              << fractionErr << "% of total cells" << std::endl;

    std::cout << "Total cumulated error: "
              << totalError * params.dx * params.dVx << "\n"
              << std::endl;

}   // end validate_results

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