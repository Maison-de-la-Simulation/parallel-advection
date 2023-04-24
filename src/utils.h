#include "unique_ref.h"
#include <AdvectionParams.h>
#include <advectors.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <sycl/sycl.hpp>

// To switch case on a str
constexpr unsigned int
str2int(const char *str, int h = 0) {
    return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h];
}

static constexpr auto error_str =
    "Should be one of: {Sequential, BasicRange2D, "
    "BasicRange1D, Hierarchical, NDRange, "
    "Scoped}";

// // ==========================================
// // ==========================================
sref::unique_ref<IAdvectorX>
getKernelImpl(std::string k, const ADVParams &params) {
    switch (str2int(k.data())) {
    case str2int("Sequential"):
        return sref::make_unique<AdvX::Sequential>();
        break;
    case str2int("BasicRange2D"):
        return sref::make_unique<AdvX::BasicRange2D>(params.nx, params.nVx);
        break;
    case str2int("BasicRange1D"):
        return sref::make_unique<AdvX::BasicRange1D>(params.nx, params.nVx);
        break;
    case str2int("Hierarchical"):
        return sref::make_unique<AdvX::Hierarchical>();
        break;
    case str2int("NDRange"):
        return sref::make_unique<AdvX::NDRange>();
        break;
    case str2int("Scoped"):
        return sref::make_unique<AdvX::Scoped>();
        break;
    default:
        auto str = k + " is not a valid kernel name.\n" + error_str;
        throw std::runtime_error(str);
        break;
    }
}

// ==========================================
// ==========================================
void
fill_buffer(sycl::queue &q, sycl::buffer<double, 2> &buff_fdist,
            const ADVParams &params) {

    sycl::host_accessor fdist(buff_fdist, sycl::write_only, sycl::no_init);

    for (int ix = 0; ix < params.nx; ++ix) {
        for (int iv = 0; iv < params.nVx; ++iv) {
            double x = params.minRealx + ix * params.dx;
            fdist[iv][ix] = sycl::sin(4 * x * M_PI);
        }
    }
}

// ==========================================
// ==========================================
void
print_buffer(sycl::buffer<double, 2> &fdist, const ADVParams &params) {
    sycl::host_accessor tab(fdist, sycl::read_only);

    for (int ix = 0; ix < params.nx; ++ix) {
        for (int iv = 0; iv < params.nVx; ++iv) {
            std::cout << tab[ix][iv] << " ";
        }
        std::cout << std::endl;
    }
}   // end print_buffer

// ==========================================
// ==========================================
void
export_result_to_file(sycl::buffer<double, 2> &buff_fdistrib,
                      const ADVParams &params) {

    sycl::host_accessor fdist(buff_fdistrib, sycl::read_only);

    auto str = "solution.log";
    std::ofstream outfile(str);

    for (int iv = 0; iv < params.nVx; ++iv) {
        for (int ix = 0; ix < params.nx; ++ix) {
            outfile << fdist[iv][ix];

            if (ix != params.nx - 1)
                outfile << ",";
        }
        outfile << std::endl;
    }
    outfile.close();
}

// ==========================================
// ==========================================
void
validate_result(sycl::queue &Q, sycl::buffer<double, 2> &buff_fdistrib,
                const ADVParams &params) {

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
double
check_result(sycl::queue &Q, sycl::buffer<double, 2> &buff_fdistrib,
             const ADVParams &params) {
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
