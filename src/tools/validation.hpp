#pragma once
#include "types.hpp"
#include <AdvectionParams.hpp>
#include <sycl/sycl.hpp>

// ==========================================
// ==========================================
real_t
validate_result_adv(sycl::queue &Q, span3d_t &data, const ADVParams &params,
                    bool do_print = true) {
    std::cout << "\nRESULTS_VALIDATION:" << std::endl;

    auto const dx = params.dx;
    auto const dvx = params.dvx;
    auto const dt = params.dt;
    auto const minRealX = params.minRealX;
    auto const minRealVx = params.minRealVx;
    auto const maxIter = params.maxIter;

    sycl::range const r2d(params.n0, params.n1);

    std::vector<real_t> all_l1_errors(params.n2);
    for (size_t i2 = 0; i2 < params.n2; i2++) {

        real_t errorL1 = 0.0;

        std::vector<real_t> errorsL1(params.n2);
        {

            sycl::buffer<real_t> errorl1_buff(&errorL1, 1);

            Q.submit([&](sycl::handler &cgh) {

#ifdef SYCL_IMPLEMENTATION_ONEAPI   // for DPCPP
                 auto errorl1_reduc =
                     sycl::reduction(errorl1_buff, cgh, sycl::plus<>());
#else   // for openSYCL
                 sycl::accessor errorl1_acc(errorl1_buff, cgh,
                                            sycl::read_write);
                 auto errorl1_reduc =
                     sycl::reduction(errorl1_acc, sycl::plus<real_t>());
#endif

                 cgh.parallel_for(
                     r2d, errorl1_reduc, [=](auto itm, auto &errorl1_reduc) {
                         auto i1 = itm[1];
                         auto i0 = itm[0];
                         auto f = data(i0, i1, i2);

                         real_t const x = minRealX + i1 * dx;
                         real_t const v = minRealVx + i0 * dvx;
                         real_t const t = maxIter * dt;

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
    const double epsilon = 1e-10;
    int high_error_count = 0;
    int low_error_count = 0;
    
    for (const auto& error : all_l1_errors) {
        if (std::abs(error - highest_l1) < epsilon) {
            high_error_count++;
        }
        if (std::abs(error - lowest_l1) < epsilon) {
            low_error_count++;
        }
    }

    if (do_print) {
        std::cout << "Highest L1 error found: " << highest_l1 << " (lowest is "
                << lowest_l1 << ")\n"
                << "Number of high errors: " << high_error_count << "\n"
                << "Number of low errors: " << low_error_count << "\n"
                << "Number of zeroes errors: " << std::count(all_l1_errors.begin(), all_l1_errors.end(), 0) << "\n"
                << "Total errors: " << all_l1_errors.size() << "\n"
                << std::endl;
    }


    return highest_l1;
}   // end validate_result

// ==========================================
// ==========================================
real_t
sum_and_normalize_conv1d(sycl::queue &Q, span3d_t data, size_t nw) {
    auto n0 = data.extent(0);
    auto n2 = data.extent(2);
    sycl::range<3> r3d(n0, nw, n2);

    real_t sum = 0;
    {
        sycl::buffer<real_t> buff_sum(&sum, 1);

        Q.submit([&](sycl::handler &cgh) {
             auto reduc_sum = sycl::reduction(buff_sum, cgh, sycl::plus<>());

             cgh.parallel_for(r3d, reduc_sum, [=](auto itm, auto &reduc_sum) {
                 auto i0 = itm[0];
                 auto i1 = itm[1];
                 auto i2 = itm[2];
                 auto f = data(i0, i1, i2);

                 reduc_sum += f;
             });
         }).wait();
    }
    sum /= (n0 * nw * n2);

    return sum;
}   // end sum_and_normalize_conv

// ==========================================
// ==========================================
void
validate_conv1d(sycl::queue &Q, span3d_t &data, size_t nw) {
    const auto n0 = data.extent(0);
    const auto n2 = data.extent(2);

    sycl::range<1> range0(n0);
    sycl::range<1> range2(n2);

    Q.parallel_for(range0, [=](unsigned i0) {
        for (auto i1 = 0; i1 < nw; ++i1) {
            for (auto i2 = 0; i2 < n2 - 1; ++i2) {
                if (data(i0, i1, i2) != data(i0, i1, i2 + 1)) {
                    // throw std::runtime_error("nn");
                    data(0, 0, 0) = -12345;
                };
            }
        }
    });

    Q.parallel_for(range2, [=](unsigned i2) {
        for (auto i1 = 0; i1 < nw; ++i1) {
            for (auto i0 = 0; i0 < n0 - 1; ++i0) {
                if (data(i0, i1, i2) != data(i0 + 1, i1, i2)) {
                    // throw std::runtime_error("nn");
                    data(0, 0, 0) = -45678;
                };
            }
        }
    });

    Q.wait();
    if (data(0, 0, 0) == -12345 || data(0, 0, 0) == -45678)
        std::cout << "WARNING: Values at same position i1 are not equivalent "
                     "throught the batchs. Check implementation."
                  << std::endl;
    else
        std::cout << "All values data[:,i1,:] are equal." << std::endl;
}   // end validate_conv1d

// ==========================================
// ==========================================
void print_perf(const double elapsed_seconds, const size_t n_cells){

    std::cout << "PERF_DIAGS:" << std::endl;
    std::cout << "elapsed_time: " << elapsed_seconds << " s\n";

    auto gcells = (n_cells / elapsed_seconds) / 1e9;
    std::cout << "upd_cells_per_sec: " << gcells << " Gcell/sec\n";
    std::cout << "estimated_throughput: " << gcells * sizeof(real_t) * 2
              << " GB/s" << std::endl;
}
