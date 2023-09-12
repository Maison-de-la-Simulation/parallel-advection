#pragma once

#include <AdvectionParams.h>
#include <fstream>
#include <iostream>
#include <sycl/sycl.hpp>

// ==========================================
// ==========================================
void
print_buffer(sycl::buffer<double, 2> &fdist, const ADVParams &params) noexcept {
    sycl::host_accessor tab(fdist, sycl::read_only);

    for (int ix = 0; ix < params.nx; ++ix) {
        for (int iv = 0; iv < params.nvx; ++iv) {
            std::cout << tab[ix][iv] << " ";
        }
        std::cout << std::endl;
    }
}   // end print_buffer

// ==========================================
// ==========================================
void
export_result_to_file(sycl::buffer<double, 2> &buff_fdistrib,
                      const ADVParams &params) noexcept {

    auto str = "solution.log";
    std::cout << "Exporting result to file " << str << "...\n" << std::endl;

    sycl::host_accessor fdist(buff_fdistrib, sycl::read_only);

    std::ofstream outfile(str);

    for (int iv = 0; iv < params.nvx; ++iv) {
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
export_error_to_file(sycl::buffer<double, 2> &buff_fdistrib,
                      const ADVParams &params) noexcept {

    auto str = "error.log";
    std::cout << "Exporting error to file " << str << "...\n" << std::endl;

    sycl::host_accessor fdist(buff_fdistrib, sycl::read_only);

    std::ofstream outfile(str);

    for (int iv = 0; iv < params.nvx; ++iv) {
        for (int ix = 0; ix < params.nx; ++ix) {

            double const x = params.minRealX + ix * params.dx;
            double const v = params.minRealVx + iv * params.dvx;
            double const t = params.maxIter * params.dt;
            auto value = sycl::sin(4 * M_PI * (x - v * t));

            outfile << sycl::fabs(fdist[iv][ix] - value);

            if (ix != params.nx - 1)
                outfile << ",";
        }
        outfile << std::endl;
    }
    outfile.close();
}