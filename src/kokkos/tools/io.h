#pragma once

#include "../kokkos_shortcut.hpp"
#include <AdvectionParams.h>
#include <fstream>
#include <iostream>

// ==========================================
// ==========================================
void
export_result_to_file(KV_double_3d &fdist, const ADVParams &params) noexcept {

    auto str = "solution.log";
    std::cout << "Exporting result to file " << str << "...\n" << std::endl;

    // sycl::host_accessor fdist(buff_fdistrib, sycl::read_only);

    // std::ofstream outfile(str);

    // for (int iv = 0; iv < params.nvx; ++iv) {
    //     for (int ix = 0; ix < params.nx; ++ix) {
    //         outfile << fdist[0][iv][ix];

    //         if (ix != params.nx - 1)
    //             outfile << ",";
    //     }
    //     outfile << std::endl;
    // }
    // outfile.close();
}

// ==========================================
// ==========================================
void
export_error_to_file(KV_double_3d &fdist, const ADVParams &params) noexcept {

    auto str = "error.log";
    std::cout << "Exporting error to file " << str << "...\n" << std::endl;

    // sycl::host_accessor fdist(buff_fdistrib, sycl::read_only);

    // std::ofstream outfile(str);

    // for (int iv = 0; iv < params.nvx; ++iv) {
    //     for (int ix = 0; ix < params.nx; ++ix) {

    //         double const x = params.minRealx + ix * params.dx;
    //         double const v = params.minRealVx + iv * params.dvx;
    //         double const t = params.maxIter * params.dt;
    //         auto value = sycl::sin(4 * M_PI * (x - v * t));

    //         outfile << sycl::fabs(fdist[0][iv][ix] - value);

    //         if (ix != params.nx - 1)
    //             outfile << ",";
    //     }
    //     outfile << std::endl;
    // }
    // outfile.close();
}