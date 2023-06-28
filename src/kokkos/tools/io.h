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
    std::cout << "Exporting result to file " << str << "..." << std::endl;

    KV_double_3d::HostMirror hostView = Kokkos::create_mirror_view(fdist);

    std::ofstream outfile(str);

    for (int iv = 0; iv < params.nvx; ++iv) {
        for (int ix = 0; ix < params.nx; ++ix) {
            // we only take first slice in dim fict
            // carefull of LayoutType of Kokkos, this is for LayoutRight
            outfile << *(hostView.data() + iv * params.nx + ix);

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
export_error_to_file(KV_double_3d &fdist, const ADVParams &params) noexcept {

    auto str = "error.log";
    std::cout << "Exporting error to file " << str << "..." << std::endl;

    KV_double_3d::HostMirror hostView = Kokkos::create_mirror_view(fdist);

    std::ofstream outfile(str);

    for (int iv = 0; iv < params.nvx; ++iv) {
        for (int ix = 0; ix < params.nx; ++ix) {

            double const x = params.minRealx + ix * params.dx;
            double const v = params.minRealVx + iv * params.dvx;
            double const t = params.maxIter * params.dt;
            auto value = Kokkos::sin(4 * Kokkos::numbers::pi * (x - v * t));

            auto f = *(hostView.data() + iv * params.nx + ix);
            outfile << Kokkos::fabs(f - value);

            if (ix != params.nx - 1)
                outfile << ",";
        }
        outfile << std::endl;
    }
    outfile.close();
}