#pragma once

#include "../kokkos_shortcut.hpp"
#include "init.h"
#include <AdvectionParams.h>
#include <iostream>

// ==========================================
// ==========================================
void
validate_result(KV_double_3d &fdist, const ADVParams &params,
                const InitParams &initParams) noexcept {

    const Kokkos::Array<int, 3> begin{0, 0, 0};
    const Kokkos::Array<int, 3> end{fdist.extent_int(0), fdist.extent_int(1),
                                    fdist.extent_int(2)};
    double sumL1 = 0;
    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_LAMBDA(const int i, const int j, const int k, double &lsum) {
            auto ix = k;
            auto ivx = j;
            auto f = fdist(i, j, k);

            double const x = params.minRealx + ix * params.dx;
            double const v = params.minRealVx + ivx * params.dvx;
            double const t = params.maxIter * params.dt;

            auto value = Kokkos::sin(4 * Kokkos::numbers::pi * (x - v * t));

            auto err = Kokkos::fabs(f - value);

            lsum += err;
        },
        sumL1);

    std::cout << "Total cumulated error: "
              << (sumL1 * params.dx * params.dvx) / params.n_fict_dim << "\n"
              << std::endl;

    // double seqSum = 0;
    // for(int i_fict = 0; i_fict < params.n_fict_dim; ++i_fict)
    //     for(int ivx=0; ivx<params.nvx; ++ivx)
    //         for(int ix=0; ix<params.nx; ++ix)
    //         {
    //             auto f = fdist(i_fict, ivx, ix);
    //             double const x = params.minRealx + ix * params.dx;
    //             double const v = params.minRealVx + ivx * params.dvx;
    //             double const t = params.maxIter * params.dt;
    //             auto value = sin(4 * Kokkos::numbers::pi * (x - v * t));

    //             auto err = fabs(f - value);
    //             seqSum += err;
    //         }
    // std::cout << "Total Sequential error L1: "
    //           << (seqSum * params.dx * params.dvx) / params.n_fict_dim << "\n"
    //           << std::endl;
}   // end validate_results