#pragma once

#include <AdvectionParams.hpp>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <sycl/sycl.hpp>
#include "bench_config.hpp"

// =============================================
// =============================================
struct BenchParams {
    int64_t gpu, n0, n1, n2, w, percent_loc, s0, s2;

    ADVParams adv_params;

    BenchParams() = delete;
    BenchParams(benchmark::State &state)
        : gpu(state.range(0)),
          n0(-1), n1(-1), n2(-1),
          w(state.range(3)),
          s0(state.range(4)), s2(state.range(5)) {
        
            init_params();
          }

    void init_params(){
        auto &p = adv_params;

        p.outputSolution = false;
        p.pref_wg_size = w;

        /* Static physicals params*/
        p.dt = 0.001;
        p.minRealX = 0;
        p.maxRealX = 1;
        p.minRealVx = -1;
        p.maxRealVx = 1;

        /* Dynamic benchmark params*/
        p.gpu = gpu;
        p.n1 = n1;
        p.n0 = n0;
        p.n2 = n2;

        p.percent_loc = static_cast<float>(percent_loc/100.0);
        p.seq_size0 = s0;
        p.seq_size2 = s2;

        p.update_deltas();
    }
};

// =============================================
// =============================================
[[nodiscard]] inline sycl::queue
createSyclQueue(const bool run_on_gpu, benchmark::State &state) {
    sycl::device d;

    if (run_on_gpu)
        try {
            d = sycl::device{sycl::gpu_selector_v};
        } catch (const sycl::exception e) {
            state.SkipWithError(
                "GPU was requested but none is available, skipping benchmark.");
        }
    else
        d = sycl::device{sycl::cpu_selector_v};
    return sycl::queue{d};
}   // end createSyclQueue

