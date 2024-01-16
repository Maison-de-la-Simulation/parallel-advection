#pragma once

#include <AdvectionParams.h>
#include <sycl/sycl.hpp>
#include <init.h>
#include <advectors.h>
#include <benchmark/benchmark.h>

enum AdvImpl : int {
    BR2D,   // 0
    BR1D,   // 1
    HIER,   // 2
    NDRA,   // 3
    SCOP    // 4
};

// =============================================
// =============================================
[[nodiscard]] inline ADVParams
createParams(const bool gpu,
             const size_t &nx,
             const size_t &nvx) {
    ADVParams p;

    p.outputSolution = false;
    p.wg_size = 128;

    /* Static physicals p*/
    p.dt = 0.001;
    p.minRealX = 0;
    p.maxRealX = 1;
    p.minRealVx = -1;
    p.maxRealVx = 1;
    p.realWidthX = p.maxRealX - p.minRealX;

    /* Dynamic benchmark p*/
    p.gpu = gpu;
    p.nx = nx;
    p.nvx = nvx;
    p.dx = p.realWidthX / p.nx;
    p.dvx = (p.maxRealVx - p.minRealVx) / p.nvx;
    p.inv_dx = 1 / p.dx;

    return p;
}

// =============================================
// =============================================
[[nodiscard]] inline sycl::queue
createSyclQueue(const bool run_on_gpu, benchmark::State &state) {
    sycl::device d;

    if (run_on_gpu)
        try {
            d = sycl::device{sycl::gpu_selector_v};
        } catch (const sycl::exception e) {
            state.SkipWithError("GPU was requested but none is available, skipping benchmark.");
        }
    else
        d = sycl::device{sycl::cpu_selector_v};
    return sycl::queue{d};
}   // end createSyclQueue

// =============================================
// =============================================
[[nodiscard]] sref::unique_ref<IAdvectorX>
advectorFactory(const AdvImpl kernel_id, const size_t &nx, const size_t &nvx,
                benchmark::State &state) {
    ADVParamsNonCopyable params;
    params.nx = nx;
    params.nvx = nvx;

    switch (kernel_id) {
    case AdvImpl::BR2D:
        params.kernelImpl = "BasicRange2D";
        break;
    case AdvImpl::BR1D:
        params.kernelImpl = "BasicRange1D";
        break;
    case AdvImpl::HIER:
        params.kernelImpl = "Hierarchical";
        break;
    case AdvImpl::NDRA:
        params.kernelImpl = "NDRange";
        break;
    case AdvImpl::SCOP:
        params.kernelImpl = "Scoped";
        break;

    default:
        auto str = "Error: wrong kernel_id.\n";
        throw std::runtime_error(str);
        break;
    }

    return kernel_impl_factory(params);
}   // end advectorFactory
