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

using bm_vec_t = std::vector<int64_t>;
static bm_vec_t NB_RANGE = benchmark::CreateRange(2 << 5, 2 << 20, 2);
static bm_vec_t NB_SMALL_RANGE = {16384, 32768, 65536};
static bm_vec_t NS_RANGE = {1, 2, 4, 8, 16, 32, 64, 128, 256};
static int64_t  NX = 1024;

static bm_vec_t WG_SIZES_X_RANGE = {1, 4, 8, 64, 128, 256, 512, 1024};

static bm_vec_t IMPL_RANGE = {AdvImpl::BR2D, AdvImpl::BR1D, AdvImpl::HIER,
                              AdvImpl::NDRA, AdvImpl::SCOP};
static bm_vec_t IMPL_NO_SCOPED_RANGE = {AdvImpl::BR2D, AdvImpl::BR1D,
                                        AdvImpl::HIER, AdvImpl::NDRA};

// =============================================
// =============================================
[[nodiscard]] inline ADVParams
createParams(const bool gpu,
             const size_t &nb,
             const size_t &nx,
             const size_t &ns) {
    ADVParams p;

    p.outputSolution = false;
    p.wg_size_x = 128;
    p.wg_size_b = 1;

    /* Static physicals params*/
    p.dt = 0.001;
    p.minRealX = 0;
    p.maxRealX = 1;
    p.minRealVx = -1;
    p.maxRealVx = 1;

    /* Dynamic benchmark params*/
    p.gpu = gpu;
    p.nx = nx;
    p.nb = nb;
    p.ns = ns;

    p.update_deltas();
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
advectorFactory(const AdvImpl kernel_id,
                ADVParams p,
                benchmark::State &state) {
    ADVParamsNonCopyable params(p);
    // params.nx = nx;
    // params.nb = nb;
    // params.ns = ns;

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
