#pragma once

#include <AdvectionParams.h>
#include <advectors.h>
#include <benchmark/benchmark.h>
#include <init.h>
#include <sycl/sycl.hpp>

enum AdvImpl : int {
    BR3D,           // 0
    HIER,           // 1
    NDRA,           // 2
    SCOP,           // 3
    STREAMY,        // 4
    STRAD,          // 5
    REVIDX,         // 6
    TWODIMWG,       // 7
    SEQ_TWODIMWG,   // 8
    EXP1,           // 9
    EXP2,           // 10
    EXP3,           // 11
    EXP4,           // 12
    EXP5,           // 13
    EXP6,           // 14
    // EXP7,     // 15
    FULLYGLOBAL,   // 16
    FULLYLOCAL,    // 17
};

using bm_vec_t = std::vector<int64_t>;
static bm_vec_t NB_LARGE_RANGE = benchmark::CreateRange(2 << 5, 2 << 20, 2);
static bm_vec_t NB_RANGE = {16384, 32768, 65535};
static bm_vec_t NS_RANGE = {1, 2, 4, 8, 16, 32, 64};
static int64_t n1 = 1024;

static bm_vec_t WG_SIZES_X_RANGE = {1, 4, 8, 64, 128, 256, 512, 1024};

static bm_vec_t IMPL_RANGE = {
    // AdvImpl::BR3D,
    AdvImpl::HIER,     AdvImpl::NDRA,         AdvImpl::SCOP,
    AdvImpl::STREAMY,  AdvImpl::STRAD,        AdvImpl::REVIDX,
    AdvImpl::TWODIMWG, AdvImpl::SEQ_TWODIMWG, AdvImpl::EXP1};

// static bm_vec_t IMPL_RANGE_NO_SCOPED = {
//                               AdvImpl::BR3D, AdvImpl::HIER, AdvImpl::NDRA,
//                               AdvImpl::STREAMY, AdvImpl::STRAD,
//                               AdvImpl::REVIDX, AdvImpl::TWODIMWG,
//                               AdvImpl::SEQ_TWODIMWG};

// =============================================
// =============================================
[[nodiscard]] inline ADVParams
createParams(const bool gpu, const size_t &n0, const size_t &n1,
             const size_t &n2) {
    ADVParams p;

    p.outputSolution = false;
    p.loc_wg_size_0 = 1;
    p.loc_wg_size_1 = 128;
    p.loc_wg_size_2 = 1;

    p.glob_wg_size_0 = 1;
    p.glob_wg_size_1 = 128;
    p.glob_wg_size_2 = 1;

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
            state.SkipWithError(
                "GPU was requested but none is available, skipping benchmark.");
        }
    else
        d = sycl::device{sycl::cpu_selector_v};
    return sycl::queue{d};
}   // end createSyclQueue

// =============================================
// =============================================
[[nodiscard]] sref::unique_ref<IAdvectorX>
advectorFactory(const sycl::queue &q, ADVParams &p, Solver &s,
                const AdvImpl kernel_id, benchmark::State &state) {
    ADVParamsNonCopyable params(p);

    switch (kernel_id) {
    case AdvImpl::BR3D:
        params.kernelImpl = "BasicRange";
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
    case AdvImpl::STREAMY:
        params.kernelImpl = "StreamY";
        break;
    case AdvImpl::STRAD:
        params.kernelImpl = "StraddledMalloc";
        break;
    case AdvImpl::REVIDX:
        params.kernelImpl = "ReverseIndexes";
        break;
    case AdvImpl::TWODIMWG:
        params.kernelImpl = "TwoDimWG";
        break;
    case AdvImpl::SEQ_TWODIMWG:
        params.kernelImpl = "SeqTwoDimWG";
        break;
    case AdvImpl::EXP1:
        params.kernelImpl = "Exp1";
        break;
    case AdvImpl::EXP2:
        params.kernelImpl = "Exp2";
        break;
    case AdvImpl::EXP3:
        params.kernelImpl = "Exp3";
        break;
    case AdvImpl::EXP4:
        params.kernelImpl = "Exp4";
        break;
    case AdvImpl::EXP5:
        params.kernelImpl = "Exp5";
        break;
    case AdvImpl::EXP6:
        params.kernelImpl = "Exp6";
        break;
    // case AdvImpl::EXP7:
    //     params.kernelImpl = "Exp7";
    //     break;
    case AdvImpl::FULLYGLOBAL:
        params.kernelImpl = "FullyGlobal";
        break;
    case AdvImpl::FULLYLOCAL:
        params.kernelImpl = "FullyLocal";
        break;
    default:
        auto str = "Error: wrong kernel_id.\n";
        throw std::runtime_error(str);
        break;
    }

    return kernel_impl_factory(q, params, s);
}   // end advectorFactory
