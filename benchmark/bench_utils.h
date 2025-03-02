#pragma once

#include <AdvectionParams.h>
#include <advectors.h>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <init.h>
#include <sycl/sycl.hpp>

enum AdvImpl : int {
    BR3D,           // 0
    HIER,           // 1
    NDRA,           // 2
    ADAPTWG,        // 3
    HYBRID          // 4
};

using bm_vec_t = std::vector<int64_t>;

static bm_vec_t N0_RANGE = {1024};
static bm_vec_t N1_RANGE = {16, 1024, 6100};
static bm_vec_t N2_RANGE = {16};
static bm_vec_t PERCENT_LOC = {10, 20, 30, 40, 50, 60, 70, 80, 90};
static bm_vec_t SEQ_SIZE0 = {1};
static bm_vec_t SEQ_SIZE2 = {1};

static std::vector<bm_vec_t> EXP_RANGE{
    {1<<17, 1<<14, 1},     //n1 trop grand pour local mem, acces coal
    {1<<10, 1<<11, 1<<10}, //n1 trop grand (WI per WG) mais rentre en local mem
    {1<<10, 1<<14, 1<<7},  //n1 trop grand pour fit en local +acces non coal.
    {1<<27, 1<<4,  1},     //n1 trop petit, acces coalescent en dim 1
    {1<<20, 1<<4,  1<<7},  //n1 trop petit et acces non coalescent
    {1<<21, 1<<10, 1},     //cas parfait: elements contigus
    {1<<14, 1<<10, 1<<7},  //elements espacés en memoire de plus de SIMD_Size
    // {0,1<<10,1<<6+1},   //non aligné et pas power of two
    {1    , 1<<10, 1<<21}, //un seul batch en d0
    {1<<11, 1<<10, 1<<10}, //profil equilibre
};

static int64_t WG_SIZE_NVI = 128;
static int64_t WG_SIZE_AMD = 256;
static int64_t WG_SIZE_PVC = 1024;

static bm_vec_t WG_SIZES_RANGE = {1, 4, 8, 64, 128, 256, 512, 1024};

static bm_vec_t IMPL_RANGE = {AdvImpl::BR3D, AdvImpl::HIER, AdvImpl::NDRA,
                              AdvImpl::ADAPTWG, AdvImpl::HYBRID};


// =============================================
// =============================================
struct BenchParams {
    int64_t gpu, n0, n1, n2, w, percent_loc, s0, s2;

    ADVParams adv_params;

    BenchParams() = delete;
    BenchParams(benchmark::State &state)
        : gpu(state.range(0)), n0(state.range(1)), n1(state.range(2)),
          n2(state.range(3)), w(state.range(4)), percent_loc(state.range(5)),
          s0(state.range(6)), s2(state.range(7)) {
        
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

// =============================================
// =============================================
[[nodiscard]] inline sref::unique_ref<IAdvectorX>
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
    case AdvImpl::ADAPTWG:
        params.kernelImpl = "AdaptiveWg";
        break;
    case AdvImpl::HYBRID:
        params.kernelImpl = "HybridMem";
        break;
    default:
        auto str = "Error: wrong kernel_id.\n";
        throw std::runtime_error(str);
        break;
    }

    return kernel_impl_factory(q, params, s);
}   // end advectorFactory
