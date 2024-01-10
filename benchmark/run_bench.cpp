#include <benchmark/benchmark.h>

#include <AdvectionParams.h>
#include <sycl/sycl.hpp>
// #include <iostream>
#include <init.h>
#include <io.h>
// #include <validation.h>
#include <advectors.h>

enum AdvImpl : int {
    BR2D,   // 0
    BR1D,   // 1
    HIER,   // 2
    NDRA,   // 3
    SCOP    // 4
};

// =============================================
// =============================================
[[nodiscard]] inline sycl::queue
createSyclQueue(const bool run_on_gpu) {
    sycl::device d;

    if (run_on_gpu)
        d = sycl::device{sycl::gpu_selector_v};
    else
        d = sycl::device{sycl::cpu_selector_v};

    return sycl::queue{d};
}   // end createSyclQueue

// =============================================
// =============================================
[[nodiscard]] sref::unique_ref<IAdvectorX>
advectorFactory(const AdvImpl kernel_id, const size_t nx, const size_t nvx,
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

    // std::cout << "KernelID: " << kernel_id << " - " << params.kernelImpl <<
    // std::endl;

    return kernel_impl_factory(params);
}   // end advectorFactory

// ==========================================
// ==========================================
static void
BM_Advector(benchmark::State &state) {
    /* Params setup */
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
    p.gpu = state.range(0);
    p.nx = state.range(1);
    p.nvx = state.range(2);
    p.dx = p.realWidthX / p.nx;
    p.dvx = (p.maxRealVx - p.minRealVx) / p.nvx;
    p.inv_dx = 1 / p.dx;

    /* SYCL setup */
    auto Q = createSyclQueue(p.gpu);
    sycl::buffer<double, 2> fdist(sycl::range<2>(p.nvx, p.nx));

    /* Physics setup */
    fill_buffer(Q, fdist, p);

    /* Advector setup */
    auto kernel_id = static_cast<AdvImpl>(static_cast<int>(state.range(3)));
    auto advector = advectorFactory(kernel_id, p.nx, p.nvx, state);

    /* Benchmark */
    state.counters.insert({
        {"gpu", p.gpu},
        {"nx", p.nx},
        {"ny", p.nvx},
        {"kernel_id", kernel_id},
    });

    for (auto _ : state)
        advector(Q, fdist, p).wait();

    state.SetBytesProcessed(int64_t(state.iterations()) *
                            int64_t(p.nvx * p.nx * sizeof(double)));
}   // end BM_Advector

// TODO : add WG SIZE as bench parameter
BENCHMARK(BM_Advector)
    /* ->Args({gpu, nx, nvx, kernel_id}) */
    ->Args({0, 128, 64, AdvImpl::BR2D})
    ->Args({0, 128, 64, AdvImpl::BR1D})
    ->Args({0, 128, 64, AdvImpl::HIER})
    ->Args({0, 128, 64, AdvImpl::NDRA})
    ->Args({0, 128, 64, AdvImpl::SCOP})->Unit(benchmark::kMillisecond);
    // ->ReportAggregatesOnly(true)
    // ->DisplayAggregatesOnly(true);

BENCHMARK_MAIN();
