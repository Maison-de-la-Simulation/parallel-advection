#include <benchmark/benchmark.h>

#include <AdvectionParams.h>
#include <sycl/sycl.hpp>
#include <init.h>
#include <io.h>
#include <validation.h>
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
createSyclQueue(const bool run_on_gpu, benchmark::State &state) {
    sycl::device d;

    if (run_on_gpu)
        try {
            d = sycl::device{sycl::gpu_selector_v};
        } catch (const sycl::runtime_error e) {
            state.SkipWithError("GPU was requested but none is available, skipping benchmark.");
        }
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

    /* Advector setup */
    auto kernel_id = static_cast<AdvImpl>(static_cast<int>(state.range(3)));
    auto advector = advectorFactory(kernel_id, p.nx, p.nvx, state);

    /* Benchmark infos */
    state.counters.insert({
        {"gpu", p.gpu},
        {"nx", p.nx},
        {"ny", p.nvx},
        // {"iterations", state.iterations()},
        {"kernel_id", kernel_id},
    });

    /* SYCL setup */
    auto Q = createSyclQueue(p.gpu, state);
    sycl::buffer<double, 2> fdist(sycl::range<2>(p.nvx, p.nx));

    /* Physics setup */
    fill_buffer(Q, fdist, p);

    /* Benchmark */
    for (auto _ : state){
        try
        {
          advector(Q, fdist, p).wait();
        }
        catch(const std::exception& e)
        {
          state.SkipWithError(e.what());
          break; // REQUIRED to prevent all further iterations.
        }
        catch(const sycl::exception& e)
        {
          state.SkipWithError(e.what());
          break;
        }
        //TODO: add error handling for ONEAPI
    }

    p.maxIter = state.iterations();

    //TODO: fix these weird values
    state.SetItemsProcessed(int64_t(p.maxIter) * int64_t(p.nvx * p.nx));
    state.SetBytesProcessed(int64_t(p.maxIter) * int64_t(p.nvx * p.nx * sizeof(double)));

    auto err = validate_result(Q, fdist, p, false);
    if(err > 10e-6){
        state.SkipWithError("Validation failled with numerical error > 10e-6.");
    }

}   // end BM_Advector

// TODO : add WG SIZE as bench parameter
BENCHMARK(BM_Advector)
    /* ->Args({gpu, nx, nvx, kernel_id}) */
    ->ArgsProduct({
        {0}, /*gpu*/
        // benchmark::CreateRange(2<<5, 2<<20, /*multi=*/2), /*nx*/
        {1024}, /*nx*/
        benchmark::CreateRange(256, 16384, /*multi=*/2), /*ny*/
        {AdvImpl::BR2D, AdvImpl::BR1D, AdvImpl::HIER, AdvImpl::NDRA, AdvImpl::SCOP} /*impl*/
      });
                                    //    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
