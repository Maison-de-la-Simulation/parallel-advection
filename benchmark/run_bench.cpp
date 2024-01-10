#include <benchmark/benchmark.h>
#include "bench_utils.h"
#include <validation.h>

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

BENCHMARK(BM_Advector)
    ->ArgsProduct({
        {0}, /*gpu*/
        {1024}, /*nx*/
        benchmark::CreateRange(256, 16384, 2), /*ny*/
        // benchmark::CreateRange(2<<5, 2<<20, /*multi=*/2), /*nx*/
        {AdvImpl::BR2D, AdvImpl::BR1D, AdvImpl::HIER, AdvImpl::NDRA, AdvImpl::SCOP} /*impl*/
      });
                                    //    ->Unit(benchmark::kMillisecond);

// TODO : add a benchmark with WG_SIZE as bench parameter for hierarchical and scoped ONLY and fewer sizes

BENCHMARK_MAIN();
