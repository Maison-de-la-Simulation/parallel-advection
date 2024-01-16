#include "bench_utils.h"
#include <validation.h>

// ==========================================
// ==========================================
/* Benchmark the impact of wg_size on Hierarchical kernel */
static void
BM_WgSize(benchmark::State &state){
    auto p = createParams(state.range(0), state.range(1), state.range(2));

    /* Advector setup */
    const auto kernel_id = AdvImpl::HIER;
    const auto hierAdv = advectorFactory(kernel_id, p.nx, p.nvx, state);
    p.wg_size = state.range(3);

    /* Benchmark infos */
    state.counters.insert({
        {"gpu", p.gpu},
        {"nx", p.nx},
        {"ny", p.nvx},
        {"kernel_id", kernel_id},
        {"wg_size", p.wg_size},
    });

    /* SYCL setup */
    auto Q = createSyclQueue(p.gpu, state);
    sycl::buffer<double, 2> fdist(sycl::range<2>(p.nvx, p.nx));

    /* Physics setup */
    fill_buffer(Q, fdist, p);

    /* Benchmark */
    for (auto _ : state) {
        try {
            hierAdv(Q, fdist, p).wait();
        } catch (const std::exception &e) {
            state.SkipWithError(e.what());
            break; // REQUIRED to prevent all further iterations.
        } catch (const sycl::exception &e) {
            state.SkipWithError(e.what());
            break;
        }
    }

    p.maxIter = state.iterations();

    state.SetItemsProcessed(p.maxIter * p.nvx * p.nx);
    state.SetBytesProcessed(p.maxIter * p.nvx * p.nx * sizeof(double));

    auto err = validate_result(Q, fdist, p, false);
    if (err > 10e-6) {
        state.SkipWithError("Validation failled with numerical error > 10e-6.");
    }
}

BENCHMARK(BM_WgSize)
    ->ArgsProduct({
        {0},//, 1},                                /*gpu*/
        {1024},                                /*nx*/
        benchmark::CreateRange(512, 16384, 2), /*ny*/
        {64, 128, 256, 512} /*wg_size*/
    })
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ==========================================
// ==========================================
/* Main benchmark, use to determine the duration and bytes processed of
one advection operation */
static void
BM_Advector(benchmark::State &state) {
    /* Params setup */
    auto p = createParams(state.range(0), state.range(1), state.range(2));

    /* Advector setup */
    auto kernel_id = static_cast<AdvImpl>(static_cast<int>(state.range(3)));
    auto advector = advectorFactory(kernel_id, p.nx, p.nvx, state);

    /* Benchmark infos */
    state.counters.insert({
        {"gpu", p.gpu},
        {"nx", p.nx},
        {"ny", p.nvx},
        {"kernel_id", kernel_id},
    });

    /* SYCL setup */
    auto Q = createSyclQueue(p.gpu, state);
    sycl::buffer<double, 2> fdist(sycl::range<2>(p.nvx, p.nx));

    /* Physics setup */
    fill_buffer(Q, fdist, p);

    /* Benchmark */
    for (auto _ : state) {
        try {
            advector(Q, fdist, p).wait();
        } catch (const std::exception &e) {
            state.SkipWithError(e.what());
            break; // REQUIRED to prevent all further iterations.
        } catch (const sycl::exception &e) {
            state.SkipWithError(e.what());
            break;
        }
    }

    p.maxIter = state.iterations();

    state.SetItemsProcessed(p.maxIter * p.nvx * p.nx);
    state.SetBytesProcessed(p.maxIter * p.nvx * p.nx * sizeof(double));

    auto err = validate_result(Q, fdist, p, false);
    if (err > 10e-6) {
        state.SkipWithError("Validation failled with numerical error > 10e-6.");
    }
}   // end BM_Advector

BENCHMARK(BM_Advector)
    ->ArgsProduct({
        {0, 1},                                /*gpu*/
        {1024},                                /*nx*/
        benchmark::CreateRange(256, 16384, 2), /*ny*/
        // benchmark::CreateRange(2<<5, 2<<20, /*multi=*/2), /*nx*/
        {AdvImpl::BR2D, AdvImpl::BR1D, AdvImpl::HIER, AdvImpl::NDRA,
         AdvImpl::SCOP} /*impl*/
    })
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ==========================================
// ==========================================
BENCHMARK_MAIN();
