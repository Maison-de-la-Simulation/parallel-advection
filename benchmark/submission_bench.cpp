#include "bench_utils.h"
#include <validation.h>

// ==========================================
// ==========================================
/* Benchmarks the impact of the queue submission on the runtime
(we .wait() only for the last iteration) */
static void
BM_Advector(benchmark::State &state) {
    /* Params setup */
    auto p = createParams(state.range(0), state.range(1), state.range(2));

    /* Advector setup */
    auto kernel_id = static_cast<AdvImpl>(static_cast<int>(state.range(3)));
    auto advector = advectorFactory(kernel_id, p.nx, p.nvx, state);

    p.maxIter = state.range(4);
    /* Benchmark infos */
    state.counters.insert({
        {"gpu", p.gpu},
        {"nx", p.nx},
        {"ny", p.nvx},
        {"kernel_id", kernel_id},
        {"nIter", p.maxIter},
    });

    /* SYCL setup */
    auto Q = createSyclQueue(p.gpu, state);
    sycl::buffer<double, 2> fdist(sycl::range<2>(p.nvx, p.nx));

    /* Physics setup */
    fill_buffer(Q, fdist, p);
    Q.wait();

    /* Benchmark */
    for (auto _ : state) {
        for (size_t i = 0; i < p.maxIter-1; i++)
            advector(Q, fdist, p);

        advector(Q, fdist, p).wait();
    }

    state.counters.insert({
        {"nRepet", state.iterations()},
    });

    state.SetItemsProcessed(state.iterations() * p.maxIter * p.nvx * p.nx);
    state.SetBytesProcessed(state.iterations() * p.maxIter * p.nvx * p.nx *
                            sizeof(double));

    p.maxIter *= state.iterations();
    auto err = validate_result(Q, fdist, p, false);
    if (err > 10e-6) {
        state.SkipWithError("Validation failled with numerical error > 10e-6.");
    }
}   // end BM_Advector

BENCHMARK(BM_Advector)
    ->ArgsProduct({
        {1},                                /*gpu*/
        {1024},                                /*nx*/
        // benchmark::CreateRange(256, 16384, 2), /*ny*/
        {16384, 32768, 65536}, /*ny*/
        {AdvImpl::BR2D, AdvImpl::BR1D, AdvImpl::HIER, AdvImpl::NDRA,
         AdvImpl::SCOP}, /*kernel_id*/
        {1, 2, 10, 50, 100, 1000, 10000} /*p.maxIter*/
    }) 
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
