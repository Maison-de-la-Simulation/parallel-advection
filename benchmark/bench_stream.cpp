#include "bench_utils.h"
#include <validation.h>

// ==========================================
// ==========================================
/* Benchmark a FakeAdvector to see the STREAM performance */
static void
BM_STREAM(benchmark::State &state){
    auto p = createParams(state.range(0), state.range(1), state.range(2));
    p.wg_size = 128;

    /* Advector setup */
    const auto advector = sref::make_unique<AdvX::FakeAdvector>();

    /* Benchmark infos */
    state.counters.insert({
        {"gpu", p.gpu},
        {"nx", p.nx},
        {"ny", p.nvx},
        {"kernel_id", -1},
        {"wg_size", p.wg_size},
    });

    /* SYCL setup */
    auto Q = createSyclQueue(p.gpu, state);
    sycl::buffer<double, 2> fdist(sycl::range<2>(p.nvx, p.nx));

    /* Physics setup */
    fill_buffer(Q, fdist, p);

    /* Benchmark */
    for (auto _ : state) {
            advector(Q, fdist, p).wait();
    }

    p.maxIter = state.iterations();

    state.SetItemsProcessed(p.maxIter * p.nvx * p.nx);
    state.SetBytesProcessed(p.maxIter * p.nvx * p.nx * sizeof(double));

    //No validation because it's FakeAdvector
}

BENCHMARK(BM_STREAM)
    ->ArgsProduct({
        {1},     /*gpu*/
        {1024},  /*nx*/
        {16384, 32768, 65536}, /*ny*/
    })
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_STREAM)
    ->ArgsProduct({
        {1},     /*gpu*/
        {1024},  /*nx*/
        {16384, 32768, 65536}, /*ny*/
    })
    ->UseRealTime() /* real time benchmark */
    ->Unit(benchmark::kMillisecond);

// ==========================================
// ==========================================
BENCHMARK_MAIN();
