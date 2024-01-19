#include "bench_utils.h"
#include <validation.h>

// ==========================================
// ==========================================
/* Benchmark a FakeAdvector to see the STREAM performance by operator() */
static void
DISABLED_BM_STREAM(benchmark::State &state){
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

// ==========================================
// ==========================================
/* Benchmark a 1D range iteration +1 */
static void
BM_STREAM(benchmark::State &state){
    auto p = createParams(state.range(0), state.range(1), state.range(2));
    p.wg_size = p.gpu ? 128 : 64;

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
    sycl::buffer<double, 1> fdist(sycl::range<1>(p.nvx*p.nx));

    /* Fill buffer with zeroes */
    Q.submit([&](sycl::handler &cgh) {
        sycl::accessor acc(fdist, cgh, sycl::read_write);
        cgh.parallel_for(fdist.get_range(), [=](sycl::id<1> itm){
            acc[itm] = 0;
        });
     }).wait();

    /* Benchmark */
    for (auto _ : state) {
        advector->stream_bench(Q, fdist).wait();
    }

    p.maxIter = state.iterations();

    state.SetItemsProcessed(p.maxIter * p.nvx * p.nx);
    state.SetBytesProcessed(p.maxIter * p.nvx * p.nx * sizeof(double));

    //No validation because it's FakeAdvector
}


// BENCHMARK(BM_STREAM)
//     ->ArgsProduct({
//         {1},     /*gpu*/
//         {1024},  /*nx*/
//         {16384, 32768, 65536}, /*ny*/
//     })
//     ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_STREAM)
    ->ArgsProduct({
        {1, 0},     /*gpu*/
        {1024},  /*nx*/
        benchmark::CreateRange(2<<5, 2<<20, /*multi=*/2), /*nx*/
    })
    ->UseRealTime() /* real time benchmark */
    ->Unit(benchmark::kMillisecond);

// ==========================================
// ==========================================
BENCHMARK_MAIN();
