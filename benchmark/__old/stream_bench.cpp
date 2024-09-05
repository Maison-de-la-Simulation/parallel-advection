#include "bench_utils.h"
#include <validation.h>

// ==========================================
// ==========================================
/* Benchmark a 1D range iteration +1 */
static void
BM_STREAM(benchmark::State &state){
    auto p = createParams(
        state.range(0), state.range(1), state.range(2), state.range(3));
    p.wg_size = p.gpu ? 128 : 64;

    /* Advector setup */
    const auto advector = sref::make_unique<AdvX::FakeAdvector>();

    /* Benchmark infos */
    state.counters.insert({
        {"gpu", p.gpu},
        {"ny", p.ny},
        {"nx", p.nx},
        {"ny1", p.ny1},
        {"kernel_id", -1},
        {"wg_size", p.wg_size},
    });

    /* SYCL setup */
    auto Q = createSyclQueue(p.gpu, state);
    sycl::buffer<double, 1> fdist(sycl::range<1>(p.ny*p.nx*p.ny1));

    /* Fill buffer with zeroes */
    Q.submit([&](sycl::handler &cgh) {
        sycl::accessor acc(fdist, cgh, sycl::read_write, sycl::no_init);
        cgh.parallel_for(fdist.get_range(), [=](sycl::id<1> itm){
            acc[itm] = 0;
        });
     }).wait();

    /* Benchmark */
    for (auto _ : state) {
        advector->stream_bench(Q, fdist).wait();
    }

    p.maxIter = state.iterations();

    state.SetItemsProcessed(p.maxIter * p.ny * p.nx);
    state.SetBytesProcessed(p.maxIter * p.ny * p.nx * sizeof(double));

    //No validation because it's FakeAdvector
}

BENCHMARK(BM_STREAM)
    ->ArgsProduct({
        {1, 0},     /*gpu*/
        NY_RANGE, /*ny*/
        {NX},  /*nx*/
        NS_RANGE /*ny1*/
    })
    ->UseRealTime() /* real time benchmark */
    ->Unit(benchmark::kMillisecond);

// ==========================================
// ==========================================
BENCHMARK_MAIN();