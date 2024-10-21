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
    auto advector = advectorFactory(kernel_id, p.n1, p.n0, state);

    p.maxIter = state.range(4);
    /* Benchmark infos */
    state.counters.insert({
        {"gpu", p.gpu},
        {"n1", p.n1},
        {"n0", p.n0},
        {"kernel_id", kernel_id},
        {"nIter", p.maxIter},
    });

    /* SYCL setup */
    auto Q = createSyclQueue(p.gpu, state);
    sycl::buffer<double, 3> fdist(sycl::range<3>(p.n0, p.n1));

    /* Physics setup */
    fill_buffer(Q, fdist, p);
    Q.wait();

    /* Benchmark */
    for (auto _ : state) {
        try
        {
            for (size_t i = 0; i < p.maxIter-1; i++)
                advector(Q, fdist, p);

            advector(Q, fdist, p).wait();

        } catch (const std::exception &e) {
            state.SkipWithError(e.what());
            break; // REQUIRED to prevent all further iterations.
        } catch (const sycl::exception &e) {
            state.SkipWithError(e.what());
            break;
        }
    }

    state.counters.insert({
        {"nRepet", state.iterations()},
    });

    state.SetItemsProcessed(state.iterations() * p.maxIter * p.n0 * p.n1);
    state.SetBytesProcessed(state.iterations() * p.maxIter * p.n0 * p.n1 *
                            sizeof(double));

    p.maxIter *= state.iterations();
    auto err = validate_result(Q, fdist, p, false);
    if (err > 10e-6) {
        state.SkipWithError("Validation failled with numerical error > 10e-6.");
    }
}   // end BM_Advector

BENCHMARK(BM_Advector)
    ->ArgsProduct({
        {0, 1}, /*gpu*/
        {n1}, /*n1*/
        NY_SMALL_RANGE, /*n0*/
        IMPL_RANGE, /*kernel_id*/
        {1, 2, 10, 50, 100, 1000, 10000} /*p.maxIter*/
    }) 
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
