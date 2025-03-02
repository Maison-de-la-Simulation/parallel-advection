#include "bench_utils.h"
#include <benchmark/benchmark.h>
#include <validation.h>


// ==========================================
// ==========================================
/* Benchmark the impact of wg_size on Hierarchical kernel */
static void
BM_HybridMem_p(benchmark::State &state) {

    BenchParams bench_params(state);
    auto& params = bench_params.adv_params;

    /* SYCL setup */
    auto Q = createSyclQueue(params.gpu, state);
    auto data =
        sycl::malloc_device<double>(params.n0 * params.n1 * params.n2, Q);

    /* Advector setup */
    Solver solver(params);
    const auto kernel_id = AdvImpl::HYBRID;
    const auto advector = advectorFactory(Q, params, solver, kernel_id, state);

    /* Benchmark infos */
    state.counters.insert({
        {"gpu", params.gpu},
        {"n0", params.n0},
        {"n1", params.n1},
        {"n2", params.n2},
        {"kernel_id", kernel_id},
        {"pref_wg_size", params.pref_wg_size},
        {"percent_loc", params.percent_loc},
        {"seq_size0", params.seq_size0},
        {"seq_size2", params.seq_size2},
    });

    /* Physics setup */
    fill_buffer(Q, data, params);
    Q.wait();

    /* Benchmark */
    for (auto _ : state) {
        try {
            advector(Q, data, solver);
            Q.wait();
        } catch (const sycl::exception &e) {
            state.SkipWithError(e.what());
        } catch (const std::exception &e) {
            state.SkipWithError(e.what());
            break;
        }
    }

    params.maxIter = state.iterations();

    state.SetItemsProcessed(params.maxIter * params.n0 * params.n1 * params.n2);
    state.SetBytesProcessed(params.maxIter * params.n0 * params.n1 * params.n2 *
                            sizeof(double));

    auto err = validate_result(Q, data, params, false);
    if (err > 10e-6) {
        state.SkipWithError("Validation failed with numerical error > 10e-6.");
    }

    sycl::free(data, Q);
}

// ==========================================
BENCHMARK(BM_HybridMem_p)
    ->ArgsProduct({
        {1},             /*gpu*/
        N0_RANGE,        /*n0*/
        N1_RANGE,            /*n1*/
        N2_RANGE,        /*n2*/
        {128, 512, 1024},         /*w*/
        PERCENT_LOC,
        SEQ_SIZE0,
        SEQ_SIZE2,
    })
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ==========================================
// ==========================================
BENCHMARK_MAIN();
