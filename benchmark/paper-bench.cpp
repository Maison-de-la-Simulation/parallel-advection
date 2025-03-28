#include "bench_config.h"
#include "bench_utils.h"
#include <benchmark/benchmark.h>
#include <validation.h>
#include "../src/types.h"


// ==========================================
// ==========================================
/* Benchmark the impact of wg_size on Hierarchical kernel */
static void
BM_Advection(benchmark::State &state) {

    BenchParams bench_params(state);
    auto& params = bench_params.adv_params;
    params.n0 = EXP_SIZES[state.range(2)].n0_;
    params.n1 = EXP_SIZES[state.range(2)].n1_;
    params.n2 = EXP_SIZES[state.range(2)].n2_;

    /* SYCL setup */
    auto Q = createSyclQueue(params.gpu, state);
    auto data =
        sycl::malloc_device<real_t>(params.n0 * params.n1 * params.n2, Q);

    /* Advector setup */
    Solver solver(params);
    const auto kernel_id = static_cast<AdvImpl>(state.range(1));
    const auto advector = advectorFactory(Q, params, solver, kernel_id, state);

    /* Benchmark infos */
    state.counters.insert({
        {"gpu", params.gpu},
        {"n0", params.n0},
        {"n1", params.n1},
        {"n2", params.n2},
        {"kernel_id", kernel_id},
        {"pref_wg_size", params.pref_wg_size},
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
                            sizeof(real_t));
    auto err = validate_result(Q, data, params, false);

    // if (err > 10e-6) {
    //     state.SkipWithError("Validation failed with numerical error > 10e-6.");
    // }

    state.counters.insert({{"err", err}});

    sycl::free(data, Q);
    Q.wait();
}

// ==========================================
BENCHMARK(BM_Advection)->Name("main-BKM-bench")
    ->ArgsProduct({
        {/*0, */1}, /*gpu*/
        IMPL_RANGE, /* impl */
        benchmark::CreateDenseRange(0, 9, 1), /*size from the array*/
        {128, 1024},         /*w*/
        SEQ_SIZE0,
        SEQ_SIZE2,
    })
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ==========================================
// ==========================================
BENCHMARK_MAIN();
