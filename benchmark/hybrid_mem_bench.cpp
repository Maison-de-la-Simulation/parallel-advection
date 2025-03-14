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
        {0, 1}, /*gpu*/
        // EXP_RANGE,
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



/*
// =============================================
// =============================================
struct BenchParams {
    int64_t gpu, n0, n1, n2, w, percent_loc, s0, s2;

    ADVParams adv_params;

    BenchParams() = delete;
    BenchParams(benchmark::State &state)
        : gpu(state.range(0)), n0(state.range(1)), n1(state.range(2)),
          n2(state.range(3)), w(state.range(4)), percent_loc(state.range(5)),
          s0(state.range(6)), s2(state.range(7)) {
        
            init_params();
          }

    void init_params(){
        auto &p = adv_params;

        p.outputSolution = false;
        p.pref_wg_size = w;

        p.dt = 0.001;
        p.minRealX = 0;
        p.maxRealX = 1;
        p.minRealVx = -1;
        p.maxRealVx = 1;

        p.gpu = gpu;
        p.n1 = n1;
        p.n0 = n0;
        p.n2 = n2;

        p.percent_loc = static_cast<float>(percent_loc/100.0);
        p.seq_size0 = s0;
        p.seq_size2 = s2;

        p.update_deltas();
    }
};

*/