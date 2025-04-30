#include "bench_config.hpp"
#include "bench_utils.hpp"
#include <AdvectionParams.hpp>
#include <AdvectionSolver.hpp>
#include <sycl/sycl.hpp>
#include <init.hpp>
#include <validation.hpp>

#include <bkma.hpp>
#include <types.hpp>
#include <impl_selector.hpp>



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
    auto const &n0 = params.n0;
    auto const &n1 = params.n1;
    auto const &n2 = params.n2;

    /* SYCL setup */
    auto Q = createSyclQueue(params.gpu, state);
    span3d_t data(sycl_alloc(n0*n1*n2, Q), n0, n1, n2);
    Q.wait();
    fill_buffer_adv(Q, data, params);
    Q.wait();
    
    /* Advector setup */
    AdvectionSolver solver(params);
    auto optim_params = create_optim_params<ADVParams>(Q, params);
    auto impl_str = state.range(1) == 0 ? "ndrange" : "adaptivewg";
    auto bkma_run_function = impl_selector<AdvectionSolver>(impl_str);

    /* Benchmark */
    for (auto _ : state) {
        try {
            bkma_run_function(Q, data, solver, optim_params, span3d_t{});
            Q.wait();
        } catch (const sycl::exception &e) {
            state.SkipWithError(e.what());
        } catch (const std::exception &e) {
            state.SkipWithError(e.what());
            break;
        }
    }

    params.maxIter = state.iterations();

    state.SetItemsProcessed(params.maxIter * n0 * n1 * n2);
    state.SetBytesProcessed(params.maxIter * n0 * n1 * n2 *
                            sizeof(real_t)*2);
    auto err = validate_result_adv(Q, data, params, false);

    // if (err > 10e-6) {
    //     state.SkipWithError("Validation failed with numerical error > 10e-6.");
    // }

        /* Benchmark infos */
        state.counters.insert({
            {"gpu", params.gpu},
            {"n0", n0},
            {"n1", n1},
            {"n2", n2},
            {"kernel_id", state.range(1)},
            {"pref_wg_size", params.pref_wg_size},
            {"seq_size0", params.seq_size0},
            {"seq_size2", params.seq_size2},
            {"err", err}
        });

    sycl::free(data.data_handle(), Q);
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
