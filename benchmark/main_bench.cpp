#include "bench_utils.h"
#include <validation.h>

// ==========================================
// ==========================================
/* Benchmark the impact of wg_size on Hierarchical kernel */
static void
BM_WgSize(benchmark::State &state){
    auto p = createParams(
        state.range(0), state.range(1), state.range(2), state.range(3));

    /* Advector setup */
    const auto kernel_id = AdvImpl::HIER;
    const auto hierAdv = advectorFactory(kernel_id, p, state);
    p.wg_size_x = state.range(4);

    /* Benchmark infos */
    state.counters.insert({
        {"gpu", p.gpu},
        {"ny", p.ny},
        {"nx", p.nx},
        {"ny1", p.ny1},
        {"kernel_id", kernel_id},
        {"wg_size_x", p.wg_size_x},
    });

    /* SYCL setup */
    auto Q = createSyclQueue(p.gpu, state);
    sycl::buffer<double, 3> fdist(sycl::range<3>(p.ny, p.nx, p.ny1));

    /* Physics setup */
    fill_buffer(Q, fdist, p);
    Q.wait();

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

    state.SetItemsProcessed(p.maxIter * p.ny * p.nx * p.ny1);
    state.SetBytesProcessed(p.maxIter * p.ny * p.nx * p.ny1 * sizeof(double));

    auto err = validate_result(Q, fdist, p, false);
    if (err > 10e-6) {
        state.SkipWithError("Validation failed with numerical error > 10e-6.");
    }
}

// ==========================================
BENCHMARK(BM_WgSize)
    ->ArgsProduct({
        {0, 1}, /*gpu*/
        NB_RANGE, /*ny*/
        {NX}, /*nx*/
        NS_RANGE, /*ny1*/
        WG_SIZES_X_RANGE /*wg_size*/
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
    auto p = createParams(
        state.range(0), state.range(1), state.range(2), state.range(3));
    p.wg_size_x = p.gpu ? 128 : 64;

    /* Advector setup */
    auto kernel_id = static_cast<AdvImpl>(static_cast<int>(state.range(4)));
    auto advector = advectorFactory(kernel_id, p, state);

    /* Benchmark infos */
    state.counters.insert({
        {"gpu", p.gpu},
        {"ny", p.ny},
        {"nx", p.nx},
        {"ny1", p.ny1},
        {"kernel_id", kernel_id},
        {"wg_size_x", p.wg_size_x},
    });

    /* SYCL setup */
    auto Q = createSyclQueue(p.gpu, state);
    sycl::buffer<double, 3> fdist(sycl::range<3>(p.ny, p.nx, p.ny1));

    /* Physics setup */
    fill_buffer(Q, fdist, p);
    Q.wait();

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

    state.SetItemsProcessed(p.maxIter * p.ny * p.nx * p.ny1);
    state.SetBytesProcessed(p.maxIter * p.ny * p.nx * p.ny1 * sizeof(double));

    state.counters.insert({{"maxIter", p.maxIter}});

    auto err = validate_result(Q, fdist, p, false);
    if (err > 10e-6) {
        state.SkipWithError("Validation failed with numerical error > 10e-6.");
    }
    if (err == 0) {
        state.SkipWithError("Validation failed with numerical error == 0. "
                            "Kernel probably didn't run");
    }
}   // end BM_Advector

// ================================================
BENCHMARK(BM_Advector)
    ->ArgsProduct({
        {1},
        {4096, 8192, 16384, 32768, 65535, 131070, 262140, 524288}, /*ny*/
        {NX}, /*nx*/
        {1}, /*ny1*/
        {AdvImpl::HIER, AdvImpl::EXP1}, /*impl*/
    })
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ==========================================
// ==========================================
BENCHMARK_MAIN();
