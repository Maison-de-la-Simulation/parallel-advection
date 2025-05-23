#include <ConvSolver.hpp>
#include <bkma.hpp>
// #include "bench_utils.hpp"
#include <benchmark/benchmark.h>
#include <types.hpp>

static constexpr auto __WG_SIZE = 512;
static constexpr real_t __INIT_VALUE = 7.3;

// ==========================================
struct BenchmarkConv1dParams {
    int batch_size;
    int input_length;
    int kernel_size;
    int channels;
};

static std::vector<BenchmarkConv1dParams> configs ={
    {16384, 512 , 3 , 1},
    {1024 , 512 , 3 , 1},
    {32768, 128 , 3 , 9},
    {16384, 256 , 5 , 6},
    {16384, 512 , 5 , 3},
    {16384, 1024, 11, 1}
};

// ==========================================
real_t
sum_and_normalize_conv(sycl::queue &Q, span3d_t data, size_t nw) {
    auto n0 = data.extent(0);
    auto n2 = data.extent(2);
    sycl::range<3> r3d(n0, nw, n2);

    real_t sum = -1;
    {
        sycl::buffer<real_t> buff_sum(&sum, 1);

        Q.submit([&](sycl::handler &cgh) {
             auto reduc_sum = sycl::reduction(buff_sum, cgh, sycl::plus<>());

             cgh.parallel_for(r3d, reduc_sum, [=](auto itm, auto &reduc_sum) {
                 auto i0 = itm[0];
                 auto i1 = itm[1];
                 auto i2 = itm[2];
                 auto f = data(i0, i1, i2);

                 reduc_sum += f;
             });
         }).wait();
    }
    sum /= (n0 * nw * n2);

    return sum;
}

// ==========================================
inline int
compute_output_size(int Lin, int kernel_size) {
    return Lin - (kernel_size - 1);
}

// ==========================================
BkmaOptimParams
create_bkma_params(sycl::queue &q, const size_t n0, const size_t n1,
                   const size_t n2, const size_t w) {

    WorkItemDispatch wi_dispatch;
    wi_dispatch.set_ideal_sizes(w, n0, n1, n2);
    auto max_elem_local_mem =
        q.get_device().get_info<sycl::info::device::local_mem_size>() /
        sizeof(real_t);
    wi_dispatch.adjust_sizes_mem_limit(max_elem_local_mem, n1);

    WorkGroupDispatch wg_dispatch;
    wg_dispatch.set_num_work_groups(n0, n2, 1, 1, wi_dispatch.w0_,
                                    wi_dispatch.w2_);

    return {{1, n0, n0},     {1, n2, n2}, wi_dispatch.w0_,   wi_dispatch.w1_,
            wi_dispatch.w2_, wg_dispatch, MemorySpace::Local};
}

static void
BM_Conv1d(benchmark::State &state) {
    sycl::queue q;

    BenchmarkConv1dParams conv_params = configs[state.range(0)];
    const size_t c_out = conv_params.channels;
    const size_t c_in = conv_params.channels;
    const size_t k = conv_params.kernel_size;
    const size_t length = conv_params.input_length;

    const size_t n0 = conv_params.batch_size;
    const size_t n1 = length * c_in;
    const size_t n2 = 1;   // TODO: try to split batch_size into n0 and n1
    /* App setup */
    span3d_t data(sycl_alloc(n0 * n1 * n2, q), n0, n1, n2);
    span3d_t warmup_data(sycl_alloc(n0 * n1 * n2, q), n0, n1, n2);
    span3d_t weights(sycl_alloc(k * c_out * c_in, q), k, c_in, c_out);
    span1d_t bias(sycl_alloc(c_out, q), c_out);
    q.wait();

    q.parallel_for(sycl::range<1>(bias.size()), [=](auto itm) {
         bias.data_handle()[itm] = 1.0;
     });
    q.parallel_for(sycl::range<1>(weights.size()), [=](auto itm) {
         weights.data_handle()[itm] = 1.5;
     });
    q.parallel_for(sycl::range<3>(n0, n1, n2), [=](auto itm) {
         data(itm[0], itm[1], itm[2]) = __INIT_VALUE;
         warmup_data(itm[0], itm[1], itm[2]) = __INIT_VALUE;
     });
     q.wait();

    ConvSolver solver{weights, bias, k, c_in, length};
    auto bkma_params = create_bkma_params(q, n0, n1, n2, __WG_SIZE);

    /* Warmup to JIT model */
    for (int i = 0; i < 3; ++i)
        bkma_run<ConvSolver, BkmaImpl::AdaptiveWg>(q, warmup_data, solver,
            bkma_params).wait();

    /* Benchmark */
    for (auto _ : state) {
        try {
            bkma_run<ConvSolver, BkmaImpl::AdaptiveWg>(q, data, solver,
                                                       bkma_params)
                .wait();
        } catch (const sycl::exception &e) {
            state.SkipWithError(e.what());
        } catch (const std::exception &e) {
            state.SkipWithError(e.what());
            break;
        }
    }

    auto const n_iters = state.iterations();

    state.SetItemsProcessed(n_iters * n0 * n1 * n2);
    state.SetBytesProcessed(n_iters * n0 * n1 * n2 * sizeof(real_t) * 2);

    auto result =
        sum_and_normalize_conv(q, data, compute_output_size(length, k));

    /* Benchmark infos */
    state.counters.insert({
        {"n0", n0},
        {"n1", n1},
        {"n2", n2},
        {"pref_wg_size", __WG_SIZE},
        {"seq_size0", bkma_params.wg_dispatch.s0_},
        {"seq_size2", bkma_params.wg_dispatch.s2_},
        {"batch_size", conv_params.batch_size},
        {"input_length", conv_params.input_length},
        {"kernel_size", conv_params.kernel_size},
        {"channels", conv_params.channels},
        {"result", result},
    });
}

// ==========================================
BENCHMARK(BM_Conv1d)
    ->Name("main-BKM-bench")
    ->Iterations(1)
    ->ArgsProduct({benchmark::CreateDenseRange(0, configs.size()-1, 1)})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ==========================================
// ==========================================
BENCHMARK_MAIN();
