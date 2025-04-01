#include <iostream>
#include <sycl/sycl.hpp>

#include <ConvSolver.hpp>
#include <Conv1dParams.hpp>
#include <bkma.hpp>
#include <types.hpp>
#include <init.hpp>

real_t
sum_and_normalize_conv(sycl::queue &Q, span3d_t data, size_t nw) {
    auto n0 = data.extent(0);
    auto n2 = data.extent(2);
    sycl::range<3> r3d(n0, nw, n2);

    real_t sum = 0;
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

void
validate_conv1d(sycl::queue &Q, span3d_t data, size_t nw) {
    const auto n0 = data.extent(0);
    const auto n2 = data.extent(2);

    sycl::range<1> range0(n0);
    sycl::range<1> range2(n2);

    Q.parallel_for(range0, [=](unsigned i0) {
        for (auto i1 = 0; i1 < nw; ++i1) {
            for (auto i2 = 0; i2 < n2 - 1; ++i2) {
                if(data(i0, i1, i2) != data(i0, i1, i2+1)){
                    // throw std::runtime_error("nn");
                    data(0,0,0) = -12345;
                };
            }
        }
    });

    Q.parallel_for(range2, [=](unsigned i2) {
        for (auto i1 = 0; i1 < nw; ++i1) {
            for (auto i0 = 0; i0 < n0 - 1; ++i0) {
                if(data(i0, i1, i2) != data(i0+1, i1, i2)){
                    // throw std::runtime_error("nn");
                    data(0,0,0) = -45678;
                };
            }
        }
    });

    Q.wait();
    if (data(0,0,0) == -12345 || data(0,0,0) == -45678)
        std::cout << "WARNING: Values at same position i1 are not equivalent "
                     "throught the batchs. Check implementation."
                  << std::endl;
    else
        std::cout << "All values data[:,i1,:] are equal." << std::endl;
}



// ==========================================
// ==========================================
int
main(int argc, char **argv) {
    /* Read input parameters */
    std::string input_file = argc > 1 ? std::string(argv[1]) : "conv1d.ini";
    ConfigMap configMap(input_file);
    Conv1dParamsNonCopyable strParams;
    strParams.setup(configMap);

    const bool run_on_gpu = strParams.gpu;
    auto device = pick_device(run_on_gpu);
    strParams.gpu = device.is_gpu() ? true : false;

    sycl::queue Q{device};

    /* Display infos on current device */
    std::cout << "Using device: "
              << Q.get_device().get_info<sycl::info::device::name>() << "\n";

    /* Make trivially copyable params based on strParams*/
    strParams.print();
    Conv1dParams params(strParams);

    const auto channel_in = params.channel_in;
    const auto channel_out = params.channel_out;
    const auto length = params.length;

    const auto n0 = params.n0;   // n
    const auto n1 = params.n1;   // l*oc
    const auto n2 = params.n2;   // n
    const auto k = params.k;

    span3d_t data(sycl::malloc_shared<real_t>(n0 * n1 * n2, Q), n0,
                              n1, n2);
    span3d_t warmup_data(
        sycl::malloc_shared<real_t>(n0 * n1 * n2, Q), n0, n1, n2);
    Q.wait();

    Q.parallel_for(sycl::range<3>(n0, n1, n2), [=](auto itm) {
         auto i0 = itm[0];
         auto i1 = itm[1];
         auto i2 = itm[2];
         //  data(i0, i1, i2) = (i0+i1+i2)%10;
          data(i0, i1, i2) = 7.3;
          warmup_data(i0, i1, i2) = 1.0;
     }).wait();

    real_t *d_weight =
        sycl::malloc_shared<real_t>(k * channel_out * channel_in, Q);
    real_t *d_bias = sycl::malloc_shared<real_t>(channel_out, Q);
    Q.wait();
    Q.parallel_for(sycl::range<1>(k * channel_out * channel_in), [=](auto itm) {
         d_weight[itm] = 1.5;
     }).wait();
    Q.parallel_for(sycl::range<1>(channel_out), [=](auto itm) {
         d_bias[itm] = 1.0;
     }).wait();

    ConvSolver solver{
        span3d_t(d_weight, k, channel_in, channel_out),
        span1d_t(d_bias, channel_out), k, channel_in, length};

    WorkItemDispatch wi_dispatch;
    wi_dispatch.set_ideal_sizes(params.pref_wg_size, n0, n1, n2);
    auto max_elem_local_mem =
        Q.get_device().get_info<sycl::info::device::local_mem_size>() /
        sizeof(real_t);
    wi_dispatch.adjust_sizes_mem_limit(max_elem_local_mem, n1);

    WorkGroupDispatch wg_dispatch;
    wg_dispatch.set_num_work_groups(n0, n2, params.seq_size0, params.seq_size2,
                                    wi_dispatch.w0_, wi_dispatch.w2_);

    BkmaOptimParams optim_params{{1, n0, n0},       // BatchConfig1D dispatch_d0
                                 {1, n2, n2},       // BatchConfig1D dispatch_d2
                                 wi_dispatch.w0_,   // size_t w0
                                 wi_dispatch.w1_,   // size_t w1
                                 wi_dispatch.w2_,   // size_t w2
                                 wg_dispatch,       // WorkGroupDispatch wg_disp
                                 MemorySpace::Local};

    auto error = sum_and_normalize_conv(Q, data, n1);
    std::cout << std::endl;
    std::cout << "Normalized Array before: " << error << std::endl;

    /* Warmup to JIT model */
    for (int i = 0; i < 3; ++i)
        bkma_run<ConvSolver, BkmaImpl::AdaptiveWg>(Q, warmup_data, solver,
                                                   optim_params)
            .wait();

    auto start = std::chrono::high_resolution_clock::now();
    bkma_run<ConvSolver, BkmaImpl::AdaptiveWg>(Q, data, solver, optim_params)
        .wait();
    auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed_seconds = end - start;

    auto err = sum_and_normalize_conv(Q, data, params.n_write);
    std::cout << "Normalized Array after: " << err << std::endl;
    std::cout << std::endl;

    validate_conv1d(Q, data, params.n_write);

    //==========================================================================
    //==========================================================================
    std::cout << "PERF_DIAGS:" << std::endl;
    std::cout << "elapsed_time: " << elapsed_seconds.count() << " s\n";

    auto gcells = ((n0 * n1 * n2) / elapsed_seconds.count()) / 1e9;
    std::cout << "upd_cells_per_sec: " << gcells << " Gcell/sec\n";
    std::cout << "estimated_throughput: " << gcells * sizeof(real_t) * 2
              << " GB/s" << std::endl;

    sycl::free(d_weight, Q);
    sycl::free(d_bias, Q);
    sycl::free(data.data_handle(), Q);
    return 0;
}
