#include <iostream>
#include <sycl/sycl.hpp>

#include <ConvSolver.hpp>
#include <impl/bkma.hpp>
#include <types.hpp>

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

inline int
compute_output_size(int Lin, int kernel_size) {
    return Lin - (kernel_size - 1);
}

// ==========================================
// ==========================================
int
main(int argc, char **argv) {
    const auto run_on_gpu = true;

    sycl::device d;
    if (run_on_gpu)
        try {
            d = sycl::device{sycl::gpu_selector_v};
        } catch (const sycl::exception e) {
            std::cout
                << "GPU was requested but none is available, running kernels "
                   "on the CPU\n"
                << std::endl;
            d = sycl::device{sycl::cpu_selector_v};
        }
    else
        d = sycl::device{sycl::cpu_selector_v};

    sycl::queue Q{d};

    /* Display infos on current device */
    std::cout << "Using device: "
              << Q.get_device().get_info<sycl::info::device::name>() << "\n";

    const auto channel_in = 3;
    const auto channel_out = channel_in;
    const auto length = 1024;

    const auto n0 = 512;                    // n
    const auto n1 = length * channel_out;   // l*oc
    const auto n2 = 512;                    // n
    const auto k = 9;

    span3d_t data(sycl::malloc_device<real_t>(n0 * n1 * n2, Q), n0,
                              n1, n2);
    span3d_t warmup_data(
        sycl::malloc_device<real_t>(n0 * n1 * n2, Q), n0, n1, n2);
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
        sycl::malloc_device<real_t>(k * channel_out * channel_in, Q);
    real_t *d_bias = sycl::malloc_device<real_t>(channel_out, Q);
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
    wi_dispatch.set_ideal_sizes(1024, n0, n1, n2);
    auto max_elem_local_mem =
        Q.get_device().get_info<sycl::info::device::local_mem_size>() /
        sizeof(real_t);
    wi_dispatch.adjust_sizes_mem_limit(max_elem_local_mem, n1);

    WorkGroupDispatch wg_dispatch;
    wg_dispatch.set_num_work_groups(n0, n2, 1, 1, wi_dispatch.w0_,
                                    wi_dispatch.w2_);

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
        bkma_run(Q, warmup_data, solver, optim_params).wait();

    auto start = std::chrono::high_resolution_clock::now();
    bkma_run(Q, data, solver, optim_params).wait();
    auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<real_t> elapsed_seconds = end - start;

    auto err = sum_and_normalize_conv(Q, data, compute_output_size(length, k));
    std::cout << "Normalized Array after: " << err << std::endl;
    std::cout << std::endl;

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
