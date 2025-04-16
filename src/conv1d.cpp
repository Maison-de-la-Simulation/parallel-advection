#include <iostream>
#include <sycl/sycl.hpp>

#include <ConvSolver.hpp>
#include <Conv1dParams.hpp>
#include <bkma.hpp>
#include <types.hpp>
#include <init.hpp>
#include <validation.hpp>

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

    const auto c_in = params.channel_in;
    const auto c_out = params.channel_out;
    const auto length = params.length;

    const auto n0 = params.n0;   // n
    const auto n1 = params.n1;   // l*oc
    const auto n2 = params.n2;   // n
    const auto k = params.k;

    span3d_t data(sycl_alloc(n0 * n1 * n2, Q), n0, n1, n2);
    span3d_t warmup_data(sycl_alloc(n0 * n1 * n2, Q), n0, n1, n2);
    span3d_t weight(sycl_alloc(k * c_out * c_in, Q), k, c_in, c_out);
    span1d_t bias(sycl_alloc(c_out, Q), c_out);
    Q.wait();

    fill_buffer_conv1d(Q, data, warmup_data, weight, bias);

    ConvSolver solver{weight, bias, k, c_in, length};

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
                                 1,1,1,1, //subgroups stuff. TODO: update for conv1d
                                 32, //simd_size
                                 MemorySpace::Local};

    auto error = sum_and_normalize_conv1d(Q, data, n1);
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

    auto err = sum_and_normalize_conv1d(Q, data, params.n_write);
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

    sycl::free(weight.data_handle(), Q);
    sycl::free(bias.data_handle(), Q);
    sycl::free(data.data_handle(), Q);
    Q.wait();

    return 0;
}
