#include <sycl/sycl.hpp>
#include <iostream>

#include "core/impl/bkma.h"
#include "core/ConvSolver.h"


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

    const auto n0 = 16384; //n
    const auto n1 = 1024; //l
    const auto n2 = 1; //n
    const auto k = 1;
    const auto channel_out = 1;
    const auto channel_in = 1;

    ConvSolver::span3d_t data(sycl::malloc_device<double>(n0 * n1 * n2, Q), n0,
                              n1, n2);
    Q.wait();

    Q.parallel_for(sycl::range<3>(n0,n1,n2), [=](auto itm){
        auto i0 = itm[0];
        auto i1 = itm[1];
        auto i2 = itm[2];
        data(i0, i1, i2) = i0+i1+i2;
    }).wait();

    double *d_weight =
        sycl::malloc_device<double>(k * channel_out * channel_in, Q);
    double* d_bias = sycl::malloc_device<double>(channel_out, Q);

    Q.parallel_for(sycl::range<1>(k * channel_out * channel_in), [=](auto itm) {
         d_weight[itm] = itm;
     }).wait();
    Q.parallel_for(sycl::range<1>(channel_out), [=](auto itm){
        d_bias[itm] = itm;
    }).wait();

    ConvSolver solver{ConvSolver::span1d_t(d_weight, k),
                      ConvSolver::span0d_t(d_bias), k, channel_in};

    WorkItemDispatch wi_dispatch;
    wi_dispatch.set_ideal_sizes(512, n0, n1, n2);
    WorkGroupDispatch wg_dispatch;
    wg_dispatch.set_num_work_groups(n0, n2, 1, 1, wi_dispatch.w0_,
                                    wi_dispatch.w2_);

    BkmaOptimParams optim_params{
        {1, n0, n0},// BatchConfig1D dispatch_d0
        {1, n2, n2},// BatchConfig1D dispatch_d2
        wi_dispatch.w0_,// size_t w0
        wi_dispatch.w1_,// size_t w1
        wi_dispatch.w2_,// size_t w2
        wg_dispatch,// WorkGroupDispatch wg_disp
        MemorySpace::Local
    };
    bkma_run(Q, data, solver, optim_params).wait();

    // std::cout << "PERF_DIAGS:" << std::endl;
    // std::cout << "elapsed_time: " << elapsed_seconds.count() << " s\n";

    // auto gcells = ((n0*n1*n2*(maxIter)) / elapsed_seconds.count()) / 1e9;
    // std::cout << "upd_cells_per_sec: " << gcells << " Gcell/sec\n";
    // std::cout << "estimated_throughput: " << gcells * sizeof(double) * 2
    //           << " GB/s" << std::endl;

    sycl::free(d_weight, Q);
    sycl::free(d_bias, Q);
    return 0;
}
