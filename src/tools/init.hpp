#pragma once
#include <AdvectionParams.hpp>
#include <cmath>
#include <sycl/sycl.hpp>
#include <bkma.hpp>

// ==========================================
// ==========================================
inline sycl::device
pick_device(bool run_on_gpu) {
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

    return d;
} //end pick_device

// ==========================================
// ==========================================
void
fill_buffer_adv(sycl::queue &q, span3d_t &data, const ADVParams &params) {
    const auto n0 = params.n0, n1 = params.n1, n2 = params.n2;

    sycl::range r3d(n0, n1, n2);
    q.submit([&](sycl::handler &cgh) {
         cgh.parallel_for(r3d, [=](auto i) {
             const size_t i0 = i[0];
             const size_t i1 = i[1];
             const size_t i2 = i[2];

             real_t x = params.minRealX + i1 * params.dx;
             data(i0, i1, i2) = sycl::sin(4 * x * M_PI);
         });      // end parallel_for
     }).wait();   // end q.submit
} // end fill_buffer_adv

// ==========================================
// ==========================================
void
fill_buffer_conv1d(sycl::queue &q, span3d_t &data, span3d_t &warmup_data,
                   span3d_t &weight, span1d_t &bias) {

    q.parallel_for(
        sycl::range<3>(data.extent(0), data.extent(1), data.extent(2)),
        [=](auto itm) {
            auto i0 = itm[0];
            auto i1 = itm[1];
            auto i2 = itm[2];

            data(i0, i1, i2) = 7.3;
            warmup_data(i0, i1, i2) = 1.0;
        });

    q.parallel_for(
        sycl::range<3>(weight.extent(0), weight.extent(1), weight.extent(2)),
        [=](auto itm) { weight(itm[0], itm[1], itm[2]) = 1.5; });
    q.parallel_for(sycl::range<1>(bias.extent(0)),
                   [=](unsigned itm) { bias(itm) = 1.0; });

    q.wait();
} // end fill_buffer_conv1d

// ==========================================
// ==========================================
template <typename Params>
BkmaOptimParams create_optim_params(sycl::queue &q, const Params &params) {
    const auto n0 = params.n0;
    const auto n1 = params.n1;
    const auto n2 = params.n2;

    WorkItemDispatch wi_dispatch;
    wi_dispatch.set_ideal_sizes(params.pref_wg_size, n0, n1, n2);
    auto max_elem_local_mem =
        q.get_device().get_info<sycl::info::device::local_mem_size>() /
        sizeof(real_t);
    wi_dispatch.adjust_sizes_mem_limit(max_elem_local_mem, n1);

    WorkGroupDispatch wg_dispatch;
    wg_dispatch.set_num_work_groups(n0, n2, params.seq_size0, params.seq_size2,
                                    wi_dispatch.w0_, wi_dispatch.w2_);

    /* TODO : here compute the number of batchs */
    return BkmaOptimParams{
        {1, n0, n0},         // BatchConfig1D dispatch_d0
        {1, n2, n2},         // BatchConfig1D dispatch_d2
        wi_dispatch.w0_,     // size_t w0
        wi_dispatch.w1_,     // size_t w1
        wi_dispatch.w2_,     // size_t w2
        wg_dispatch,         // WorkGroupDispatch wg_disp
        MemorySpace::Local}; /* TODO : change this depending on params*/
} //end create_optim_params
