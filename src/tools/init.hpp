#pragma once
#include <AdvectionParams.hpp>
#include <cmath>
#include <sycl/sycl.hpp>

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
}

// ==========================================
// ==========================================
void
fill_buffer_adv(sycl::queue &q, real_t *fdist_dev, const ADVParams &params) {
    const auto n0 = params.n0, n1 = params.n1, n2 = params.n2;

    sycl::range r3d(n0, n1, n2);
    q.submit([&](sycl::handler &cgh) {
         cgh.parallel_for(r3d, [=](auto i) {
             span3d_t fdist(fdist_dev, n0, n1, n2);
             const size_t i0 = i[0];
             const size_t i1 = i[1];
             const size_t i2 = i[2];

             real_t x = params.minRealX + i1 * params.dx;
             fdist(i0, i1, i2) = sycl::sin(4 * x * M_PI);
         });      // end parallel_for
     }).wait();   // end q.submit
}

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
}
