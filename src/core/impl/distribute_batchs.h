#pragma once
#include <IAdvectorX.h>

template <class Functor>
inline sycl::event
distribute_batchs(sycl::queue &Q, double *fdist_dev, const Solver &solver,
                  const BatchConfig1D &dispatch_d0,
                  const BatchConfig1D &dispatch_d2, const size_t orig_w0,
                  const size_t w1, const size_t orig_w2,
                  WorkGroupDispatch wg_dispatch, const size_t n0,
                  const size_t n1, const size_t n2, Functor submit_kernels) {

    sycl::event last_event;

    auto const &n_batch0 = dispatch_d0.n_batch_;
    auto const &n_batch2 = dispatch_d2.n_batch_;

    for (size_t i0_batch = 0; i0_batch < n_batch0; ++i0_batch) {
        bool last_i0 = (i0_batch == n_batch0 - 1);
        auto const offset_d0 = dispatch_d0.offset(i0_batch);

        for (size_t i2_batch = 0; i2_batch < n_batch2; ++i2_batch) {
            bool last_i2 = (i2_batch == n_batch2 - 1);
            auto const offset_d2 = dispatch_d2.offset(i2_batch);

            // Select the correct batch size
            auto &batch_size_d0 = last_i0 ? dispatch_d0.last_batch_size_
                                          : dispatch_d0.batch_size_;
            auto &batch_size_d2 = last_i2 ? dispatch_d2.last_batch_size_
                                          : dispatch_d2.batch_size_;

            if (last_i0 && last_i2) {
                last_event =
                    submit_kernels(Q, fdist_dev, solver, batch_size_d0,
                                   offset_d0, batch_size_d2, offset_d2, orig_w0,
                                   w1, orig_w2, wg_dispatch, n0, n1, n2);
            } else {
                submit_kernels(Q, fdist_dev, solver, batch_size_d0, offset_d0,
                               batch_size_d2, offset_d2, orig_w0, w1, orig_w2,
                               wg_dispatch, n0, n1, n2)
                    .wait();
            }
        }
    }

    return last_event;
}