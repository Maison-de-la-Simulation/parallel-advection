#include "submit_local_kernels.h"

// ==========================================
// ==========================================
sycl::event
AdvX::AdaptiveWg::operator()(sycl::queue &Q, double *fdist_dev,
                             const Solver &solver) {

    sycl::event last_event;
    auto const &n_batch0 = dispatch_dim0_.n_batch_;
    auto const &n_batch2 = dispatch_dim2_.n_batch_;

    for (size_t i0_batch = 0; i0_batch < n_batch0; ++i0_batch) {
        bool last_i0 = (i0_batch == n_batch0 - 1);
        auto const offset_d0 = dispatch_dim0_.offset(i0_batch);

        for (size_t i2_batch = 0; i2_batch < n_batch2; ++i2_batch) {
            bool last_i2 = (i2_batch == n_batch2 - 1);
            auto const offset_d2 = dispatch_dim2_.offset(i2_batch);

            // Select the correct batch size
            size_t &batch_size_d0 = last_i0 ? dispatch_dim0_.last_batch_size_
                                            : dispatch_dim0_.batch_size_;
            size_t &batch_size_d2 = last_i2 ? dispatch_dim2_.last_batch_size_
                                            : dispatch_dim2_.batch_size_;

            if (last_i0 && last_i2) {
                last_event = submit_local_kernel(
                    Q, fdist_dev, solver, batch_size_d0, offset_d0,
                    batch_size_d2, offset_d2, local_size_.w0_, local_size_.w1_,
                    local_size_.w2_, wg_dispatch_, solver.params.n0,
                    solver.params.n1, solver.params.n2);
            } else {
                submit_local_kernel(
                    Q, fdist_dev, solver, batch_size_d0, offset_d0,
                    batch_size_d2, offset_d2, local_size_.w0_, local_size_.w1_,
                    local_size_.w2_, wg_dispatch_, solver.params.n0,
                    solver.params.n1, solver.params.n2)
                    .wait();
            }
        }
    }

    return last_event;
}
