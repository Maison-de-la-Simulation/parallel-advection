#include "distribute_batchs.h"
#include "impl/submit_global_kernels.h"
#include "submit_local_kernels.h"

// ==========================================
// ==========================================
sycl::event
AdvX::HybridKernels::operator()(sycl::queue &Q, double *fdist_dev,
                                const Solver &solver) {

    sycl::event last_event_local, last_event_global;

    auto const &n_batch0 = dispatch_dim0_.n_batch_;
    auto const &n_batch2 = dispatch_dim2_.n_batch_;

    for (size_t i0_batch = 0; i0_batch < n_batch0; ++i0_batch) {
        bool last_i0 = (i0_batch == n_batch0 - 1);
        auto const offset_d0 = dispatch_dim0_.offset(i0_batch);

        for (size_t i2_batch = 0; i2_batch < n_batch2; ++i2_batch) {
            bool last_i2 = (i2_batch == n_batch2 - 1);
            auto const offset_d2 = dispatch_dim2_.offset(i2_batch);

            auto &k_local = last_i0 ? last_kernel_dispatch_.k_local_
                                    : kernel_dispatch_.k_local_;
            auto &k_global = last_i0 ? last_kernel_dispatch_.k_global_
                                     : kernel_dispatch_.k_global_;

            // Same batch_size in dim2 for both k_local and k_global
            auto &batch_size_d2 = last_i2 ? dispatch_dim2_.last_batch_size_
                                          : dispatch_dim2_.batch_size_;

            if (k_local > 0)
                last_event_local = submit_local_kernel(
                    Q, fdist_dev, solver, k_local, offset_d0, batch_size_d2,
                    offset_d2, local_size_.w0_, local_size_.w1_,
                    local_size_.w2_, wg_dispatch_, solver.params.n0,
                    solver.params.n1, solver.params.n2);

            if (k_global > 0)
                last_event_global = submit_global_kernel(
                    Q, fdist_dev, global_scratch_, solver, k_global,
                    offset_d0 + k_local, batch_size_d2, offset_d2,
                    local_size_global_kernels_.w0_,
                    local_size_global_kernels_.w1_,
                    local_size_global_kernels_.w2_, wg_dispatch_global_kernels_,
                    solver.params.n0, solver.params.n1, solver.params.n2);

            if (last_i0 && last_i2) {
                // Do nothing
                last_event_local.wait();
            } else {
                Q.wait();
            }
        }
    }

    // maybe we should wait the last with most work: likely to be the longest
    return last_event_global;
}
