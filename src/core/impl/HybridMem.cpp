#include "distribute_batchs.h"
#include "impl/submit_global_kernels.h"
#include "submit_local_kernels.h"

sycl::event
AdvX::HybridMem::operator()(sycl::queue &Q, double *fdist_dev,
                                const Solver &solver) {
    return distribute_batchs(
        Q, fdist_dev, solver, dispatch_dim0_, dispatch_dim2_, local_size_.w0_,
        local_size_.w1_, local_size_.w2_, wg_dispatch_, solver.params.n0,
        solver.params.n1, solver.params.n2,
        [&](sycl::queue &Q, double *fdist_dev, const Solver &solver,
            size_t batch_size_d0, size_t offset_d0, size_t batch_size_d2,
            size_t offset_d2, size_t w0, size_t w1, size_t w2,
            WorkGroupDispatch wg_dispatch, size_t n0, size_t n1, size_t n2) {
            return submit_local_kernel(
                Q, fdist_dev, solver, kernel_dispatch_.k_local_, offset_d0,
                batch_size_d2, offset_d2, local_size_.w0_, local_size_.w1_,
                local_size_.w2_, wg_dispatch, n0, n1, n2);
        },
        [&](sycl::queue &Q, double *fdist_dev, const Solver &solver,
            size_t batch_size_d0, size_t offset_d0, size_t batch_size_d2,
            size_t offset_d2, size_t w0, size_t w1, size_t w2,
            WorkGroupDispatch wg_dispatch, size_t n0, size_t n1, size_t n2) {
            return submit_global_kernel(
                Q, fdist_dev, global_scratch_, solver,
                kernel_dispatch_.k_global_,
                offset_d0 + kernel_dispatch_.k_local_, batch_size_d2, offset_d2,
                local_size_global_kernels_.w0_, local_size_global_kernels_.w1_,
                local_size_global_kernels_.w2_, wg_dispatch_global_kernels_, n0,
                n1, n2);
        });
}
