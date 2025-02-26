#include "submit_kernels.h"
#include "distribute_batchs.h"

// ==========================================
// ==========================================
sycl::event
AdvX::AdaptiveWg::operator()(sycl::queue &Q, double *fdist_dev,
                             const Solver &solver) {

    return distribute_batchs(
        Q, fdist_dev, solver, dispatch_dim0_, dispatch_dim2_, local_size_.w0_,
        local_size_.w1_, local_size_.w2_, wg_dispatch_, solver.params.n0,
        solver.params.n1, solver.params.n2, submit_local_kernels);
}
