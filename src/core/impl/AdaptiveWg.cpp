#include "bkma.h"
#include <advectors.h>

// ==========================================
// ==========================================
sycl::event
AdvX::AdaptiveWg::operator()(sycl::queue &Q, real_t *fdist_dev,
                             const AdvectionSolver &solver) {

    BkmaOptimParams optim_params{
        dispatch_dim0_,  dispatch_dim2_, local_size_.w0_,   local_size_.w1_,
        local_size_.w2_, wg_dispatch_,   MemorySpace::Local};

    span3d_t data(fdist_dev, solver.params.n0, solver.params.n1,
                    solver.params.n2);
    return bkma_run(Q, data, solver, optim_params);
}
