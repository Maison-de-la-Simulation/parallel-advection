#pragma once
#include "IAdvectorX.h"
#include "impl/bkma.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <experimental/mdspan>

/* Contains headers for different implementations of advector interface */
namespace AdvX {

//==============================================================================
class BasicRange : public IAdvectorX {
  protected:
    sycl::queue q_;
    real_t *ftmp_;

  public:
    BasicRange(const AdvectionSolver &solver, sycl::queue q) {
        const auto n0 = solver.params.n0;
        const auto n1 = solver.params.n1;
        const auto n2 = solver.params.n2;

        ftmp_ = sycl::malloc_device<real_t>(n0 * n1 * n2, q_);
        q_.wait();
    }

    ~BasicRange() {
        sycl::free(ftmp_, q_);
        q_.wait();
    }

    sycl::event operator()(sycl::queue &Q, real_t *fdist_dev,
                           const AdvectionSolver &solver) override;
};

//==============================================================================
class NDRange : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q, real_t *fdist_dev,
                           const AdvectionSolver &solver) override;
};

//==============================================================================
class AdaptiveWg : public IAdvectorX {
  protected:
    using IAdvectorX::IAdvectorX;

    /* We should be able to query max_batchs to the API.
             |    x    |   y/z   |
        CUDA:| 2**31-1 | 2**16-1 |
        HIP :| 2**32-1 | 2**32-1 |
        L0  :| 2**32-1 | 2**32-1 | (compile with -fno-sycl-query-fit-in-int)
        CPU :        a lot             */
    const size_t max_batchs_x_ = 65536 - 1;
    const size_t max_batchs_yz_ = 65536 - 1;

    BatchConfig1D dispatch_dim0_;
    BatchConfig1D dispatch_dim2_;
    WorkItemDispatch local_size_;
    WorkGroupDispatch wg_dispatch_;

  public:
    sycl::event operator()(sycl::queue &Q, real_t *fdist_dev,
                           const AdvectionSolver &solver) override;

    AdaptiveWg() = delete;

    AdaptiveWg(const AdvectionSolver &solver, sycl::queue q) {
        const auto n0 = solver.params.n0;
        const auto n1 = solver.params.n1;
        const auto n2 = solver.params.n2;

        dispatch_dim0_ = init_1d_blocking(n0, max_batchs_x_);
        dispatch_dim2_ = init_1d_blocking(n2, max_batchs_yz_);

        // SYCL query returns the size in bytes
        auto max_elem_local_mem =
            q.get_device().get_info<sycl::info::device::local_mem_size>() /
            sizeof(real_t);

        local_size_.set_ideal_sizes(solver.params.pref_wg_size, n0, n1, n2);
        local_size_.adjust_sizes_mem_limit(max_elem_local_mem, n1);

        wg_dispatch_.s0_ = solver.params.seq_size0;
        wg_dispatch_.s2_ = solver.params.seq_size2;

        // TODO: this line is overriden inside the kernel!!! useless
        wg_dispatch_.set_num_work_groups(n0, n2, dispatch_dim0_.n_batch_,
                                         dispatch_dim2_.n_batch_,
                                         local_size_.w0_, local_size_.w2_);
    }
};
}   // namespace AdvX
