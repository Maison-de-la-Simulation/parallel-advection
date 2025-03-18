#pragma once
#include "IAdvectorX.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <experimental/mdspan>
#include <limits>

using real_t = double;

using mdspan3d_t =
    std::experimental::mdspan<real_t, std::experimental::dextents<size_t, 3>,
                              std::experimental::layout_right>;
using mdspan2d_t =
    std::experimental::mdspan<real_t, std::experimental::dextents<size_t, 2>,
                              std::experimental::layout_right>;

/* Contains headers for different implementations of advector interface */
namespace AdvX {
using buff3d = sycl::buffer<double, 3>;

class Sequential : public IAdvectorX {
    using IAdvectorX::IAdvectorX;   // Inheriting constructor

  public:
    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;
};

/* For BasicRange kernels we have to do it out-of-place so we need a global
buffer that is the same size as the fdistrib buffer */
class BasicRange : public IAdvectorX {
  protected:
    buff3d m_global_buff_ftmp;

  public:
    BasicRange(const size_t n1, const size_t nvx, const size_t n2)
        : m_global_buff_ftmp{sycl::range<3>(nvx, n1, n2)} {}

    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;
};

// class Hierarchical : public IAdvectorX {
//     using IAdvectorX::IAdvectorX;

//   public:
//     sycl::event operator()(sycl::queue &Q, double *fdist_dev,
//                            const Solver &solver) override;
// };

class NDRange : public IAdvectorX {
    using IAdvectorX::IAdvectorX;

  public:
    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;
};

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
    sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                           const Solver &solver) override;

    AdaptiveWg() = delete;

    AdaptiveWg(const Solver &solver, sycl::queue q) {
        const auto n0 = solver.params.n0;
        const auto n1 = solver.params.n1;
        const auto n2 = solver.params.n2;

        dispatch_dim0_ = init_1d_blocking(n0, max_batchs_x_);
        dispatch_dim2_ = init_1d_blocking(n2, max_batchs_yz_);

        // SYCL query returns the size in bytes
        auto max_elem_local_mem =
            q.get_device().get_info<sycl::info::device::local_mem_size>() /
            sizeof(double);

        local_size_.set_ideal_sizes(solver.params.pref_wg_size, n0, n1, n2);
        local_size_.adjust_sizes_mem_limit(max_elem_local_mem, n1);

        wg_dispatch_.s0_ = solver.params.seq_size0;
        wg_dispatch_.s2_ = solver.params.seq_size2;
        //TODO: this line is overriden inside the kernel!!! useless
        wg_dispatch_.set_num_work_groups(n0, n2, dispatch_dim0_.n_batch_,
                                         dispatch_dim2_.n_batch_,
                                         local_size_.w0_, local_size_.w2_);

        // std::cout << "--------------------------------" << std::endl;
        // std::cout << "n_batch0        : " << dispatch_dim0_.n_batch_
        //           << std::endl;
        // std::cout << "n_batch2        : " << dispatch_dim2_.n_batch_
        //           << std::endl;
        // std::cout << "g0_: " << wg_dispatch_.g0_ << std::endl;
        // std::cout << "g2_: " << wg_dispatch_.g2_ << std::endl;
        // std::cout << "s0_: " << wg_dispatch_.s0_ << std::endl;
        // std::cout << "s2_: " << wg_dispatch_.s2_ << std::endl;

        // print_range("local_size", local_size_.range());
        // std::cout << "--------------------------------" << std::endl;
    }
};
}   // namespace AdvX
