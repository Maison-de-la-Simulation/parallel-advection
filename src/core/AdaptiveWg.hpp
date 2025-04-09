#pragma once
#include <bkma_tools.hpp>

// //==============================================================================
// class AdaptiveWg : public IAdvectorX {
//     protected:
//       using IAdvectorX::IAdvectorX;
  
//       /* We should be able to query max_batchs to the API.
//                |    x    |   y/z   |
//           CUDA:| 2**31-1 | 2**16-1 |
//           HIP :| 2**32-1 | 2**32-1 |
//           L0  :| 2**32-1 | 2**32-1 | (compile with -fno-sycl-query-fit-in-int)
//           CPU :        a lot             */
//       const size_t max_batchs_x_ = 65536 - 1;
//       const size_t max_batchs_yz_ = 65536 - 1;
  
//       BatchConfig1D dispatch_dim0_;
//       BatchConfig1D dispatch_dim2_;
//       WorkItemDispatch local_size_;
//       WorkGroupDispatch wg_dispatch_;
  
//     public:
//       sycl::event operator()(sycl::queue &Q, real_t *fdist_dev,
//                              const AdvectionSolver &solver) override;
  
//       AdaptiveWg() = delete;
  
//       AdaptiveWg(const AdvectionSolver &solver, sycl::queue q) {
//           const auto n0 = solver.params.n0;
//           const auto n1 = solver.params.n1;
//           const auto n2 = solver.params.n2;
  
//           dispatch_dim0_ = init_1d_blocking(n0, max_batchs_x_);
//           dispatch_dim2_ = init_1d_blocking(n2, max_batchs_yz_);
  
//           // SYCL query returns the size in bytes
//           auto max_elem_local_mem =
//               q.get_device().get_info<sycl::info::device::local_mem_size>() /
//               sizeof(real_t);
  
//           local_size_.set_ideal_sizes(solver.params.pref_wg_size, n0, n1, n2);
//           local_size_.adjust_sizes_mem_limit(max_elem_local_mem, n1);
  
//           wg_dispatch_.s0_ = solver.params.seq_size0;
//           wg_dispatch_.s2_ = solver.params.seq_size2;
  
//           // TODO: this line is overriden inside the kernel!!! useless
//           // wg_dispatch_.set_num_work_groups(n0, n2, dispatch_dim0_.n_batch_,
//           //                                  dispatch_dim2_.n_batch_,
//           //                                  local_size_.w0_, local_size_.w2_);
//       }
//   };

// ==========================================
// ==========================================
template <MemorySpace MemType, class MySolver, BkmaImpl Impl>
inline std::enable_if_t<Impl == BkmaImpl::AdaptiveWg, sycl::event>
submit_kernels(sycl::queue &Q, span3d_t data, const MySolver &solver,
               const size_t b0_size, const size_t b0_offset,
               const size_t b2_size, const size_t b2_offset,
               const size_t orig_w0, const size_t w1, const size_t orig_w2,
               WorkGroupDispatch wg_dispatch,
               span3d_t global_scratch = span3d_t{}) {

    const auto w0 = sycl::min(orig_w0, b0_size);
    const auto w2 = sycl::min(orig_w2, b2_size);

    wg_dispatch.set_num_work_groups(b0_size, b2_size, 1, 1, w0, w2);
    auto const seq_size0 = wg_dispatch.s0_;
    auto const seq_size2 = wg_dispatch.s2_;
    auto const g0 = wg_dispatch.g0_;
    auto const g2 = wg_dispatch.g2_;

    const sycl::range<3> global_size(g0 * w0, w1, g2 * w2);
    const sycl::range<3> local_size(w0, w1, w2);

    auto n0 = data.extent(0);
    auto n1 = data.extent(1);
    auto n2 = data.extent(2);

    const auto window = solver.window();
    const auto nw = n1 - (window-1);

    return Q.submit([&](sycl::handler &cgh) {
        auto mallocator = [&]() {
            if constexpr (MemType == MemorySpace::Local) {
                sycl::range<3> acc_range(w0, w2, nw);
                return MemAllocator<MemType>(acc_range, cgh);
            } else {
                extents_t ext(b0_size, n2, n1);
                return MemAllocator<MemType>(global_scratch);
            }
        }();

        cgh.parallel_for(
            sycl::nd_range<3>{global_size, local_size},
            [=](auto itm) {
                span3d_t scr(mallocator.get_pointer(),
                             mallocator.get_extents());

                const auto i1 = itm.get_local_id(1);
                const auto local_i0 = compute_index<MemType>(itm, 0);
                const auto local_i2 = compute_index<MemType>(itm, 2);

                auto scratch_slice = std::experimental::submdspan(
                    scr, local_i0, local_i2, std::experimental::full_extent);

                const auto start_idx0 = b0_offset + itm.get_global_id(0);
                const auto stop_idx0 = sycl::min(n0, start_idx0 + b0_size);
                for (size_t global_i0 = start_idx0; global_i0 < stop_idx0;
                     global_i0 += g0 * w0) {

                    const auto start_idx2 = b2_offset + itm.get_global_id(2);
                    const auto stop_idx2 = sycl::min(n2, start_idx2 + b2_size);
                    for (size_t global_i2 = start_idx2; global_i2 < stop_idx2;
                         global_i2 += g2 * w2) {

                        auto data_slice = std::experimental::submdspan(
                            data, global_i0, std::experimental::full_extent,
                            global_i2);

                        for (int ii1 = i1; ii1 < n1; ii1 += w1) {
                            scratch_slice(ii1) = data_slice(ii1);
                        }

                        sycl::group_barrier(itm.get_group());
                        
                        for (int iw = i1; iw < nw; iw += w1) {
                            auto const ii1 = iw + window - 1;
                            data_slice(iw) = solver(scratch_slice, global_i0,
                                                    ii1, global_i2);
                        }
                    }   // end for ii2
                }   // end for ii0
            }       // end lambda in parallel_for
        );          // end parallel_for nd_range
    });      // end Q.submit
}   // end submit_kernels
