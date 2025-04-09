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

    // const sycl::range<3> global_size(g0 * w0, w1, g2 * w2);
    // const sycl::range<3> local_size(w0, w1, w2);

    auto n0 = data.extent(0);
    auto n1 = data.extent(1);
    auto n2 = data.extent(2);

    const auto window = solver.window();
    const auto nw = n1 - (window-1);

    //==========================================
    auto const SEQ_SIZE_SUBGROUPS = 4;
    size_t simd_size =
        Q.get_device().get_info<sycl::info::device::preferred_vector_width_int>();
    constexpr auto N_SUBGROUPS = 2; 
    
    sycl::range<1> global_size(
        n0*
        simd_size* //n1
        (n2/SEQ_SIZE_SUBGROUPS/N_SUBGROUPS)
    );

    //w0 = 1
    //w1 = simd_size
    //w2 = 1

    auto const ndra = sycl::nd_range<1>{global_size, {N_SUBGROUPS*simd_size}};

    return Q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<real_t, 1> local_scratch(nw*N_SUBGROUPS, cgh);

        cgh.parallel_for(
            ndra,
            [=](auto itm) {
                size_t subgroup_id = itm.get_sub_group().get_group_id();
                span1d_t scratch_slice(local_scratch.GET_POINTER(), nw*N_SUBGROUPS);

                const auto linear_id = itm.get_global_id(0);

                size_t n2_local = n2 / SEQ_SIZE_SUBGROUPS / N_SUBGROUPS;
                // size_t n1 = simd_size;
                
                size_t i0 = linear_id / (simd_size * n2_local);
                size_t res = linear_id % (simd_size * n2_local);
                size_t i1 = res % simd_size;

                size_t block_id = res / simd_size;
                size_t i2 = (block_id * N_SUBGROUPS + subgroup_id) * SEQ_SIZE_SUBGROUPS;

                for (int s = 0; s < SEQ_SIZE_SUBGROUPS; ++s) {
                    size_t global_i2 = i2 + s;
                    
                    auto data_slice = std::experimental::submdspan(
                        data, i0, std::experimental::full_extent,
                        global_i2);

                    for (int ii1 = i1; ii1 < n1; ii1 += simd_size) {
                        auto const iw = ii1 - (window - 1);
                        if(iw >= 0)
                            scratch_slice(subgroup_id * nw + iw) = solver(
                                data_slice, i0, ii1, global_i2);
                    }

                    sycl::group_barrier(itm.get_sub_group()); // is a __syncwarp();

                    // sycl::group_barrier(itm.get_group());

                    for (int iw = i1; iw < nw; iw += simd_size) {
                        data_slice(subgroup_id * nw + iw) = scratch_slice(subgroup_id * nw + iw);
                    }
                }

            }       // end lambda in parallel_for
        );          // end parallel_for nd_range
    });      // end Q.submit
}   // end submit_kernels
