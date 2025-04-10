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
    const auto nw = n1;// - (window-1);

    //==========================================

    auto sg_sizes = Q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    if(sg_sizes.size() > 1){
        std::cout << "WARNING, MORE THAN ONE SUBGROUP SIZE AVAILABLE" << std::endl;
    }

    const int simd_size = sg_sizes[0];
    const auto SEQ_SIZE_SUBGROUPS = 1;
    constexpr auto N_SUBGROUPS = 2;
    
    sycl::range<1> global_size(
        n0*
        simd_size* //n1
        (n2/SEQ_SIZE_SUBGROUPS)//*N_SUBGROUPS
    );

    sycl::range<1> local_size(
        1*
        simd_size*
        N_SUBGROUPS
    );

    std::cout << "SIMD Size: " << simd_size << std::endl;
    std::cout << "N_SUBGROUPS: " << N_SUBGROUPS << std::endl;
    std::cout << "SEQ_SIZE_SUBGROUPS: " << SEQ_SIZE_SUBGROUPS << std::endl;
    std::cout << "global_size: (" << n0 << ", " << simd_size
              << ", " << n2/SEQ_SIZE_SUBGROUPS << ")" << std::endl;
    std::cout << "local_size: (" << 1 << ", " << simd_size
              << ", " << N_SUBGROUPS << ")" << std::endl;

    auto const ndra = sycl::nd_range<1>{global_size, local_size};
    return Q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<real_t, 2> local_scratch({nw, N_SUBGROUPS}, cgh);
    
        cgh.parallel_for(
            ndra,
            [=](auto itm) [[sycl::reqd_sub_group_size(16)]] {
                span2d_t scratch_slice(local_scratch.GET_POINTER(), nw, N_SUBGROUPS);

                //  === sizes ===
                // global
                const size_t global_d_size_0 = n0;
                const size_t global_d_size_1 = n1;
                const size_t global_d_size_2 = n2;

                const size_t global_g_size_0 = n0;
                const size_t global_g_size_1 = 1;
                const size_t global_g_size_2 = n2/(N_SUBGROUPS*SEQ_SIZE_SUBGROUPS);

                const size_t global_i_size_0 = n0;
                const size_t global_i_size_1 = simd_size;
                const size_t global_i_size_2 = n2/SEQ_SIZE_SUBGROUPS;

                //group
                const size_t group_d_size_0 = 1;
                const size_t group_d_size_1 = n1;
                const size_t group_d_size_2 = N_SUBGROUPS*SEQ_SIZE_SUBGROUPS;

                const size_t group_i_size_0 = 1;
                const size_t group_i_size_1 = simd_size;
                const size_t group_i_size_2 = N_SUBGROUPS;

                const size_t group_sg_size_0 = 1;
                const size_t group_sg_size_1 = 1;
                const size_t group_sg_size_2 = N_SUBGROUPS;

                //subgroup
                const size_t subgroup_d_size_0 = 1;
                const size_t subgroup_d_size_1 = n1;
                const size_t subgroup_d_size_2 = SEQ_SIZE_SUBGROUPS;

                const size_t subgroup_i_size_0 = 1;
                const size_t subgroup_i_size_1 = simd_size;
                const size_t subgroup_i_size_2 = 1;

                //item
                const size_t item_d_size_0 = 1;
                const size_t item_d_size_1 = n1/simd_size;
                const size_t item_d_size_2 = SEQ_SIZE_SUBGROUPS;

                // === indexes ===
                const size_t sg_i0 = 0;
                const size_t sg_i1 = itm.get_local_id(0) % simd_size;
                const size_t sg_i2 = 0;
                
                const size_t group_sg0 = 0;
                const size_t group_sg1 = 0;
                const size_t group_sg2 = itm.get_sub_group().get_group_id();

                const size_t group_i0 = 0;
                const size_t group_i1 = sg_i1;
                const size_t group_i2 = group_sg2;

                const size_t global_g0 = itm.get_group().get_group_id(0) / (global_g_size_2*global_g_size_1);
                const size_t global_g1 = 0;
                const size_t global_g2 = itm.get_group().get_group_id(0) % global_g_size_2;

                const size_t global_i0 = itm.get_global_id(0) / (global_i_size_1*global_i_size_2);
                const size_t global_i1 = itm.get_local_id(0) % simd_size;
                const size_t global_i2 = global_g2 * global_g_size_2 + group_sg2;


                // size_t block_id = itm.get_group().get_group_id(0) % (n2/N_SUBGROUPS);
                // size_t i2 = block_id*N_SUBGROUPS + subgroup_id;

                // size_t kernel_simd_size = itm.get_sub_group().get_local_size();
                
                size_t i0 = global_i0;
                size_t i1 = global_i1;
                size_t i2 = global_i2;

                for (int s = 0; s < SEQ_SIZE_SUBGROUPS; ++s) {
                    sycl::group_barrier(itm.get_sub_group());
                    size_t ii2 = i2 + s;// * n2_local;
    
                    auto data_slice = std::experimental::submdspan(
                        data, i0, std::experimental::full_extent, ii2);
    
                    for (int ii1 = i1; ii1 < n1; ii1 += simd_size) {
                        auto const iw = ii1 - (window - 1);
                        if (iw >= 0)
                            scratch_slice(iw, group_sg2) = solver(data_slice, i0, ii1, ii2);
                    }
    
                    sycl::group_barrier(itm.get_sub_group());
    
                    for (int ii1 = i1; ii1 < n1; ii1 += simd_size) {
                        auto const iw = ii1 - (window - 1);
                        if (iw >= 0)
                            data_slice(iw) =
                                scratch_slice(iw, group_sg2);

                        static const __attribute__((opencl_constant)) char FMT[] =
                        "linear: %d, local: %d, i0: %d, iw: %d, i2: %d, "
                        "subgroup_id: %d, block_id: %d, group_id: %d\n";
                    sycl::ext::oneapi::experimental::printf(
                        FMT, itm.get_global_id(0), itm.get_local_id(0), i0, iw, ii2, group_sg2, global_g2, itm.get_group().get_group_id(0));
                    }
                }
            } // end lambda in parallel_for
        );    // end parallel_for nd_range
    });        // end Q.submit
}   // end submit_kernels
