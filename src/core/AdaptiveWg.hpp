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
//           L0  :| 2**32-1 | 2**32-1 | (compile with
//           -fno-sycl-query-fit-in-int) CPU :        a lot             */
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

//           local_size_.set_ideal_sizes(solver.params.pref_wg_size, n0, n1,
//           n2); local_size_.adjust_sizes_mem_limit(max_elem_local_mem, n1);

//           wg_dispatch_.s0_ = solver.params.seq_size0;
//           wg_dispatch_.s2_ = solver.params.seq_size2;

//           // TODO: this line is overriden inside the kernel!!! useless
//           // wg_dispatch_.set_num_work_groups(n0, n2,
//           dispatch_dim0_.n_batch_,
//           //                                  dispatch_dim2_.n_batch_,
//           //                                  local_size_.w0_,
//           local_size_.w2_);
//       }
//   };

// ==========================================
// ==========================================
template <MemorySpace MemType, class MySolver, BkmaImpl Impl>
inline std::enable_if_t<Impl == BkmaImpl::AdaptiveWg, sycl::event>
submit_kernels(sycl::queue &Q, span3d_t data, const MySolver &solver,
               BkmaOptimParams &optim_params,
               span3d_t global_scratch = span3d_t{}) {
    // // Aliases
    // const auto& orig_w0 = optim_params.w0;
    // const auto& orig_w2 = optim_params.w2;
    // const auto& b0_size = optim_params.dispatch_d0.batch_size_; //TODO: this is wrong, depends on which batch we are in
    // const auto& b2_size = optim_params.dispatch_d2.batch_size_; //TODO: clear this as well, this should be computed in calling function

    // const auto w0 = sycl::min(orig_w0, b0_size);
    // const auto w2 = sycl::min(orig_w2, b2_size);

    // optim_params.wg_dispatch.set_num_work_groups(b0_size, b2_size, 1, 1, w0, w2);
    // auto const seq_size0 = optim_params.wg_dispatch.s0_;
    // auto const seq_size2 = optim_params.wg_dispatch.s2_;
    // auto const g0 = optim_params.wg_dispatch.g0_;
    // auto const g2 = optim_params.wg_dispatch.g2_;
    // const sycl::range<3> global_size(g0 * w0, w1, g2 * w2);
    // const sycl::range<3> local_size(w0, w1, w2);

    const auto &n0 = data.extent(0);
    const auto &nw = data.extent(1); //advection: nw = n1
    const auto &n2 = data.extent(2);
 
    const auto& simd_size         = optim_params.simd_size;
    const auto& nSubgroups_Local  = optim_params.nSubgroups_Local;
    const auto& nSubgroups_Global = optim_params.nSubgroups_Global;
    const auto& seqSize_Local     = optim_params.seqSize_Local;
    const auto& seqSize_Global    = optim_params.seqSize_Global;

    const auto N_Subgroups = nSubgroups_Local + nSubgroups_Global;
    const auto Total_SeqSize = (nSubgroups_Local * seqSize_Local +
                                nSubgroups_Global * seqSize_Global) /
                               N_Subgroups;

    //==========================================
    sycl::range<1> global_size(
        (n0 / Total_SeqSize)*
        simd_size*
        n2
    );

    sycl::range<1> local_size(
        N_Subgroups*
        simd_size*
        1
    );

    //  === sizes ===
    // global group size (how many groups in the global grid)
    const size_t global_g_size_0 = (n0 / Total_SeqSize)/N_Subgroups;
    const size_t global_g_size_1 = 1;
    const size_t global_g_size_2 = n2; //ndrage.get_group_range

    // group data size (how many data elements in a group)
    const size_t group_d_size_0 = N_Subgroups * Total_SeqSize;
    const size_t group_d_size_1 = nw;
    const size_t group_d_size_2 = 1;

    // group item size (how many work-items in a work-group)
    const size_t group_i_size_0 = N_Subgroups;
    const size_t group_i_size_1 = simd_size;
    const size_t group_i_size_2 = 1;

    // item data size (how many data elements processed per item)
    const size_t item_d_size_1 = nw / simd_size;
    const size_t item_d_size_2 = 1;

    auto const ndra = sycl::nd_range<1>{global_size, local_size};
    return Q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<real_t, 3> local_scratch({nSubgroups_Local, 1, nw}, cgh);

        // [[sycl::reqd_sub_group_size(32)]]
        cgh.parallel_for(
            ndra,
            [=](auto itm) {
                const span3d_t local_span(local_scratch.GET_POINTER(), nSubgroups_Local, 1, nw);
                const bool is_local = itm.get_sub_group().get_group_id() >= nSubgroups_Global;

                const span3d_t& scratch_slice = is_local ? local_span : global_scratch;

                // === indexes ===
                // within the sub-group, the index of the work-item
                const size_t sg_i0 = 0;
                const size_t sg_i1 = itm.get_sub_group().get_local_id();//itm.get_local_id(0) % simd_size;
                const size_t sg_i2 = 0;

                // within the group, the index of the sub-group
                const size_t group_sg0 = itm.get_sub_group().get_group_id();
                const size_t group_sg1 = 0;
                const size_t group_sg2 = 0;

                // within the group, the index of the item
                const size_t group_i0 = sg_i0 + group_sg0;
                const size_t group_i1 = sg_i1 + group_sg1;
                const size_t group_i2 = sg_i2 + group_sg2;

                // within the global group grid, the index of the group
                // const size_t global_g0 = itm.get_group().get_group_id(0) % global_g_size_0;
                const size_t global_g0 = itm.get_group().get_group_id(0) / (global_g_size_2*global_g_size_1);
                const size_t global_g1 = 0;
                // const size_t global_g2 = itm.get_group().get_group_id(0) / (global_g_size_0*global_g_size_1);
                const size_t global_g2 = itm.get_group().get_group_id(0) % global_g_size_2;

                // within the global grid, the index of the work-item
                const size_t global_i0 = global_g0 * group_d_size_0 + group_i0;
                const size_t global_i1 = global_g1 * group_d_size_1 + group_i1;
                const size_t global_i2 = global_g2 * group_d_size_2 + group_i2;

                // corresponding index of the data element
                size_t global_d2 = global_i2;

                // correponding index in the scratch slice (local or global)
                const size_t scratch_i2 = is_local ? 0 : global_d2;
                const size_t item_d_size_0 = is_local ? seqSize_Local : seqSize_Global;

                for (int item_d0 = 0; item_d0 < item_d_size_0; ++item_d0) {
                    size_t global_d0 = global_i0 + item_d0 * nSubgroups_Local; //TODO!! attention ça ça ne marche que si nSubgroups_Global=1
 
                    const size_t scratch_i0 = is_local ? group_sg0 - nSubgroups_Global : global_g0 + group_sg0;

                    auto data_slice = std::experimental::submdspan(
                        data, global_d0, std::experimental::full_extent,
                        global_d2);

                    for (int item_d1 = 0; item_d1 < item_d_size_1; ++item_d1) {
                        size_t global_d1 = global_i1+item_d1 * group_i_size_1;

                        scratch_slice(scratch_i0, scratch_i2, global_d1) =
                            solver(data_slice, global_d0, global_d1, global_d2);
                    }

                    sycl::group_barrier(itm.get_sub_group());

                    for (int item_d1 = 0; item_d1 < item_d_size_1; ++item_d1) {
                        size_t global_d1 = global_i1+item_d1 * group_i_size_1;
                        data_slice(global_d1) =
                            scratch_slice(scratch_i0, scratch_i2, global_d1);
                    }

                    sycl::group_barrier(itm.get_sub_group());
                }
            }   // end lambda in parallel_for
        );      // end parallel_for nd_range
    });         // end Q.submit
}   // end submit_kernels

    // std::cout << "SIMD Size: " << simd_size << std::endl;
    // std::cout << "N_SUBGROUPS: " << N_SUBGROUPS << std::endl;
    // std::cout << "Total_SeqSize: " << Total_SeqSize << std::endl;
    // std::cout << "global_size: (" << n0 << ", " << simd_size
    //           << ", " << n2/Total_SeqSize << ")" << std::endl;
    // std::cout << "local_size: (" << 1 << ", " << simd_size
    //           << ", " << N_SUBGROUPS << ")" << std::endl;
//===========================
// size_t block_id = itm.get_group().get_group_id(0) % (n2/N_SUBGROUPS);
// size_t i2 = block_id*N_SUBGROUPS + subgroup_id;

// auto kernel_simd_size = itm.get_sub_group().get_local_range();
// // auto n_subgroups_in_wg = itm.get_sub_group().get_group_range();
//     static const __attribute__((opencl_constant)) char FMT[] =
//     "kernel_simd_size: %d\n";
// sycl::ext::oneapi::experimental::printf(FMT, kernel_simd_size[0]);
//==========================
// if (global_d2 >= n2 || global_d2 < 0){
//     static const
//         __attribute__((opencl_constant)) char FMT[] =
//             "global_d2: %d\n";
//     sycl::ext::oneapi::experimental::printf(
//         FMT, global_d2);
// }
// if (global_d2 >= n2) continue;
//===========================
